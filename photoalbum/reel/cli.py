"""make-reel: short vertical videos (TikTok, Reels, Shorts) from an event folder.

    make-reel "Trip"                    one new video, a new random seed
    make-reel "Trip" --variants 4       four different previews + index.html
    make-reel "Trip" --seed 48213 --final    re-make a favourite at full quality
    make-reel "Trip" 30s punchy film    say it in words

Every run uses a seed. Different seeds give different edits; the same seed
always gives the same edit, written down in a .recipe.json beside the video.
"""
import argparse
import json
import os
import re
import sys
import time

from .. import cli as album_cli, colour, cover, enhance, faces, ingest, places
from ..deps import report as deps_report, np
from . import editor, ffm, moments, render, sheet, sources, beats, thumbs

AUDIO_EXT = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus"}
WORDS_PACING = set(editor.PACING)
WORDS_FX = set(editor.TRANSITIONS)
WORDS_ORDER = set(editor.ORDERS)
WORDS_EMPHASIS = set(editor.EMPHASES)


def _music_files(event_dir, root):
    out = []
    for folder in (event_dir, os.path.join(root, "music")):
        if os.path.isdir(folder):
            for fn in sorted(os.listdir(folder)):
                if os.path.splitext(fn)[1].lower() in AUDIO_EXT and not fn.startswith("."):
                    out.append(os.path.join(folder, fn))
    return out


def _music_analysis(path, cache_dir):
    key = "%s|%d|%d" % (os.path.abspath(path), os.path.getsize(path), int(os.path.getmtime(path)))
    import hashlib
    cpath = os.path.join(cache_dir, "music_%s.json" % hashlib.sha1(key.encode()).hexdigest()[:16])
    if os.path.exists(cpath):
        with open(cpath) as fh:
            return json.load(fh)
    a = beats.analyse(path)
    if a:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cpath, "w") as fh:
            json.dump(a, fh)
    return a


def _length(word):
    """'90s', '3m', '2min', '1m30s', '2.5m' -> seconds, else None."""
    m = re.fullmatch(r"(?:(\d+(?:\.\d+)?)\s*(?:m|min|mins))?(?:(\d+)\s*s(?:ec)?)?", word)
    if not m or not (m.group(1) or m.group(2)):
        return None
    secs = float(m.group(1) or 0) * 60 + float(m.group(2) or 0)
    return int(round(secs)) if 5 <= secs <= 1800 else None


def parse_words(words):
    """Plain-language options: 30s, punchy, film, kinetic, arc, nomusic, final."""
    opts, unknown = {}, []
    for w in words:
        low = w.lower()
        length = _length(low)
        if length:
            opts["duration"] = length
        elif low in WORDS_PACING:
            opts["pacing"] = low
        elif low in WORDS_FX:
            opts["transitions"] = low
        elif low in WORDS_EMPHASIS:
            opts["emphasis"] = low
        elif low in WORDS_ORDER:
            opts["order"] = low
        elif low in enhance.LOOKS or low == "auto":
            opts["look"] = low
        elif low in ("nomusic", "no-music"):
            opts["no_music"] = True
        elif low in ("final", "preview"):
            opts["profile"] = low
        else:
            unknown.append(w)
    return opts, unknown


def _seed_for(base, i):
    return base if i == 0 else (base * 1000003 + i * 7919) & 0x7fffffff


def _differs(a, b):
    """Two variants are worth comparing if they differ in what they put
    first (their emphasis), or in at least two other ways: pacing, colour,
    transitions, length, or largely different moments."""
    if a["style"].get("emphasis") != b["style"].get("emphasis"):
        return True
    n = sum(a["style"].get(k) != b["style"].get(k)
            for k in ("pacing", "family", "transitions"))
    n += abs(a["style"]["duration"] - b["style"]["duration"]) >= 6
    sa = {(g["src"], g.get("in")) for g in a["segments"]}
    sb = {(g["src"], g.get("in")) for g in b["segments"]}
    n += len(sa & sb) / float(max(1, len(sa | sb))) < 0.6
    return n >= 2


def make(event, inbox, outbox, root, opts, seed, variants=1, recipe_file=None,
         dry_run=False, verbose=True):
    event_dir = os.path.join(inbox, event)
    out_dir = os.path.join(outbox, event, "reels")
    # Analysis lives outside outbox/, so clearing old videos from there never
    # throws away the (slow) analysis of the footage.
    cache = os.path.join(root, "cache", "reels", event)
    work = os.path.join(out_dir, "work")
    os.makedirs(out_dir, exist_ok=True)
    keep, skip = sources.read_lists(event_dir)

    prog = album_cli.Progress("watching", 1, verbose)
    dropped = []
    clips = sources.clips(event_dir, cache, skip=skip, dropped=dropped,
                          progress=lambda d, n, name: (setattr(prog, "total", n),
                                                       prog.step(1, name)))
    prog.done()
    for name, why in dropped:
        print("  skipped %s - %s" % (name, why), file=sys.stderr)
    photos = sources.photos(event_dir, os.path.join(outbox, event, "album.json"),
                            cache, skip=skip, keep=keep)
    if not clips and not photos:
        print("  nothing to use: no readable videos or photos", file=sys.stderr)
        return 1
    if verbose:
        print("  %d clips, %d photos" % (len(clips), len(photos)))
    sources.chapters(clips, photos)
    colour.balance(list(clips) + list(photos))
    found = moments.collect(clips, photos)
    # Hand-shake measured at full frame rate for every clip moment (cached).
    shake_path = os.path.join(cache, "shake.json")
    shake = {}
    if os.path.exists(shake_path):
        with open(shake_path) as fh:
            shake = json.load(fh)
    if moments.measure_shake(found, shake):
        os.makedirs(cache, exist_ok=True)
        with open(shake_path, "w") as fh:
            json.dump(shake, fh)
    shaky = sum(1 for m in found if m.shake is not None and m.shake > 0.02)
    if verbose and shaky:
        print("  %d shaky moments left out, %d to be stabilised" % (
            shaky, sum(1 for m in found if m.shake is not None and 0.01 < m.shake <= 0.02)))
    by_rel = {c.rel: c for c in clips}
    by_rel.update({p.rel: p for p in photos})

    music_files = [] if opts.get("no_music") else (
        [opts["music"]] if opts.get("music") else _music_files(event_dir, root))
    title = cover.display_title(album_cli._pretty_title(event))
    subtitle = album_cli._fmt_range(list(clips) + list(photos))
    # Where it was, when the footage knows: the town most of it was shot in.
    if places.available():
        names = []
        for item in list(clips) + list(photos):
            if getattr(item, "gps", None):
                hit = places.nearest(*item.gps)
                if hit:
                    names.append(hit[0])
        if names:
            town = max(sorted(set(names)), key=names.count)
            if town.lower() not in title.lower():
                subtitle = "%s  ·  %s" % (town, subtitle) if subtitle else town

    frames = {}

    def frame_at(clip, t):
        # A crop is chosen from the nearest keyframe at a modest size: the
        # cropper works on a 220-pixel map, and a full 4K decode per cut made
        # choosing a 3-minute edit take minutes.
        key = (clip.path, round(t * 2) / 2.0)
        if key not in frames:
            size = ffm.fit_size(clip.width, clip.height, 640)
            frames[key] = next(ffm.decode_frames(clip.path, max(0.0, t - 0.5), 1.5, 4, size,
                                                 hdr=clip.hdr, keyframes=True), None)
            if frames[key] is None:
                frames[key] = next(ffm.decode_frames(clip.path, max(0.0, t - 0.05), 0.4, 4,
                                                     size, hdr=clip.hdr), None)
        return frames[key]

    def prefetch(pairs):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda ct: frame_at(*ct), pairs))

    frame_at.prefetch = prefetch

    photo_cache = {}

    def photo_rgb(photo):
        if photo.path not in photo_cache:
            rgb = np.asarray(ingest.load(photo, max_long_edge=1024).convert("RGB"))
            boxes = list(getattr(photo, "face_boxes", None) or [])
            if not boxes:
                h, w = rgb.shape[:2]
                boxes = faces.summarise(faces.detect(rgb), w, h)["boxes"]
            photo_cache[photo.path] = (rgb, boxes)
        return photo_cache[photo.path]

    profile = opts.get("profile") or ("preview" if variants > 1 else "final")
    made = []
    for i in range(variants):
        if recipe_file:
            with open(recipe_file) as fh:
                recipe = json.load(fh)
            s = recipe["seed"]
        else:
            s = _seed_for(seed, i)
            for _attempt in range(25):
                # Cheap check first: a variant should put something else first.
                used = {m[1]["style"].get("emphasis") for m in made}
                if len(used) < len(editor.EMPHASES) and not opts.get("emphasis") and \
                        editor.choose_style(s, [], opts)["emphasis"] in used:
                    s = (s * 31 + 17) & 0x7fffffff
                    continue
                track = None
                if music_files:
                    path = editor.stream(s, "music").choice(music_files)
                    a = _music_analysis(path, cache)
                    if a:
                        track = {"path": path, "rel": os.path.relpath(path, root), "analysis": a}
                recipe = editor.build(event, title, subtitle, found, s, opts, music=track,
                                      frame_at=frame_at, photo_rgb=photo_rgb)
                if all(_differs(recipe, m[1]) for m in made) or not made:
                    break
                s = (s * 31 + 17) & 0x7fffffff
        name = "%s-s%d%s" % (event, s, "-preview" if profile == "preview" else "")
        base = os.path.join(out_dir, name)
        with open(base + ".recipe.json", "w") as fh:
            json.dump(recipe, fh, indent=1)
        if verbose:
            print("  seed %d: %s" % (s, editor.summary(recipe)))
            if recipe["audio"]["mode"] != "music" and not any(
                    getattr(by_rel.get(g["src"]), "has_audio", False)
                    for g in recipe["segments"]):
                print("  note: no music and no clip sound - this video is silent. "
                      "Put music files in music/ or the event folder.")
        if not dry_run:
            music_path = None
            if recipe["audio"]["mode"] == "music":
                music_path = os.path.join(root, recipe["audio"]["music"])
            p = album_cli.Progress("render", len(recipe["segments"]), verbose)
            t0 = time.time()
            render.render(recipe, event_dir, by_rel, base + ".mp4", work, profile,
                          music_path=music_path,
                          progress=lambda d, n, nm: (setattr(p, "n", d - 1), p.step(1, nm)))
            p.done()
            render.strip(base + ".mp4", base + ".strip.jpg")
            covers = thumbs.make(recipe, by_rel, base, title, subtitle,
                                 look=recipe["style"].get("look", "natural")
                                 if recipe["style"].get("look") != "auto" else "natural",
                                 cache_dir=cache)
            if verbose:
                print("  -> %s  (%.0fs)" % (base + ".mp4", time.time() - t0))
                for c in covers:
                    print("     %s" % c)
        made.append((s, recipe))
        if recipe_file:
            break
    if not dry_run:
        page = sheet.write(out_dir, event)
        if verbose:
            print("  compare: %s" % page)
    return 0


def main(argv=None):
    root = album_cli._project_root()
    def_in, def_out = album_cli._default_dirs(root)
    ap = argparse.ArgumentParser(prog="make-reel", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("event", nargs="*", help="event folder in inbox/, then optional words "
                    "such as: 30s punchy film kinetic nomusic final")
    ap.add_argument("--seed", type=int, help="re-make the edit with this seed")
    ap.add_argument("--variants", type=int, default=1, help="make N different versions")
    ap.add_argument("--final", action="store_true", help="full 1080x1920 quality")
    ap.add_argument("--preview", action="store_true", help="fast 540x960 preview")
    ap.add_argument("--duration", help="target length: 180, 90s, 3m, 1m30s "
                    "(default 3 minutes, or as long as the footage allows)")
    ap.add_argument("--music", help="use this music file")
    ap.add_argument("--no-music", action="store_true", help="keep the clips' own sound")
    ap.add_argument("--look", choices=sorted(list(enhance.LOOKS) + ["auto"]))
    ap.add_argument("--recipe", help="render this .recipe.json again")
    ap.add_argument("--dry-run", action="store_true", help="choose the edit, do not render")
    ap.add_argument("--list", action="store_true", help="what each event folder holds")
    ap.add_argument("--inbox", default=def_in)
    ap.add_argument("--outbox", default=def_out)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    if not ffm.available():
        print("ffmpeg and ffprobe are needed for reels: sudo apt install ffmpeg",
              file=sys.stderr)
        return 2
    available = sorted(d for d in os.listdir(args.inbox)
                       if os.path.isdir(os.path.join(args.inbox, d)) and not d.startswith("."))
    if args.list:
        from . import probe
        for d in available:
            full = os.path.join(args.inbox, d)
            n_v = len(probe.scan(full))
            n_p = len(ingest.scan(full)[0])
            n_m = len(_music_files(full, root))
            print("  %-40s %3d videos %4d photos %2d music" % (d, n_v, n_p, n_m))
        return 0

    events = [w for w in args.event if w in available]
    opts, unknown = parse_words([w for w in args.event if w not in available])
    if unknown:
        print("not understood: %s" % " ".join(unknown), file=sys.stderr)
        if not events:
            print("available: %s" % ", ".join(available), file=sys.stderr)
            return 2
    if not events:
        ap.print_help()
        return 1
    if args.final:
        opts["profile"] = "final"
    if args.preview:
        opts["profile"] = "preview"
    if args.duration:
        secs = _length(args.duration) or (int(args.duration) if args.duration.isdigit() else None)
        if not secs:
            print("could not read --duration %r (try 180, 90s, 3m)" % args.duration,
                  file=sys.stderr)
            return 2
        opts["duration"] = secs
    if args.music:
        opts["music"] = os.path.abspath(args.music)
    if args.no_music:
        opts["no_music"] = True
    if args.look:
        opts["look"] = args.look

    seed = args.seed if args.seed is not None else int(time.time() * 1000) & 0x7fffffff
    if not args.quiet:
        print("make-reel   |  %s  |  ffmpeg" % deps_report())
    status = 0
    for ev in events:
        if not args.quiet:
            print("\n%s\n%s" % (ev, "-" * max(8, len(ev))))
        status |= make(ev, args.inbox, args.outbox, root, opts, seed,
                       variants=max(1, args.variants), recipe_file=args.recipe,
                       dry_run=args.dry_run, verbose=not args.quiet)
    return status


if __name__ == "__main__":
    sys.exit(main())
