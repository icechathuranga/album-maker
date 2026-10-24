"""The editor: moments + a seed -> a recipe (the edit decision list).

It cuts the way an editor cuts a travel or event film:

  * The event in the order it happened, chapter by chapter. Each chapter
    gets screen time by its weight in the trip (no chapter more than 30%),
    so a long stretch of one activity cannot swallow the mornings, and a
    chapter of photos only becomes a short photo montage.
  * An establishing shot opens, held long enough for the title; the widest
    shot of the last chapter closes, with a fade. When there is too much,
    the middle is thinned by score - the end of the trip is never cut off.
  * A tempo per chapter (busy scenes cut faster, the first and last slower),
    a held "breath" shot every half minute or so, cuts placed just before
    the action, the next shot's sound brought in early (a J-cut).
  * Mostly hard cuts: a dip only where the day changes, a dissolve only
    between two photos; photos drift gently rather than zoom.

Everything that varies between runs is drawn from one seed, each kind of
decision from its own stream, so adding a decision later does not reshuffle
the others. Variants differ in *emphasis* - people, places, every part of
the day equally, or by how much was filmed - as well as pacing and colour.
"""
import math
import random

from ..deps import np
from .. import crop, enhance, places
from . import reframe

FPS = 30
ASPECT = 9.0 / 16.0
LONG_REEL = 60.0          # above this, only calm pacings
DEFAULT_DURATION = 180.0  # seconds; as long as the footage allows, up to this
# Average seconds a cut, by pacing, for short and for long reels. A three-
# minute film wants 60-70 shots of 2.6-3 s; a 30-second reel about 1-1.3 s.
CUT = {"short": {"punchy": 1.1, "steady": 1.6, "cinematic": 2.4},
       "long": {"steady": 2.2, "cinematic": 3.0}}
PACING = CUT["short"]
MAX_CLIP_CUT = {"punchy": 2.5, "steady": 5.0, "cinematic": 6.0}
BEAT_CHOICES = {"punchy": (1, 2, 2), "steady": (2, 2, 4), "cinematic": (4, 4, 8)}
ORDERS = ("story",)       # always capture-time order: a reel tells the event
TRANSITIONS = ("clean", "soft")
EMPHASES = ("people", "places", "balanced", "footage")
FAMILIES = {"true": ("natural", "soft", "crisp", "film"),
            "warm": ("warm", "golden", "natural", "film")}
CHAPTER_CAP = 0.30        # no chapter takes more than this share of the film
SAME_MOMENT = 10.0        # two moments of one clip closer than this repeat
SCENE_GAP = 600.0         # clips of a chapter within 10 minutes are one scene
NEAR_DUP_BITS = 18
BREATH_EVERY = 35.0       # seconds between held shots
BREATH_LEN = 4.0
DAY_GAP = 2 * 3600        # a gap this long, or a new day, gets a dip


def stream(seed, group):
    """An independent random stream for one kind of decision."""
    return random.Random("%s:%s" % (seed, group))


def _frames(seconds):
    return max(1, int(round(seconds * FPS)))


def _hamming(a, b):
    return bin(a ^ b).count("1") if a is not None and b is not None else 64


def _when(m):
    """Capture time of a moment: a photo's own time, a clip's start plus the
    moment's place inside it. Undated items go last, by name."""
    return (m.taken is None, m.taken or 0, m.src.name, m.centre)


# --- style ------------------------------------------------------------------

def choose_style(seed, sources, opts):
    """The big decisions, each overridable from the command line."""
    style = {"order": "story"}
    style["duration"] = float(opts.get("duration") or DEFAULT_DURATION)
    long_reel = style["duration"] >= LONG_REEL
    pacings = sorted(CUT["long"] if long_reel else CUT["short"])
    style["pacing"] = opts.get("pacing") if opts.get("pacing") in pacings \
        else stream(seed, "pacing").choice(pacings)
    style["cut"] = CUT["long" if long_reel else "short"][style["pacing"]]
    # From the seed alone, so `--seed N` always re-makes the same edit;
    # --variants finds seeds whose emphases differ rather than forcing one.
    style["emphasis"] = opts.get("emphasis") or stream(seed, "emphasis").choice(EMPHASES)
    style["transitions"] = opts.get("transitions") if opts.get("transitions") in TRANSITIONS \
        else stream(seed, "fx").choice(TRANSITIONS)
    style["family"] = stream(seed, "family").choice(sorted(FAMILIES))
    style["look"] = opts.get("look") if opts.get("look") not in (None, "auto") else "auto"
    # The random draw is always made, so --no-mono changes nothing else in
    # the edit: the same seed gives the same film, only all in colour.
    mono_draw = stream(seed, "mono").random() < 0.35
    style["mono_chapter"] = style["look"] == "auto" and mono_draw and not opts.get("no_mono")
    style["photo_share"] = {"people": 0.30, "places": 0.15}.get(
        style["emphasis"], round(stream(seed, "photos").uniform(0.15, 0.25), 2))
    return style


def _value(m, style):
    """A moment's worth under this variant's emphasis."""
    v = m.score
    if style["emphasis"] == "people":
        v += 0.15 * m.people
    elif style["emphasis"] == "places" and m.wide:
        v += 0.12
    return v


# --- chapters: screen time and selection -------------------------------------

class _Chapter:
    def __init__(self, seg, moments):
        self.seg = seg
        self.moments = sorted(moments, key=_when)
        self.clips = {id(m.src): m.src for m in moments if m.kind == "clip"}
        self.footage = sum(c.duration for c in self.clips.values())
        self.photos = sum(1 for m in moments if m.kind == "photo")
        self.has_clips = bool(self.clips)
        self.budget = 0.0
        self.tempo = 1.0
        self.look = "natural"


def _budgets(chapters, duration, emphasis):
    """Screen seconds per chapter: by sqrt of how much it holds (every part
    of the day counts), or by footage for the footage emphasis; capped."""
    if emphasis == "footage":
        w = [c.footage + 2.0 * c.photos for c in chapters]
    else:
        # How much the chapter holds, with video counting for what it is: a
        # minute of footage weighs as much as four photos.
        w = [math.sqrt(len(c.moments) + c.footage / 15.0) for c in chapters]
    total = sum(w) or 1.0
    alloc = [duration * x / total for x in w]
    cap = CHAPTER_CAP * duration
    for _ in range(8):
        excess = sum(max(0.0, a - cap) for a in alloc)
        if excess < 0.01:
            break
        free = [i for i, a in enumerate(alloc) if a < cap]
        alloc = [min(a, cap) for a in alloc]
        fw = sum(w[i] for i in free) or 1.0
        for i in free:
            alloc[i] += excess * w[i] / fw
    for c, a in zip(chapters, alloc):
        c.budget = a


def _tempos(chapters):
    """Busy chapters cut faster, calm ones slower; first and last slow."""
    energy = []
    for c in chapters:
        clips_ = [m for m in c.moments if m.kind == "clip"]
        energy.append(np.mean([m.motion + 0.5 * min(1.0, m.loud) for m in clips_])
                      if clips_ else 0.0)
    order = sorted(range(len(chapters)), key=lambda i: energy[i])
    for rank, i in enumerate(order):
        frac = rank / float(max(1, len(chapters) - 1))
        chapters[i].tempo = 1.3 - 0.6 * frac
    if chapters:
        chapters[0].tempo = max(chapters[0].tempo, 1.2)
        chapters[-1].tempo = max(chapters[-1].tempo, 1.2)


def _scenes(chapter):
    """Clips of one chapter shot within ten minutes of each other are one scene."""
    clips_ = sorted(chapter.clips.values(), key=lambda c: (c.taken is None, c.taken or 0))
    scene_of, scene, last = {}, 0, None
    for c in clips_:
        if last is not None and c.taken and last.taken and \
                (c.taken - last.taken).total_seconds() - last.duration > SCENE_GAP:
            scene += 1
        scene_of[id(c)] = scene
        last = c
    secs = {}
    for c in clips_:
        secs[scene_of[id(c)]] = secs.get(scene_of[id(c)], 0.0) + c.duration
    # A long continuous scene can give many shots: one per minute filmed.
    caps = {s: int(min(16, 3 + v // 60)) for s, v in secs.items()}
    return scene_of, caps


def _select(chapter, style, rnd):
    """Moments for one chapter: enough for its budget, varied, no twins."""
    cut = style["cut"] * chapter.tempo
    target = max(1, int(round(chapter.budget / cut)))
    want = int(math.ceil(target * 1.3)) + 1
    jitter = {id(m): rnd.uniform(0.0, 0.2) for m in chapter.moments}
    scene_of, caps = _scenes(chapter)
    max_photos = [want if not chapter.has_clips else int(round(style["photo_share"] * want))]
    chosen, per_scene = [], {}
    max_cut = MAX_CLIP_CUT[style["pacing"]]

    def seconds(m):
        """Roughly how long this moment will hold the screen."""
        if m.kind == "photo":
            return 1.4 if not chapter.has_clips else 2.0
        return min(m.span, max(cut, 1.0), max_cut)

    # Fill the chapter's screen time (with room to thin later). Photos stay
    # within the variant's share while there is footage to use, and fill
    # the rest when a chapter's clips run out.
    goal = chapter.budget * 1.2

    def ok(m):
        if m.kind == "clip":
            if m.shake is not None and m.shake > 0.02:
                return False
            s = scene_of.get(id(m.src), 0)
            if per_scene.get(s, 0) >= caps.get(s, 3):
                return False
        elif sum(1 for c in chosen if c.kind == "photo") >= max_photos[0]:
            return False
        for c in chosen:
            if c.src is not m.src and _hamming(c.hash, m.hash) <= NEAR_DUP_BITS:
                return False
            if c.src is m.src and abs(c.centre - m.centre) < SAME_MOMENT:
                return False
        return True

    # Video first: clip moments fill the chapter's time before any photo is
    # considered, so long recordings are used as fully as they allow.
    ranked = (sorted([m for m in chapter.moments if m.kind == "clip"],
                     key=lambda m: -(_value(m, style) + jitter[id(m)])) +
              sorted([m for m in chapter.moments if m.kind == "photo"],
                     key=lambda m: -(_value(m, style) + jitter[id(m)])))
    for relax in (False, True):
        if relax:
            max_photos[0] = len(chapter.moments)
        for m in ranked:
            if sum(seconds(c) for c in chosen) >= goal:
                break
            if m not in chosen and ok(m):
                chosen.append(m)
                if m.kind == "clip":
                    s = scene_of.get(id(m.src), 0)
                    per_scene[s] = per_scene.get(s, 0) + 1
    # Every chapter shows where it was: include its best wide shot.
    if chosen and not any(m.wide for m in chosen):
        wides = [m for m in chapter.moments if m.wide and m not in chosen
                 and (m.shake is None or m.shake <= 0.02)]
        if wides:
            chosen.remove(min(chosen, key=lambda m: _value(m, style)))
            chosen.append(max(wides, key=lambda m: _value(m, style)))
    return chosen


def _opener(chapter, style):
    """The establishing shot: early in the first chapter, nobody close, calm,
    sharp, not soft 1080p - or the people moment for the people emphasis."""
    early = chapter.moments[:max(1, int(math.ceil(len(chapter.moments) * 0.4)))]
    clips_ = [m for m in early if m.kind == "clip" and not m.low_res
              and (m.shake is None or m.shake <= 0.02)]
    if style["emphasis"] == "people" and clips_:
        return max(clips_, key=lambda m: m.people + 0.3 * m.score)
    calm = [m for m in clips_ if m.faces == 0 and m.motion < 0.3 and m.sharp > 0.5]
    if calm:
        return max(calm, key=lambda m: m.score)
    photos = [m for m in early if m.kind == "photo"]
    if photos:
        return max(photos, key=lambda m: m.score + (0.1 if m.wide else 0.0))
    return max(clips_ or early, key=lambda m: m.score)


def _closer(chapter, style):
    """The widest shot late in the last chapter - a calm way out."""
    late = chapter.moments[-max(1, int(math.ceil(len(chapter.moments) * 0.4))):]
    good = [m for m in late if not m.low_res and (m.shake is None or m.shake <= 0.02)]
    wides = [m for m in good if m.wide]
    pool = wides or good or late
    return max(pool, key=lambda m: m.score)


def _tidy(seq, chapter_of, keep):
    """Drop (never move) shots that break the rhythm in story order: two
    moments of one clip close together, three close-ups in a row, and runs
    of photos or of people-less shots longer than the chapter can justify.

    The limits are proportionate: in a chapter that is mostly photos (a
    morning with no video), runs of five photos are the montage, not a fault.
    """
    in_seq = {}
    for m in seq:
        c = chapter_of[id(m)]
        k = in_seq.setdefault(id(c), [0, 0])
        k[0 if m.kind == "clip" else 1] += 1

    def photo_run_limit(m):
        clips_, photos_ = in_seq[id(chapter_of[id(m)])]
        if photos_ >= 3 * clips_:
            return 10 ** 6          # essentially a photo montage
        return 5 if photos_ > clips_ else 3

    def weakest(run):
        cands = [m for m in run if id(m) not in keep]
        return min(cands, key=lambda m: m.score) if cands else None

    changed = True
    while changed:
        changed = False
        for i in range(1, len(seq)):
            a, b = seq[i - 1], seq[i]
            drop = None
            if a.src is b.src and abs(a.centre - b.centre) < SAME_MOMENT:
                drop = weakest([a, b])
            if drop is None and i >= 2 and all(m.close_up for m in seq[i - 2:i + 1]):
                drop = weakest(seq[i - 1:i])
            # Runs are judged inside one chapter, and only where the chapter
            # has footage: a photo-only chapter is a montage by design.
            ch = chapter_of[id(b)]
            same = lambda run: all(chapter_of[id(m)] is ch for m in run)
            if drop is None and ch.has_clips:
                lim = photo_run_limit(b)
                run = seq[i - lim:i + 1] if i >= lim else []
                if run and same(run) and all(m.kind == "photo" for m in run):
                    drop = weakest(run)
            if drop is None and ch.has_clips and i >= 6:
                run = seq[i - 6:i + 1]
                if same(run) and all(m.faces == 0 for m in run):
                    drop = weakest(run)
            if drop is not None:
                seq.remove(drop)
                c = chapter_of[id(drop)]
                in_seq[id(c)][0 if drop.kind == "clip" else 1] -= 1
                changed = True
                break
    return seq


# --- timing ------------------------------------------------------------------------

def _lengths(seq, chapters_of, style, seed, opener, closer):
    """Seconds (and playback speed) for every shot, before any music grid."""
    rnd = stream(seed, "lengths")
    cut = style["cut"]
    max_cut = MAX_CLIP_CUT[style["pacing"]]
    out, since_breath, slowmo_used = [], 0.0, False
    for m in seq:
        ch = chapters_of[id(m)]
        speed = 1.0
        if m is opener:
            d = rnd.uniform(4.0, 6.0)
        elif m is closer:
            d = rnd.uniform(4.0, 5.0)
        elif m.kind == "photo":
            d = rnd.uniform(1.2, 1.6) if not ch.has_clips else rnd.uniform(1.6, 2.4)
        else:
            d = cut * ch.tempo * rnd.uniform(0.85, 1.15)
            if since_breath >= BREATH_EVERY and m.span >= BREATH_LEN and m.motion < 0.4:
                d = BREATH_LEN + rnd.uniform(0.0, 1.0)
            # A long, empty stretch can run at double speed; one fast action
            # moment from 50-60 fps footage can run at half speed.
            elif m.faces == 0 and m.motion < 0.05 and m.span >= 2 * d + 1:
                speed = 2.0
            elif (not slowmo_used and m.tag == "action" and m.src.fps >= 50
                  and style["pacing"] != "punchy"):
                speed, d, slowmo_used = 0.5, min(d, 2.0), True
            d = min(d, max_cut)
        if m.kind == "clip":
            d = min(d, m.span / speed)
        since_breath = 0.0 if d >= BREATH_LEN else since_breath + d
        out.append([m, d, speed])

    # Too long: thin the middle by score, never the opener, closer, or a
    # chapter's last shot - the end of the trip must stay.
    total = sum(x[1] for x in out)
    while total > style["duration"] + 0.5:
        counts = {}
        for x in out:
            counts[id(chapters_of[id(x[0])])] = counts.get(id(chapters_of[id(x[0])]), 0) + 1
        cands = [x for x in out if x[0] is not opener and x[0] is not closer
                 and counts[id(chapters_of[id(x[0])])] > 1]
        if not cands:
            break
        drop = min(cands, key=lambda x: x[0].score)
        out.remove(drop)
        total -= drop[1]
    # Too short: let clips run longer, then photos, but no longer than the
    # pacing allows - the film should fill its length (a whole song) if the
    # footage can, without any shot outstaying its welcome.
    room = style["duration"] - total
    photo_max = 3.5 if style["pacing"] == "cinematic" else 3.0
    for kind in ("clip", "photo"):
        while room > 0.2:
            grew = 0.0
            for x in out:
                m, d, sp = x
                if m.kind != kind or m is opener or m is closer:
                    continue
                cap = (min(m.span / sp, max_cut) if kind == "clip" else photo_max)
                step = min(0.5, cap - d, room - grew)
                if step > 0:
                    x[1] += step
                    grew += step
            if grew <= 0:
                break
            room -= grew
    return out


REUSE_GAP = 1.0     # seconds of clip that must separate two shots of it


def _start(m, d, sp):
    """Where in the clip a shot starts: just before the action."""
    need = d * sp
    a = m.action if m.action is not None else m.centre
    return min(max(a - 0.4, m.lo), max(m.lo, m.hi - need))


def _no_reuse(items, keep):
    """Never show the same footage twice. Each clip shot claims the stretch
    of the clip it plays (padded a little for later beat-fitting); a shot
    that would overlap one already placed moves to its own centre, and is
    dropped if that still overlaps."""
    used, out = {}, []
    for x in items:
        m, d, sp = x
        if m.kind != "clip":
            out.append(x)
            continue
        need = d * sp * 1.25 + REUSE_GAP
        spans = used.setdefault(id(m.src), [])

        def free(a):
            return all(a + need <= s0 or a >= s1 for s0, s1 in spans)

        start = _start(m, d, sp)
        if not free(start):
            alt = min(max(m.centre - need / 2, m.lo), max(m.lo, m.hi - d * sp))
            if free(alt):
                m.action = alt + 0.4          # the shot now starts here
                start = alt
            elif id(m) not in keep:
                continue
        spans.append((start, start + need))
        out.append(x)
    return out


def _on_beats(items, style, seed, music):
    """Fit shot lengths to whole beats, the film to whole 16-beat phrases,
    and start the music on a downbeat with room for all of it."""
    beats = music["beats"]
    if len(beats) < 32:
        return [(m, _frames(d) / float(FPS), sp) for m, d, sp in items], None
    period = (beats[-1] - beats[0]) / (len(beats) - 1)
    ks = []
    for m, d, sp in items:
        k = max(1, int(round(d / period)))
        while k > 1 and m.kind == "clip" and k * period * sp > m.span:
            k = 2 if k > 2 else 1
        if m.kind == "clip" and k * period * sp > m.span:
            ks.append(0)            # cannot fill even one beat: skip it
            continue
        ks.append(k)
    keep = [(x, k) for x, k in zip(items, ks) if k]
    total = sum(k for _, k in keep)
    down = music.get("downbeat", 0)
    whole = style.get("whole_song")
    if whole:
        # The whole song from its first downbeat: as many beats as the song
        # has, or as many as the footage can fill.
        target = min(total, len(beats) - 1 - down)
    elif style.get("exact"):
        # A length was asked for: as many whole beats as it holds; the last
        # shot is then trimmed to the exact frame.
        target = int(math.ceil(style["duration"] / period))
    else:
        # Whole phrases: the music then ends where it naturally resolves.
        target = max(16, int(round(total / 16.0)) * 16)
    i = len(keep) - 1
    while total != target and keep:
        (m, d, sp), k = keep[i]
        step = 1 if total < target else -1
        if k + step >= 1 and (m.kind != "clip" or (k + step) * period * sp <= m.span):
            keep[i] = ((m, d, sp), k + step)
            total += step
        else:
            i -= 1
            if i < 0:
                break
            continue
        i = i - 1 if i > 0 else len(keep) - 1
    energy = music.get("energy", [1.0] * len(beats))
    starts = [b for b in range(down, len(beats), 4)
              if b + total < len(beats) and energy[b] >= 0.5 * max(energy)]
    if whole or style.get("from_start"):
        start = down
    else:
        start = stream(seed, "music-start").choice(starts) if starts else down
    if start + total >= len(beats):
        start = max(0, len(beats) - 1 - total)
    t0 = beats[start]
    out, b = [], start
    for (m, d, sp), k in keep:
        b1 = min(b + k, len(beats) - 1)
        f0 = int(round((beats[b] - t0) * FPS))
        f1 = int(round((beats[b1] - t0) * FPS))
        out.append((m, max(1, f1 - f0) / float(FPS), sp))
        b = b1
    fade = 2.0
    if style.get("exact") and out:
        total_s = sum(x[1] for x in out)
        over = total_s - style["duration"]
        if over > 0:
            m, d, sp = out[-1]
            out[-1] = (m, _frames(d - over) / float(FPS), sp)
    if whole and out and b >= len(beats) - 1:
        # The footage reaches the end of the song: hold the last shot through
        # the song's own ending rather than fading it out early.
        tail = max(0.0, music["duration"] - beats[-1] - 0.05)
        m, d, sp = out[-1]
        if m.kind == "photo" or (m.span / sp) >= d + tail:
            out[-1] = (m, (_frames(d + tail)) / float(FPS), sp)
            fade = 0.5
    plan = {"offset": round(t0, 3), "bpm": music["bpm"], "beats": total,
            "fade": fade, "whole_song": bool(whole)}
    return out, plan


# --- framing -----------------------------------------------------------------------

def _norm_box(box, w, h):
    x, y, bw, bh = box
    return [round(x / w, 4), round(y / h, 4), round(bw / w, 4), round(bh / h, 4)]


def _followed(tl, i0, i1):
    """The face of the main subject at each sample in [i0, i1]: the largest
    at first, then whoever is nearest the last position. None if absent."""
    out, last = [], None
    for i in range(i0, i1 + 1):
        boxes = tl["boxes"][i]
        if not boxes:
            out.append(None)
            continue
        big = max(b[2] * b[3] for b in boxes)
        near = [b for b in boxes if b[2] * b[3] >= 0.5 * big]
        if last is None:
            pick = max(near, key=lambda b: b[2] * b[3])
        else:
            pick = min(near, key=lambda b: abs(b[0] + b[2] / 2 - last[0])
                       + abs(b[1] + b[3] / 2 - last[1]))
        last = (pick[0] + pick[2] / 2, pick[1] + pick[3] / 2)
        out.append(pick)
    return out


def span_window(clip, t0, t1, frame):
    """A 9:16 window that keeps the main subject in frame for the whole shot,
    not just its middle. Returns (window, last time the subject is seen)."""
    tl = clip.timeline
    ts = tl["t"]
    i0 = min(range(len(ts)), key=lambda k: abs(ts[k] - t0))
    i1 = min(range(len(ts)), key=lambda k: abs(ts[k] - t1))
    faces = _followed(tl, i0, i1)
    win_w = min(1.0, ASPECT * clip.height / float(clip.width))
    win_h = min(1.0, clip.width / ASPECT / float(clip.height))
    seen = [f for f in faces if f]
    last_seen = None
    if seen:
        idx = [i for i, f in enumerate(faces) if f]
        last_seen = ts[i0 + idx[-1]]
    if not seen:
        if frame is None:
            return [(1 - win_w) / 2, (1 - win_h) / 2, win_w, win_h], None
        h, w = frame.shape[:2]
        return _norm_box(crop.find(frame, ASPECT, face_boxes=[], max_zoom_out=1.0), w, h), None

    def best(axis_len, lo_key, size_key):
        grid = np.linspace(0, 1 - axis_len, 41)
        score = []
        for x in grid:
            s = 0.0
            for f in seen:
                a, l = f[lo_key], f[size_key]
                inside = max(0.0, min(x + axis_len, a + l) - max(x, a)) / max(l, 1e-6)
                centre = abs((a + l / 2) - (x + axis_len / 2)) / axis_len
                s += inside - 0.2 * centre
            score.append(s)
        return float(grid[int(np.argmax(score))])

    if win_w < 0.98:
        x = best(win_w, 0, 2)
        return [round(x, 4), 0.0, round(win_w, 4), 1.0], last_seen
    # Portrait footage: the window spans the width; place it vertically,
    # keeping faces in the upper middle, clear of the app's buttons below.
    y = best(win_h, 1, 3)
    return [0.0, round(y, 4), 1.0, round(win_h, 4)], last_seen


def photo_windows(rgb, boxes, zoom_in, amount):
    """Start and end windows for a gentle 4-8% drift in or out."""
    h, w = rgb.shape[:2]
    x, y, bw, bh = _norm_box(crop.find(rgb, ASPECT, face_boxes=boxes, max_zoom_out=1.0), w, h)
    k = 1.0 - amount
    tight = [round(x + bw * (1 - k) / 2, 4), round(y + bh * (1 - k) / 2, 4),
             round(bw * k, 4), round(bh * k, 4)]
    wide = [x, y, bw, bh]
    return (wide, tight) if zoom_in else (tight, wide)


# --- colour ------------------------------------------------------------------------

def _chapter_colour(chapters, style, seed):
    """One look per chapter, all from one family, so cuts inside a scene
    match but the night is not graded like the day."""
    if style["look"] != "auto":
        for c in chapters:
            c.look = style["look"]
        return
    fam = FAMILIES[style["family"]]
    for c in chapters:
        kind = enhance._chapter_kind([m.src for m in c.moments]) if c.moments else "mixed"
        cands = enhance.AUTO_CANDIDATES[kind]
        c.look = "film" if kind == "dim" else next((l for l in cands if l in fam), fam[0])
    if style["mono_chapter"] and len(chapters) > 2:
        middle = chapters[1:-1]
        pick = max(middle, key=lambda c: sum(m.people for m in c.moments) / max(1, len(c.moments)))
        pick.look = "mono"


def _chapter_wb(chapter):
    """One white balance for a whole chapter: the median of its clips' gains."""
    gains = [(s["wb_r"], s["wb_g"], s["wb_b"]) for s in
             ((c.scores or {}) for c in chapter.clips.values()) if "wb_r" in s]
    if not gains:
        return None
    return [round(float(np.median([g[i] for g in gains])), 4) for i in range(3)]


# --- the recipe ---------------------------------------------------------------------

def _town(items):
    names = []
    for it in items:
        gps = getattr(it, "gps", None)
        if gps and places.available():
            hit = places.nearest(*gps)
            if hit:
                names.append(hit[0])
    return max(sorted(set(names)), key=names.count) if names else ""


def build(event, title, subtitle, moments, seed, opts, music=None,
          frame_at=None, photo_rgb=None):
    """Return the recipe dict for this seed.

    frame_at(clip, t) -> RGB frame; photo_rgb(photo) -> (RGB, face boxes).
    """
    sources = list({id(m.src): m.src for m in moments}.values())
    if music and not opts.get("duration"):
        # With music and no length asked for, the film is the whole song.
        a = music["analysis"]
        song = a["duration"] - a["beats"][a.get("downbeat", 0)]
        opts = dict(opts, duration=round(song, 1))
        style = choose_style(seed, sources, opts)
        style["whole_song"] = True
    else:
        style = choose_style(seed, sources, opts)
        style["exact"] = bool(opts.get("duration"))
        # A song cut at a chosen point plays from its beginning.
        style["from_start"] = bool(opts.get("music_until"))

    by_seg = {}
    for m in moments:
        by_seg.setdefault(m.segment, []).append(m)
    floor = 3       # a chapter of one or two shots is an interlude
    chapters = [_Chapter(s, by_seg[s]) for s in sorted(by_seg)]
    real = [c for c in chapters if len(c.moments) >= floor] or chapters
    _budgets(real, style["duration"], style["emphasis"])
    _tempos(real)
    chapter_of = {}
    for c in chapters:
        for m in c.moments:
            chapter_of[id(m)] = c

    pick = stream(seed, "pick")
    chosen = []
    for c in real:
        chosen.extend(_select(c, style, pick))
    opener = _opener(real[0], style)
    closer = _closer(real[-1], style) if len(real) > 1 or len(real[0].moments) > 1 else None
    if closer is opener:
        closer = None
    for m in (opener, closer):
        if m is not None and m not in chosen:
            chosen.append(m)
    seq = sorted(chosen, key=_when)
    # Nothing before the opener, nothing after the closer.
    seq = seq[seq.index(opener):]
    if closer is not None:
        seq = seq[:seq.index(closer) + 1]
    keep = {id(opener)} | ({id(closer)} if closer else set())
    seq = _tidy(seq, chapter_of, keep)

    items = _lengths(seq, chapter_of, style, seed, opener, closer)
    items = _no_reuse(items, keep)
    if music:
        timed, music_plan = _on_beats(items, style, seed, music["analysis"])
    else:
        timed = [(m, _frames(d) / float(FPS), sp) for m, d, sp in items]
        music_plan = None

    _chapter_colour(real, style, seed)
    wb = {id(c): _chapter_wb(c) for c in real}
    bright = {}
    for c in real:
        vals = [c2.scores.get("mean_brightness") for c2 in c.clips.values()
                if c2.scores and c2.scores.get("mean_brightness")]
        bright[id(c)] = float(np.median(vals)) if vals else None

    # Frames for every crop, fetched together (each is a seek into a 4K file).
    # Cut just before the action, so the shot opens on the build-up.
    clip_start = _start

    if frame_at is not None and hasattr(frame_at, "prefetch"):
        frame_at.prefetch([(m.src, clip_start(m, d, sp) + d * sp / 2)
                           for m, d, sp in timed if m.kind == "clip"])

    kb = {c.seg: stream(seed, "drift:%s" % c.seg).random() < 0.6 for c in chapters}
    drift = stream(seed, "drift")
    segments, segments_taken, t = [], [], 0.0
    prev_date, day = None, 0
    overlays = []
    for i, (m, d, sp) in enumerate(timed):
        ch = chapter_of[id(m)]
        seg = {"id": i, "kind": m.kind, "src": m.src.rel, "dur": round(d, 4),
               "chapter": m.segment, "tag": m.tag,
               "role": "open" if m is opener else ("close" if m is closer else "body"),
               "look": ch.look, "fx": {"in": "cut", "out": "cut"}}
        if m.kind == "clip":
            start = clip_start(m, d, sp)
            frame = frame_at(m.src, start + d * sp / 2) if frame_at else None
            window, last_seen = span_window(m.src, start, start + d * sp, frame)
            # The subject walks out: end the shot as they leave, not after.
            if not music and last_seen is not None and m.faces and \
                    last_seen + 0.25 < start + d * sp and last_seen + 0.25 - start >= 1.5 * sp:
                d = _frames((last_seen + 0.25 - start) / sp) / float(FPS)
                seg["dur"] = round(d, 4)
            seg["in"] = round(start, 3)
            seg["out"] = round(start + d * sp, 3)
            seg["speed"] = sp
            seg["window"] = window
            path = reframe.track(m.src, start, start + d * sp, window) \
                if window[2] < 0.98 and m.faces else None
            if path and sp == 1.0:
                seg["track"] = path
            elif window[2] < 0.98 and m.faces == 0 and sp == 1.0:
                # Scenery from a landscape clip: a slow drift across it.
                way = 1 if drift.random() < 0.5 else -1
                x0 = window[0]
                x1 = min(max(x0 + way * 0.04, 0.0), 1.0 - window[2])
                n = _frames(d)
                seg["track"] = [round(x0 + (x1 - x0) * k / max(1, n - 1), 5) for k in range(n)]
            if m.shake is not None and 0.01 < m.shake <= 0.02:
                seg["stabilise"] = True
            gains = list(wb[id(ch)] or [1.0, 1.0, 1.0])
            b = (m.src.scores or {}).get("mean_brightness")
            if bright.get(id(ch)) and b:
                k = float(np.clip(bright[id(ch)] / b, 0.85, 1.15))
                gains = [round(g * k, 4) for g in gains]
            seg["gain"] = gains
            if sp != 1.0:
                seg["sound"] = "mute"
            elif music:
                seg["sound"] = "up" if (m.onset and d <= 4.0) else "mute"
            else:
                seg["sound"] = "own"
            seg["speech"] = round(m.speech, 2)
            seg["onset"] = m.onset
        else:
            rgb, boxes = photo_rgb(m.src) if photo_rgb else (None, [])
            if rgb is not None:
                if d >= 1.5:
                    a, b = photo_windows(rgb, boxes, kb[m.segment], drift.uniform(0.04, 0.08))
                else:
                    a = b = photo_windows(rgb, boxes, True, 0.0)[0]   # too short to move
                seg["kenburns"] = {"start": a, "end": b}
            seg["sound"] = "bed"
        # Where the day changes, or after a long gap: a dip, and a caption.
        date = m.taken.date() if m.taken else None
        if date and date != prev_date:
            day += 1
            if prev_date is not None:
                seg["fx"]["in"] = "dip"
                segments[-1]["fx"]["out"] = "dip"
                day_items = [x.src for x, _, _ in timed if x.taken and x.taken.date() == date]
                town = _town(day_items)
                text = "DAY %d  ·  %s%s" % (day, (town.upper() + "  ·  ") if town else "",
                                            m.taken.strftime("%d %B").upper())
                overlays.append({"kind": "caption", "text": text,
                                 "t0": round(t + 0.4, 2), "t1": round(t + 2.6, 2)})
            prev_date = date
        elif m.taken and segments and segments_taken[-1] and \
                (m.taken - segments_taken[-1]).total_seconds() > DAY_GAP:
            seg["fx"]["in"] = "dip"
            segments[-1]["fx"]["out"] = "dip"
        # A dissolve between two photos, for the soft style.
        if style["transitions"] == "soft" and m.kind == "photo" and segments and \
                segments[-1]["kind"] == "photo" and seg["fx"]["in"] == "cut":
            seg["fx"]["in"] = "dissolve"
        segments.append(seg)
        segments_taken.append(m.taken or (segments_taken[-1] if segments_taken else None))
        t += d

    # The next shot's own sound comes in early (a J-cut) when it has
    # something to hear - a voice, a cheer - and both shots carry sound.
    if not music:
        for a, b in zip(segments, segments[1:]):
            if b["kind"] == "clip" and a.get("sound") == "own" and b.get("sound") == "own" \
                    and (b.get("speech", 0) > 0.5 or b.get("onset")):
                b["jcut"] = 0.4
        # Under a photo, the previous clip's sound carries on quietly (an L-cut).
        last_clip = None
        for s_ in segments:
            if s_["kind"] == "clip" and s_.get("sound") == "own":
                last_clip = s_
            elif s_["kind"] == "photo" and last_clip is not None:
                s_["ambience"] = {"src": last_clip["src"], "from": last_clip["out"]}
                last_clip = dict(last_clip, out=last_clip["out"] + s_["dur"])

    total = round(sum(s["dur"] for s in segments), 3)
    # The title over the opener, placed away from the faces in it.
    open_seg = segments[0] if segments else None
    title_y = 0.40
    if open_seg is not None:
        fy = _face_y(opener, open_seg)
        if fy is not None:
            title_y = 0.72 if fy < 0.55 else 0.16
    t1 = round(max(1.6, (open_seg["dur"] if open_seg else 3.0) - 0.4), 2)
    overlays.insert(0, {"kind": "title", "text": title, "subtitle": subtitle,
                        "y": title_y, "t0": 0.6, "t1": t1})
    if segments:
        segments[-1]["fx"]["out"] = "fade"
    recipe = {
        "version": 2, "event": event, "seed": seed,
        "output": {"fps": FPS, "duration": total},
        "style": style,
        "overlays": overlays,
        "audio": ({"mode": "music", "music": music["rel"], **music_plan}
                  if music and music_plan else {"mode": "original"}),
        "segments": segments,
    }
    return recipe


def _face_y(m, seg):
    """Vertical centre (0-1, in the output frame) of the opener's faces."""
    if m.kind == "clip":
        tl = m.src.timeline
        i = min(range(len(tl["t"])), key=lambda k: abs(tl["t"][k] - seg["in"]))
        boxes = tl["boxes"][i]
        if not boxes:
            return None
        b = max(boxes, key=lambda b: b[2] * b[3])
        y = b[1] + b[3] / 2
        wy, wh = seg["window"][1], seg["window"][3]
        return (y - wy) / wh
    boxes = getattr(m.src, "face_boxes", None) or []
    if not boxes or "kenburns" not in seg:
        return None
    b = max(boxes, key=lambda b: b[2] * b[3])
    y = b[1] + b[3] / 2
    wy, wh = seg["kenburns"]["start"][1], seg["kenburns"]["start"][3]
    return (y - wy) / wh


def summary(recipe):
    s = recipe["style"]
    kinds = [g["kind"] for g in recipe["segments"]]
    return "%.0fs · %d shots (%d clips, %d photos) · %s · %s emphasis · %s colour%s" % (
        recipe["output"]["duration"], len(kinds), kinds.count("clip"),
        kinds.count("photo"), s["pacing"], s.get("emphasis", "balanced"),
        s.get("family", s.get("look")),
        " · %s BPM" % recipe["audio"]["bpm"] if recipe["audio"]["mode"] == "music" else "")
