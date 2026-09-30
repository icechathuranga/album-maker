"""Command line entry point.

    make-album "Birthday Party"     one event folder from inbox/
    make-album --all                every event folder in inbox/
    make-album "Trip" --dry-run     review report only, no PDF

Everything the tool decides is written to outbox/<Event>/report.html, and can
be overridden with keep.txt / skip.txt in the event folder. No AI involved at
run time - the same folder always produces the same album.
"""
import argparse
import datetime
import os
import sys
import time

from . import (colour, config, cover, curate, dedupe, enhance, faces, ingest,
               layout, pdf, places, preview, quality, report, route)
from .deps import report as deps_report, HAS_REPORTLAB


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_dirs(root):
    """Find inbox/outbox beside the tool, or one level up.

    Both layouts are common: album/inbox next to the code, or inbox/ as a
    sibling of album/. Whichever inbox actually exists wins.
    """
    here_in = os.path.join(root, "inbox")
    parent = os.path.dirname(root)
    up_in = os.path.join(parent, "inbox")
    if os.path.isdir(here_in):
        return here_in, os.path.join(root, "outbox")
    if os.path.isdir(up_in):
        return up_in, os.path.join(parent, "outbox")
    return here_in, os.path.join(root, "outbox")


def _pretty_title(folder_name):
    """Folder names are typed for sorting; covers are read by people.

    "Thailand-Bangkok-2026-March" -> "Thailand Bangkok 2026 March"
    """
    out = folder_name.replace("_", " ").replace("-", " ")
    out = " ".join(out.split())
    return out or folder_name


def _pick_cover(selected, page_w, page_h, named=None):
    """Choose the cover photograph.

    A place beats a person here. A close-up says nothing about where you were,
    crops badly to the paper, and singles out one member of a family album; a
    landmark, a street or a view is a picture OF the trip rather than of one
    moment inside it. So scenery is preferred outright, a group photograph is
    the fallback, and a portrait is chosen only when there is nothing else.

    The cover must also print well: it fills the whole page, so a small
    messaging-app copy is ruled out however good it looks on screen.
    """
    if not selected:
        return None

    # An explicit choice always wins: whether a photo represents the trip is a
    # judgement, not a measurement.
    if named:
        for p in selected:
            if p.name.lower() in named:
                return p

    page_aspect = page_w / float(page_h) if page_h else 1.0

    def rank(p):
        m = p.scores or {}
        faces = m.get("faces", 0)
        largest = m.get("face_largest", 0.0)
        group = m.get("group_score", 0.0)

        value = p.score
        if faces == 0:
            # A place or an object: the preferred cover by a clear margin.
            value += 40.0 * min(1.0, p.technical / 0.85)
        else:
            value += 12.0 * group          # a group shot is the fallback
            if faces <= 2 and largest > 0.045:
                value -= 18.0              # a close-up portrait is the last resort

        # It is printed full page, so it needs the pixels for one - and it is
        # the first thing anyone sees, so softness disqualifies it outright.
        # Sharp pixels, not file pixels: a smeared 16 MP frame is no cover.
        mp = (p.scores or {}).get("effective_mp", p.megapixels)
        if mp < 3.0:
            value -= 30.0 * (1.0 - mp / 3.0)
        sharp = (p.scores or {}).get("sharpness", 0.0)
        if sharp < 0.72:
            value -= 45.0 * (0.72 - sharp) / 0.72
        if p.technical < 0.80:
            value -= 25.0

        # It has to fill the paper without losing half the picture.
        loss = 1.0 - (min(p.aspect, page_aspect) / max(p.aspect, page_aspect))
        value -= loss * 30.0
        return value

    return max(selected, key=rank)


def _chapter_place(photos):
    """The town most of a chapter's photos were taken in, or ''."""
    names = [p.place[0] for p in photos if getattr(p, "place", None)]
    return max(sorted(set(names)), key=names.count) if names else ""


def _chapter_when(photo):
    """'Saturday evening · 01 March' - the line under a chapter title."""
    if not photo.taken:
        return ""
    return "%s %s  ·  %s" % (photo.taken.strftime("%A"),
                             places.part_of_day(photo.taken),
                             photo.taken.strftime("%d %B"))


def _fmt_range(photos):
    # Only photos whose capture time is actually known. Undated ones carry the
    # date they were copied, which stretched a one-week trip across six months.
    times = [p.taken for p in photos
             if p.taken and p.taken_src != ingest.UNKNOWN_TIME]
    if not times:
        times = [p.taken for p in photos if p.taken]
    if not times:
        return ""
    lo, hi = min(times), max(times)
    if lo.date() == hi.date():
        return lo.strftime("%d %B %Y")
    if (lo.year, lo.month) == (hi.year, hi.month):
        return "%s-%s" % (lo.strftime("%d"), hi.strftime("%d %B %Y"))
    return "%s - %s" % (lo.strftime("%d %b %Y"), hi.strftime("%d %b %Y"))


def _human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%.1f %s" % (n, unit) if unit != "B" else "%d B" % n
        n /= 1024.0


class Progress:
    """Single-line progress that stays quiet when output is redirected."""

    def __init__(self, label, total, enabled=True):
        self.label, self.total = label, total
        self.enabled = enabled and sys.stderr.isatty()
        self.start = time.time()
        self.n = 0

    def step(self, n=1, note=""):
        self.n += n
        if not self.enabled:
            return
        pct = 100.0 * self.n / max(1, self.total)
        bar = "#" * int(pct / 4) + "." * (25 - int(pct / 4))
        sys.stderr.write("\r  %-9s [%s] %3.0f%% %-28s" % (
            self.label, bar, pct, note[:28]))
        sys.stderr.flush()

    def done(self, note=""):
        if self.enabled:
            sys.stderr.write("\r  %-9s [%s] 100%% %-28s\n" % (
                self.label, "#" * 25, note[:28]))
            sys.stderr.flush()


def _reset_selection(photos):
    """Undo a selection pass so the next attempt starts clean."""
    for p in photos:
        if p.status in ("selected", "not-selected"):
            p.status = "pending"
        p.hero = False
        p.used_slot = None
        p.segment = 0
        p.reasons = [r for r in p.reasons
                     if not r.startswith(("best photo of this part",
                                          "the best frame of this scene",
                                          "kept as an establishing shot",
                                          "the album already has its share",
                                          "printed smaller - ",
                                          "already ", "below the album",
                                          "ranked below the cut"))]


def fit_to_pages(photos, cfg, page_w, page_h, keep=None, verbose=False):
    """Choose photos so the finished album lands on the page count asked for.

    Page count is an OUTPUT of the layout, not an input: how many pages a set
    of photos fills depends on their shapes and on which templates fit them.
    So rather than guess a photos-per-page ratio and hope, this composes the
    album, measures it, adjusts how many photos to select, and repeats until
    the result is inside the requested tolerance.

    Cheap to iterate: scoring is already done, and selection and layout are
    pure arithmetic on the results.
    """
    keep = keep or set()
    target = max(1, int(cfg["target_pages"]))
    tol = max(0.0, float(cfg.get("page_tolerance", 0.10)))
    low, high = target * (1.0 - tol), target * (1.0 + tol)
    # Aim a little high inside the band: an album with a few more memories in
    # it is the better miss, and photos are what the user is here for.
    aim = target * (1.0 + tol * 0.45)

    per_page = float(cfg["photos_per_page"])
    style = cfg.get("page_style", "paper")
    attempts = []
    last_count = None

    for _ in range(12):
        _reset_selection(photos)
        selected = curate.select(
            photos, target_pages=target, photos_per_page=per_page,
            max_per_scene=cfg["max_per_scene"], hero_fraction=cfg["hero_fraction"],
            min_score=cfg["min_score"], technical_floor=cfg["technical_floor"],
            sharpness_floor=cfg["sharpness_floor"],
            face_sharpness_floor=cfg["face_sharpness_floor"],
            scenic_share=cfg["scenic_share"], scenic_max=cfg["scenic_max"],
            hero_separation_seconds=cfg["hero_separation_seconds"])
        # keep.txt wins over any scoring, and must be in the selection BEFORE
        # the pages are built or the forced photos never reach a page.
        if keep:
            chosen = set(id(p) for p in selected)
            for p in photos:
                if p.name.lower() in keep and id(p) not in chosen \
                        and p.status != "rejected":
                    p.status = "selected"
                    if "forced in by keep.txt" not in p.reasons:
                        p.reasons.append("forced in by keep.txt")
                    selected.append(p)
            selected.sort(key=lambda p: (p.taken or p.seq, p.seq))

        if not selected:
            return [], [], per_page

        spaced = style in ("paper", "float")
        openers = {}
        ordered = selected
        if spaced and cfg.get("chapter_openers", True):
            openers = layout.plan_openers(selected, page_w, page_h, target,
                                          min_dpi=cfg["min_dpi"],
                                          place_of=_chapter_place, when_of=_chapter_when)
            # An opener leads its chapter, so move it to the chapter's start.
            ordered = []
            for p in selected:
                if id(p) in openers:
                    at = next((i for i, q in enumerate(ordered)
                               if q.segment == p.segment), len(ordered))
                    ordered.insert(at, p)
                else:
                    ordered.append(p)
        pages = layout.compose(
            ordered, page_w, page_h,
            # Paper and floating pages use one margin and one gap for the whole
            # book; blended pages cross-fade, so they have no gap at all.
            margin=cfg["page_margin_mm"] if spaced else cfg["margin_mm"],
            gutter=(cfg["page_gap_mm"] if spaced else
                    0.0 if style == "blend" else cfg["gutter_mm"]),
            min_dpi=cfg["min_dpi"],
            # Curated pages: 1-3 photos, one clearly leading.
            max_per_page=min(3, cfg["max_photos_per_page"]) if spaced
            else cfg["max_photos_per_page"],
            hero_full_page=cfg["hero_full_page"],
            multi_photo_bias=cfg["multi_photo_bias"],
            bleed_mm=cfg["bleed_mm"], bleed_heroes=cfg["bleed_heroes"],
            bleed_singles=cfg["bleed_singles"],
            bleed_multi_every=(1 if style == "blend" else 0 if spaced
                               else cfg["bleed_multi_every"]),
            overlap_bonus=0.4 if spaced else 0.0,
            detail_pages=spaced, pacing=spaced, openers=openers)

        n = len(pages)
        # Chapters and heroes are what the next attempt resets, so keep them
        # with this one in case it turns out to be the best.
        marks = {id(p): (p.segment, p.hero) for p in selected}
        attempts.append((abs(n - target), n, list(selected), pages, per_page, marks))
        if low <= n <= high:
            return selected, pages, per_page

        # The folder simply has no more photos worth printing.
        if last_count is not None and len(selected) == last_count and n < low:
            break
        last_count = len(selected)

        # Pages scale with photos, so scale the request by how far off we are.
        per_page = max(0.6, min(6.0, per_page * (aim / float(max(1, n)))))

    attempts.sort(key=lambda a: a[0])
    _, _, selected, pages, per_page, marks = attempts[0]
    # Restore the winning selection so photo statuses match what is printed.
    _reset_selection(photos)
    chosen = set(id(p) for p in selected)
    for p in photos:
        if id(p) in chosen:
            p.status = "selected"
            p.segment, p.hero = marks[id(p)]
        elif p.status == "pending":
            p.status = "not-selected"
    for pg in pages:
        for slot in pg.slots:
            if slot.photo is not None:
                slot.photo.used_slot = slot
    return selected, pages, per_page


def build(event_dir, out_dir, cfg, verbose=True, dry_run=False, cover_sheet=False):
    """Run the whole pipeline for one event folder. Returns a result dict."""
    event = os.path.basename(os.path.normpath(event_dir))
    t0 = time.time()

    photos, skipped = ingest.scan(event_dir)
    if not photos:
        return {"event": event, "error": "no readable images found", "skipped": skipped}

    if verbose:
        print("  %d image files found" % len(photos))

    prog = Progress("reading", len(photos), verbose)
    for p in photos:
        ingest.probe(p)
        prog.step(note=p.name)
    prog.done()

    photos = [p for p in photos if p.status != "unreadable"]
    if not photos:
        return {"event": event, "error": "every file failed to open", "skipped": skipped}

    # Photos that arrived with no EXIF at all may also have lost their
    # orientation tag, leaving them sideways with nothing to say so.
    rotated = 0
    if cfg.get("auto_rotate", True) and faces.available():
        from .deps import np as _np

        def _load(ph):
            return ingest.load(ph, max_long_edge=900)

        def _find(img):
            return faces.detect(_np.asarray(img.convert("RGB"), dtype=_np.uint8))

        candidates = [p for p in photos if p.taken_src == ingest.UNKNOWN_TIME
                      or p.orientation == 1 and p.taken_src == "file-mtime"]
        prog = Progress("rotation", len(candidates), verbose and bool(candidates))
        for p in candidates:
            turn = ingest.detect_upright(p, _load, _find)
            if turn:
                p.extra_rotation = turn
                p.width, p.height = p.height, p.width
                p.reasons.append("turned upright (its orientation tag was stripped)")
                rotated += 1
            prog.step(note=p.name)
        prog.done()

    located = 0
    if cfg.get("captions", "auto") != "off" and places.available():
        for p in photos:
            if p.gps:
                p.place = places.nearest(*p.gps)
                if p.place:
                    located += 1

    repaired = ingest.reconcile_times(photos)
    ingest.order(photos)
    no_time = len(ingest.undated(photos))

    if cfg["detect_faces"] and not faces.available():
        print("  WARNING: face detection is unavailable, so people will not be "
              "ranked.\n           Photos will be judged on scene quality only.",
              file=sys.stderr)

    prog = Progress("scoring", len(photos), verbose)
    for p in photos:
        quality.analyse(p, detect_faces=cfg["detect_faces"],
                        content_mix=cfg["content_mix"],
                        print_sharpness=cfg.get("print_sharpness", "normal"))
        p.reasons.extend(quality.describe(p))
        prog.step(note=p.name)
    prog.done()

    n_groups = dedupe.group(photos, burst_seconds=cfg["burst_seconds"],
                            burst_bits=cfg["burst_bits"],
                            exact_bits=cfg["exact_bits"])

    # Manual overrides always win over anything the tool decided.
    keep = config.read_list_file(event_dir, "keep.txt")
    skip = config.read_list_file(event_dir, "skip.txt")
    cover_order = config.read_list_ordered(event_dir, "cover.txt")
    cover_named = set(cover_order)
    # A photo named for the cover is used on the cover whatever selection
    # made of it, but it is not forced into the pages: a cover photo is often
    # one frame of a burst the album already prints, and forcing it in put two
    # near-identical full pages side by side.
    for p in photos:
        low = p.name.lower()
        if low in skip:
            p.status = "rejected"
            p.reasons = ["excluded by skip.txt"]
        elif low in keep and p.status in ("duplicate", "rejected"):
            p.status = "pending"
            p.reasons.append("forced in by keep.txt")

    page_w, page_h = layout.page_size(cfg["page_size"], cfg["orientation"], photos)
    layout.FILL_CROP_BUDGET = cfg["fill_crop_budget"]
    selected, pages, used_ratio = fit_to_pages(photos, cfg, page_w, page_h,
                                               keep=keep, verbose=verbose)

    if not selected:
        return {"event": event, "error": "no photos cleared the quality bar",
                "skipped": skipped}

    # Chapters are final now, so each can be given one colour correction
    # and, with look=auto, one look.
    colour.balance(selected)
    album_look = cfg["look"]
    if album_look == enhance.AUTO:
        chosen = enhance.auto_looks(pages, seed=event)
        page_looks = [pg.look for pg in pages if pg.look]
        # The cover wears the look most of the album wears.
        album_look = max(sorted(set(page_looks)), key=page_looks.count) \
            if page_looks else enhance.DEFAULT_LOOK
    # A few pages in black and white, the way current albums mix them in.
    mono = enhance.mono_pages(pages, share=float(cfg.get("mono_share", 0.10))) \
        if album_look != "mono" else []
    route_stops = route.stops(selected) if cfg.get("route_map", True) else []

    stats = layout.stats(pages)
    stats["segments"] = n_groups and len(set(p.segment for p in selected))
    if cfg["look"] == enhance.AUTO:
        stats["looks"] = {str(k): v for k, v in sorted(chosen.items())}
    stats["mono_pages"] = [pg.index + 1 for pg in mono]
    stats["openers"] = [pg.index + 1 for pg in pages if pg.kind == "opener"]
    stats["detail_pages"] = [pg.index + 1 for pg in pages if pg.kind == "detail"]
    stats["route_stops"] = [s["label"] for s in route_stops]

    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(os.path.dirname(os.path.normpath(out_dir)), event + ".pdf")

    result = {
        "event": event, "photos": len(photos), "selected": len(selected),
        "pages": len(pages), "stats": stats, "skipped": skipped,
        "timestamps_repaired": repaired, "located": located,
        "undated": no_time, "rotated": rotated, "pdf": pdf_path,
        "target_pages": cfg["target_pages"],
        "on_target": abs(len(pages) - cfg["target_pages"])
                     <= cfg["target_pages"] * cfg.get("page_tolerance", 0.10) + 0.5,
    }

    cover_pool = selected + [p for p in photos
                             if p.name.lower() in cover_named and p not in selected
                             and p.status != "unreadable" and p.name.lower() not in skip]
    hero = (_pick_cover(cover_pool, page_w, page_h, named=cover_named)
            if cfg["cover_photo"] else None)
    style = cfg.get("cover_style", cover.DEFAULT_STYLE)
    if style not in cover.STYLES:
        print("  unknown cover_style %r, using %s" % (style, cover.DEFAULT_STYLE),
              file=sys.stderr)
        style = cover.DEFAULT_STYLE
    cover_args = dict(title=_pretty_title(event), subtitle=_fmt_range(selected),
                      detail="%d photographs" % len(selected))

    # Works on a dry run too: choosing a cover should not need the whole album.
    if cover_sheet and hero is not None:
        sheet = os.path.join(out_dir, "covers.pdf")
        shown = pdf.render_covers(
            sheet, page_w, page_h, hero=hero, pool=cover_pool, look=album_look,
            named=cover_order, gap_seconds=cfg["hero_separation_seconds"],
            **cover_args)
        result["covers"] = sheet
        result["cover_styles"] = shown

    if not dry_run:
        pdf.JPEG_QUALITY = cfg["jpeg_quality"]
        prog = Progress("rendering", len(selected), verbose)

        def on_progress(done, total, name):
            prog.n = done - 1
            prog.step(1, name)

        info = pdf.render(
            pages, pdf_path, page_w, page_h, **cover_args,
            target_dpi=float(cfg["target_dpi"]), do_enhance=cfg["enhance"],
            cover=cfg["cover_page"], page_numbers=cfg["page_numbers"],
            progress=on_progress, smart_crop=cfg["smart_crop"], look=album_look,
            captions=cfg.get("captions", "auto"),
            cover_photo=hero, cover_style=style, cover_pool=cover_pool,
            cover_named=cover_order, cover_gap=cfg["hero_separation_seconds"],
            page_style=cfg.get("page_style", "paper"), route_stops=route_stops)
        prog.done()
        result.update(info)

    if not dry_run and cfg.get("slideshow", True):
        show = preview.build(pdf_path, out_dir, _pretty_title(event),
                             _fmt_range(selected), dpi=cfg.get("slideshow_dpi", 96))
        if show:
            result["slideshow"] = show

    report.write_html(event, photos, selected, pages, stats, cfg, out_dir, pdf_path)
    report.write_json(event, photos, selected, pages, stats, cfg, out_dir)
    result["report"] = os.path.join(out_dir, "report.html")
    result["seconds"] = round(time.time() - t0, 1)
    return result


def main(argv=None):
    root = _project_root()
    def_in, def_out = _default_dirs(root)

    ap = argparse.ArgumentParser(
        prog="make-album",
        description="Build a printable PDF album from a folder of event photos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  make-album "Birthday Party"        build inbox/Birthday Party -> outbox/Birthday Party.pdf
  make-album --all                   build every folder in inbox/
  make-album "Trip" A4 landscape 35pages   size, orientation and length in plain words
  make-album "Trip" --pages 30       aim for a 30-page album
  make-album "Trip" --dry-run        write the review report, skip the PDF
  make-album "Trip" collage          choose a cover design by name
  make-album "Trip" --covers         also write covers.pdf with every design
  make-album --list                  show what is waiting in inbox/
  make-album --init-config           write album.json you can edit
""")
    ap.add_argument("event", nargs="*",
                    help="event folder name(s) inside inbox/, optionally followed "
                         "by a spec such as: A4 landscape 35pages")
    ap.add_argument("--spec", help='album spec in plain words, e.g. '
                                   '"A4 landscape 35pages vivid"')
    ap.add_argument("--all", action="store_true", help="process every folder in inbox/")
    ap.add_argument("--list", action="store_true", help="list event folders and exit")
    ap.add_argument("--inbox", default=def_in)
    ap.add_argument("--outbox", default=def_out)
    ap.add_argument("--pages", type=int, dest="target_pages",
                    help="target page count (default 24)")
    ap.add_argument("--page-size", dest="page_size",
                    choices=sorted(layout.PAGE_SIZES_MM), help="paper size")
    ap.add_argument("--orientation", choices=["auto", "portrait", "landscape"])
    ap.add_argument("--dpi", type=int, dest="target_dpi")
    ap.add_argument("--scenic", type=float, dest="scenic_share",
                    help="share of the album kept for places and things "
                         "with no people in them (0-1, default 0.28)")
    ap.add_argument("--look", choices=sorted(list(enhance.LOOKS) + [enhance.AUTO]),
                    help="colour treatment (default auto: chosen per chapter)")
    ap.add_argument("--no-enhance", action="store_true", help="skip all corrections")
    ap.add_argument("--no-faces", action="store_true", help="skip face detection")
    ap.add_argument("--no-cover", action="store_true", help="omit the cover page")
    ap.add_argument("--cover-style", dest="cover_style", choices=cover.STYLES,
                    help="cover design (default %s)" % cover.DEFAULT_STYLE)
    ap.add_argument("--covers", action="store_true",
                    help="also write covers.pdf showing every cover style")
    ap.add_argument("--max-per-page", type=int, dest="max_photos_per_page")
    ap.add_argument("--dry-run", action="store_true",
                    help="analyse and report, but do not render the PDF")
    ap.add_argument("--init-config", action="store_true",
                    help="write a default album.json next to the tool")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.init_config:
        path = config.write_default(os.path.join(root, config.CONFIG_NAME))
        print("wrote %s" % path)
        return 0

    if not os.path.isdir(args.inbox):
        print("inbox not found: %s" % args.inbox, file=sys.stderr)
        print("create it and put one folder per event inside.", file=sys.stderr)
        return 2

    available = sorted(d for d in os.listdir(args.inbox)
                       if os.path.isdir(os.path.join(args.inbox, d))
                       and not d.startswith("."))

    if args.list:
        if not available:
            print("inbox is empty: %s" % args.inbox)
            return 0
        print("event folders in %s:" % args.inbox)
        for d in available:
            full = os.path.join(args.inbox, d)
            n = len(ingest.scan(full)[0])
            print("  %-40s %4d photos" % (d, n))
        return 0

    # Words after the event name that are not folders are read as a spec, so
    # `make-album "Trip" A4 landscape 35pages` works without flag names.
    spec_words = []
    events = []
    for token in args.event:
        if token in available or os.path.isdir(os.path.join(args.inbox, token)):
            events.append(token)
        else:
            spec_words.append(token)

    spec_text = " ".join(filter(None, [args.spec, " ".join(spec_words)]))
    spec_overrides, unknown = config.parse_spec(spec_text,
                                                set(layout.PAGE_SIZES_MM))
    if unknown and not events:
        # Every word was unrecognised - most likely a misspelled folder name.
        print("no such event folder: %s" % ", ".join(unknown), file=sys.stderr)
        print("available: %s" % (", ".join(available) or "(none)"), file=sys.stderr)
        return 2
    if unknown:
        print("ignored unrecognised words in the spec: %s" % " ".join(unknown),
              file=sys.stderr)
    args.event = events

    if args.all:
        targets = available
    elif args.event:
        targets = args.event
    else:
        ap.print_help()
        print("\nevent folders available: %s" % (", ".join(available) or "(none)"))
        return 1

    if not targets:
        print("nothing to do - inbox has no event folders", file=sys.stderr)
        return 1

    if not HAS_REPORTLAB and not args.dry_run:
        print("reportlab is missing, cannot write PDFs. Run install.sh.", file=sys.stderr)
        return 2

    if not args.quiet:
        print("photoalbum  |  %s" % deps_report())
        if spec_overrides:
            print("spec        |  %s" % "  ".join(
                "%s=%s" % (k, v) for k, v in sorted(spec_overrides.items())))

    overrides = {
        "target_pages": args.target_pages, "page_size": args.page_size,
        "orientation": args.orientation, "target_dpi": args.target_dpi,
        "max_photos_per_page": args.max_photos_per_page,
        "look": args.look,
        "scenic_share": args.scenic_share,
        "enhance": False if args.no_enhance else None,
        "detect_faces": False if args.no_faces else None,
        "cover_page": False if args.no_cover else None,
        "cover_style": args.cover_style,
    }

    failures = 0
    for name in targets:
        event_dir = os.path.join(args.inbox, name)
        if not os.path.isdir(event_dir):
            print("no such event folder: %s" % event_dir, file=sys.stderr)
            failures += 1
            continue

        merged = dict(spec_overrides)
        merged.update({k: v for k, v in overrides.items() if v is not None})
        cfg = config.load(root, event_dir, merged)
        out_dir = os.path.join(args.outbox, name)

        if not args.quiet:
            print("\n%s" % name)
            print("-" * max(8, len(name)))

        try:
            res = build(event_dir, out_dir, cfg, verbose=not args.quiet,
                        dry_run=args.dry_run, cover_sheet=args.covers)
        except KeyboardInterrupt:
            print("\ninterrupted", file=sys.stderr)
            return 130
        except Exception as exc:
            print("  failed: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
            failures += 1
            continue

        if res.get("error"):
            print("  %s" % res["error"], file=sys.stderr)
            failures += 1
            continue

        st = res["stats"]
        print("  %d photos -> %d selected -> %d pages%s" % (
            res["photos"], res["selected"], res["pages"],
            "" if res.get("on_target", True)
            else "  (asked for %d; the folder has no more worth printing)"
                 % res.get("target_pages", 0)))
        print("  print quality: median %d DPI, lowest %d DPI%s" % (
            st.get("median_dpi", 0), st.get("min_dpi", 0),
            ", %d slot(s) below the minimum" % st["below_min_dpi"]
            if st.get("below_min_dpi") else ""))
        if res.get("vanished"):
            print("  %d photo(s) disappeared from the inbox mid-run and were "
                  "left out: %s" % (len(res["vanished"]),
                                    ", ".join(res["vanished"][:3])), file=sys.stderr)
        if res.get("located"):
            print("  identified the location of %d photos from GPS" % res["located"])
        if res.get("timestamps_repaired"):
            print("  recovered %d timestamps from neighbouring files"
                  % res["timestamps_repaired"])
        if res.get("rotated"):
            print("  turned %d sideways photos upright (EXIF orientation was "
                  "stripped)" % res["rotated"])
        if res.get("undated"):
            print("  %d photos have no capture time at all (EXIF stripped - "
                  "typically\n           shared through a messaging app). They are "
                  "placed after the\n           dated photos rather than guessed into "
                  "the timeline." % res["undated"])
        for fn, why in res.get("skipped", [])[:5]:
            print("  skipped %s - %s" % (fn, why))
        if args.dry_run:
            print("  dry run - no PDF written")
        else:
            print("  PDF:    %s  (%s)" % (res["pdf"], _human_size(res["size_bytes"])))
        if res.get("cover_style"):
            asked = cfg.get("cover_style")
            print("  cover:  %s%s" % (res["cover_style"],
                  "" if res["cover_style"] == asked
                  else "  (too few photos for %s)" % asked))
        if res.get("covers"):
            print("  covers: %s  (%s)" % (res["covers"], ", ".join(res["cover_styles"])))
        if res.get("slideshow"):
            print("  view:   %s" % res["slideshow"])
        print("  review: %s" % res["report"])
        print("  took %.1fs" % res["seconds"])

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
