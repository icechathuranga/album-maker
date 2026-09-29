"""The review contact sheet.

The point of this file is that the user never has to ask an AI why a photo did
or did not make the album. Every frame appears with its two scores, the page
it landed on, and a plain-English reason - and next to each one, the exact
line to paste into keep.txt or skip.txt to overrule the decision.
"""
import html
import json
import os

from .deps import Image, ImageOps
from .colour import note as colour_note

THUMB = 320

STATUS_LABEL = {
    "selected": ("In the album", "#1a7f37"),
    "not-selected": ("Not used", "#8a8f98"),
    "duplicate": ("Duplicate", "#9a6700"),
    "rejected": ("Rejected", "#b42318"),
    "unreadable": ("Unreadable", "#b42318"),
    "pending": ("Unprocessed", "#8a8f98"),
}


def write_thumbs(photos, outdir, size=THUMB):
    """Small JPEGs for the contact sheet, referenced relatively by the HTML."""
    tdir = os.path.join(outdir, "thumbs")
    os.makedirs(tdir, exist_ok=True)
    made = {}
    for p in photos:
        dest = os.path.join(tdir, "%04d_%s.jpg" % (
            p.seq, os.path.splitext(p.name)[0][:40].replace(os.sep, "_")))
        try:
            with Image.open(p.path) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                img.thumbnail((size, size), Image.LANCZOS)
                img.save(dest, "JPEG", quality=80, optimize=True)
            made[p.name] = os.path.relpath(dest, outdir)
        except Exception:
            made[p.name] = None
    return made


def _page_of(photo, pages):
    for pg in pages:
        for s in pg.slots:
            if s.photo is photo:
                return pg
    return None


def write_html(event, photos, selected, pages, stats, cfg, outdir, pdf_path):
    """Write report.html - the human review surface for one album."""
    thumbs = write_thumbs(photos, outdir)
    rows = []

    order = {"selected": 0, "duplicate": 1, "not-selected": 2,
             "rejected": 3, "unreadable": 4, "pending": 5}
    listed = sorted(photos, key=lambda p: (order.get(p.status, 9), -p.score))

    for p in listed:
        label, colour = STATUS_LABEL.get(p.status, (p.status, "#8a8f98"))
        pg = _page_of(p, pages)
        slot = p.used_slot
        thumb = thumbs.get(p.name)

        detail = []
        if pg is not None:
            detail.append("page %d" % (pg.index + 1 + (1 if cfg["cover_page"] else 0)))
            detail.append("%s layout" % pg.template.replace("_", " "))
            if pg.look:
                detail.append("%s look" % pg.look)
        if slot is not None and slot.effective_dpi:
            warn = " LOW" if slot.effective_dpi < cfg["min_dpi"] else ""
            detail.append("%d DPI%s" % (round(slot.effective_dpi), warn))
            if slot.crop_loss > 0.12:
                detail.append("%.0f%% cropped" % (slot.crop_loss * 100))
        if "wb_r" in (p.scores or {}):
            detail.append(colour_note(p.scores))
        if p.taken:
            detail.append(p.taken.strftime("%H:%M"))
        if p.taken_src != "exif":
            detail.append("time from %s" % p.taken_src)

        reasons = "; ".join(p.reasons) if p.reasons else ""
        rows.append({
            "name": p.name, "thumb": thumb, "label": label, "colour": colour,
            "status": p.status, "score": p.score,
            "technical": p.technical * 100, "content": p.content * 100,
            "hero": p.hero, "dims": "%d x %d" % (p.width, p.height),
            "mp": p.megapixels, "detail": " · ".join(detail), "reasons": reasons,
            "metrics": p.scores,
        })

    counts = {}
    for p in photos:
        counts[p.status] = counts.get(p.status, 0) + 1

    css = """
    :root { color-scheme: light dark; }
    * { box-sizing: border-box; }
    body { margin:0; font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",
           Roboto,Helvetica,Arial,sans-serif; background:#f6f7f9; color:#1b1f24; }
    header { padding:28px 32px 20px; background:#fff; border-bottom:1px solid #e3e6ea; }
    h1 { margin:0 0 4px; font-size:22px; letter-spacing:-0.01em; }
    .sub { color:#656d76; font-size:13px; }
    .bar { display:flex; flex-wrap:wrap; gap:22px; margin-top:18px; }
    .stat b { display:block; font-size:19px; font-weight:600; }
    .stat span { font-size:11px; text-transform:uppercase; letter-spacing:.06em;
                 color:#656d76; }
    .wrap { padding:24px 32px 60px; }
    .filters { margin-bottom:16px; display:flex; gap:8px; flex-wrap:wrap; }
    .filters button { border:1px solid #d0d7de; background:#fff; padding:6px 12px;
                      border-radius:20px; cursor:pointer; font-size:12px; }
    .filters button.on { background:#1b1f24; color:#fff; border-color:#1b1f24; }
    .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(228px,1fr));
            gap:16px; }
    .card { background:#fff; border:1px solid #e3e6ea; border-radius:10px;
            overflow:hidden; display:flex; flex-direction:column; }
    .card img { width:100%; aspect-ratio:4/3; object-fit:cover; display:block;
                background:#eceff2; }
    .card .body { padding:10px 12px 12px; flex:1; display:flex;
                  flex-direction:column; gap:6px; }
    .nm { font-weight:600; font-size:12.5px; word-break:break-all; }
    .tag { display:inline-block; padding:2px 8px; border-radius:20px;
           font-size:10.5px; font-weight:600; color:#fff; }
    .hero { background:#6f42c1; }
    .scores { display:flex; gap:10px; font-size:11px; color:#656d76; }
    .scores b { color:#1b1f24; }
    .meta { font-size:11px; color:#656d76; }
    .why { font-size:11.5px; color:#57606a; font-style:italic; }
    .ovr { font-size:10.5px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
           background:#f6f8fa; border:1px solid #e3e6ea; border-radius:5px;
           padding:4px 6px; color:#57606a; user-select:all; }
    @media (prefers-color-scheme: dark) {
      body { background:#0d1117; color:#e6edf3; }
      header, .card { background:#161b22; border-color:#30363d; }
      .filters button { background:#161b22; border-color:#30363d; color:#e6edf3; }
      .filters button.on { background:#e6edf3; color:#0d1117; }
      .ovr { background:#0d1117; border-color:#30363d; color:#8b949e; }
      .scores b { color:#e6edf3; }
    }
    """

    def card(r):
        img = ('<img src="%s" alt="%s" loading="lazy">' % (html.escape(r["thumb"]),
               html.escape(r["name"]))) if r["thumb"] else '<div class="img"></div>'
        hero = '<span class="tag hero">Hero</span>' if r["hero"] else ""
        override = ("skip.txt" if r["status"] == "selected" else "keep.txt")
        return """
        <div class="card" data-status="%s">
          %s
          <div class="body">
            <div class="nm">%s</div>
            <div><span class="tag" style="background:%s">%s</span> %s</div>
            <div class="scores"><span>overall <b>%.0f</b></span>
                 <span>quality <b>%.0f</b></span><span>content <b>%.0f</b></span></div>
            <div class="meta">%s · %.1f MP</div>
            <div class="meta">%s</div>
            <div class="why">%s</div>
            <div class="ovr">%s &rarr; %s</div>
          </div>
        </div>""" % (
            html.escape(r["status"]), img, html.escape(r["name"]),
            r["colour"], html.escape(r["label"]), hero,
            r["score"], r["technical"], r["content"],
            html.escape(r["dims"]), r["mp"],
            html.escape(r["detail"]), html.escape(r["reasons"]),
            html.escape(r["name"]), override)

    stat_items = [
        ("%d" % len(photos), "photos found"),
        ("%d" % len(selected), "in the album"),
        ("%d" % stats.get("pages", 0), "album pages"),
        ("%d" % counts.get("duplicate", 0), "duplicates"),
        ("%d" % counts.get("not-selected", 0), "not used"),
        ("%d" % counts.get("rejected", 0), "rejected"),
        ("%d" % stats.get("median_dpi", 0), "median DPI"),
        ("%d" % stats.get("below_min_dpi", 0), "below min DPI"),
    ]

    doc = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>%s - album review</title><style>%s</style></head><body>
<header>
  <h1>%s</h1>
  <div class="sub">%s &middot; <a href="%s">open the PDF</a></div>
  <div class="bar">%s</div>
</header>
<div class="wrap">
  <div class="filters">
    <button class="on" data-f="all">All</button>
    <button data-f="selected">In the album</button>
    <button data-f="not-selected">Not used</button>
    <button data-f="duplicate">Duplicates</button>
    <button data-f="rejected">Rejected</button>
  </div>
  <div class="grid">%s</div>
</div>
<script>
document.querySelectorAll('.filters button').forEach(function(b){
  b.addEventListener('click', function(){
    document.querySelectorAll('.filters button').forEach(function(x){
      x.classList.remove('on'); });
    b.classList.add('on');
    var f = b.dataset.f;
    document.querySelectorAll('.card').forEach(function(c){
      c.hidden = !(f === 'all' || c.dataset.status === f);
    });
  });
});
</script>
</body></html>""" % (
        html.escape(event), css, html.escape(event),
        html.escape("%d photos reviewed, %d printed across %d pages"
                    % (len(photos), len(selected), stats.get("pages", 0))),
        html.escape(os.path.relpath(pdf_path, outdir)),
        "".join('<div class="stat"><b>%s</b><span>%s</span></div>' % (v, k)
                for v, k in stat_items),
        "".join(card(r) for r in rows))

    path = os.path.join(outdir, "report.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path


def write_json(event, photos, selected, pages, stats, cfg, outdir):
    """Machine-readable twin of the report, for scripting or re-runs."""
    data = {
        "event": event,
        "config": {k: v for k, v in cfg.items() if not k.endswith("_weights")},
        "stats": stats,
        "pages": [
            {"index": pg.index, "template": pg.template, "look": pg.look,
             "photos": [s.photo.name for s in pg.slots if s.photo]}
            for pg in pages
        ],
        "photos": [p.to_dict() for p in photos],
    }
    path = os.path.join(outdir, "album.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    return path
