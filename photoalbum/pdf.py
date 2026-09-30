"""Render composed pages to a print-ready PDF.

Each photo is resized to exactly the pixel count its slot needs at the target
DPI, enhanced, then written as a JPEG and handed to reportlab by path.
reportlab copies JPEG bytes into the PDF untouched; passing it a PIL object or
a PNG instead makes it re-encode losslessly and the file balloons several
times over for no visible gain.
"""
import os
import shutil
import tempfile

from .deps import np, cv2, Image, ImageDraw, ImageFont, ImageFilter, HAS_REPORTLAB
from . import ingest, enhance, crop, cover as covers, faces as facedet, places
from .layout import Slot

if HAS_REPORTLAB:
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.colors import Color, black, white
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

MM_TO_PT = 72.0 / 25.4
JPEG_QUALITY = 92

# Cropping towards the top of a frame is safer than dead centre when we have
# no face information: heads sit high, feet are expendable.
DEFAULT_CROP_BIAS_Y = 0.42


def _register_fonts():
    """Use a real system font if one is available; fall back to Helvetica."""
    candidates = [
        ("AlbumSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ("AlbumSans", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    ]
    bold = [
        ("AlbumSans-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("AlbumSans-Bold", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ]
    regular_name, bold_name = "Helvetica", "Helvetica-Bold"
    for name, path in candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                regular_name = name
                break
            except Exception:
                pass
    for name, path in bold:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                bold_name = name
                break
            except Exception:
                pass
    return regular_name, bold_name


def _register_display_fonts():
    """A light serif for poster titles and a light sans for spaced subtitles.

    Falls back through what is commonly installed, ending on the core PDF
    fonts, so a missing font changes the look but never fails the album.
    """
    def first(name, paths, fallback):
        for path in paths:
            if os.path.exists(path):
                try:
                    pdfmetrics.registerFont(TTFont(name, path))
                    return name
                except Exception:
                    pass
        return fallback
    tt = "/usr/share/fonts/truetype/"
    serif = first("AlbumSerif", [tt + "fonts-yrsa-rasa/Yrsa-Regular.ttf",
                                 tt + "liberation2/LiberationSerif-Regular.ttf",
                                 tt + "liberation/LiberationSerif-Regular.ttf",
                                 tt + "dejavu/DejaVuSerif.ttf"], "Times-Roman")
    light = first("AlbumLight", [tt + "ubuntu/Ubuntu-L.ttf",
                                 tt + "dejavu/DejaVuSans-ExtraLight.ttf",
                                 tt + "liberation2/LiberationSans-Regular.ttf"],
                  "Helvetica")
    return serif, light


def _spaced_width(c, text, font, size, tracking):
    return c.stringWidth(text, font, size) + tracking * max(0, len(text) - 1)


def _draw_spaced(c, cx, y, text, font, size, tracking, color, shadow=0.0):
    """Letter-spaced text centred on cx, with an optional faint drop shadow."""
    x = cx - _spaced_width(c, text, font, size, tracking) / 2.0
    for dx, dy, col in ([(size * 0.02, -size * 0.03, Color(0, 0, 0, alpha=shadow))]
                        if shadow else []) + [(0, 0, color)]:
        t = c.beginText(x + dx, y + dy)
        t.setFont(font, size)
        t.setCharSpace(tracking)
        t.setFillColor(col)
        t.textOut(text)
        c.drawText(t)


def _draw_display_title(c, page_w, page_h, title, subtitle, text):
    """Poster lettering: widely spaced capitals, subtitle between two rules.

    The title fills most of the width; a name too long for one line at a
    readable size breaks onto two at the word nearest the middle.
    """
    serif, light = _register_display_fonts()
    pw, ph = page_w * MM_TO_PT, page_h * MM_TO_PT
    cx, yc = pw / 2.0, ph * text["y"]
    name = covers.display_title(title).upper()
    track = 0.30                      # letter spacing, as a share of the size

    def fits(lines, size):
        return all(_spaced_width(c, ln, serif, size, size * track) <= pw * 0.86
                   for ln in lines)

    lines, size = [name], text.get("size", 58)
    while size > 30 and not fits(lines, size):
        size -= 1
    if not fits(lines, size) and " " in name:
        words = name.split()
        cut = min(range(1, len(words)), key=lambda i: abs(
            len(" ".join(words[:i])) - len(" ".join(words[i:]))))
        lines, size = [" ".join(words[:cut]), " ".join(words[cut:])], text.get("size", 58)
        while size > 18 and not fits(lines, size):
            size -= 1
    else:
        while size > 18 and not fits(lines, size):
            size -= 1

    fg = Color(*text["fg"])
    lead = size * 1.12
    top = yc + (len(lines) - 1) * lead / 2.0
    for i, ln in enumerate(lines):
        _draw_spaced(c, cx, top - i * lead - size * 0.34, ln, serif, size,
                     size * track, fg, shadow=0.30)

    if subtitle:
        sub = subtitle.upper()
        ss = max(11.0, size * 0.24)
        st = ss * 0.32
        sy = top - (len(lines) - 1) * lead - size * 0.34 - size * 0.42 - ss
        _draw_spaced(c, cx, sy, sub, light, ss, st, Color(*text["sub"]), shadow=0.30)
        half = _spaced_width(c, sub, light, ss, st) / 2.0
        gap, margin = ss * 1.1, pw * 0.07
        c.setStrokeColor(Color(*text["rule"]))
        c.setLineWidth(0.6)
        ry = sy + ss * 0.33
        if cx - half - gap > margin + 10:
            c.line(margin, ry, cx - half - gap, ry)
            c.line(cx + half + gap, ry, pw - margin, ry)


def cover_crop(img, target_w_px, target_h_px, focus=None):
    """Scale to fill the slot and crop the overflow, keeping the subject.

    `focus` is a normalised (x, y) point - the centre of the detected faces -
    so a tall group shot is trimmed at the feet rather than through the heads.
    """
    if target_w_px <= 0 or target_h_px <= 0:
        return img
    src_w, src_h = img.size
    scale = max(target_w_px / src_w, target_h_px / src_h)
    new_w = max(target_w_px, int(round(src_w * scale)))
    new_h = max(target_h_px, int(round(src_h * scale)))

    resample = Image.LANCZOS if scale < 1.0 else Image.BICUBIC
    img = img.resize((new_w, new_h), resample)

    fx = focus[0] if focus and focus[0] >= 0 else 0.5
    fy = focus[1] if focus and focus[1] >= 0 else DEFAULT_CROP_BIAS_Y

    left = int(round((new_w - target_w_px) * float(np.clip(fx, 0.0, 1.0))))
    top = int(round((new_h - target_h_px) * float(np.clip(fy, 0.0, 1.0))))
    left = max(0, min(left, new_w - target_w_px))
    top = max(0, min(top, new_h - target_h_px))
    return img.crop((left, top, left + target_w_px, top + target_h_px))


CONTAIN_THRESHOLD = 0.34   # above this much loss, letterbox instead of crop
CROP_ANALYSIS_EDGE = 900


def _letterbox(img, w_px, h_px, background):
    """Fit the whole photo inside the slot, padding with the page colour.

    Used when the shapes are so far apart that cropping would amputate the
    picture - a tall phone photo in a wide slot, for instance. A bordered
    photo on the page is the normal album treatment for that, and it keeps
    the photograph the photographer actually took.
    """
    fitted = img.copy()
    fitted.thumbnail((w_px, h_px), Image.LANCZOS)
    bg = tuple(int(round(c * 255)) for c in background)
    canvas = Image.new("RGB", (w_px, h_px), bg)
    canvas.paste(fitted, ((w_px - fitted.width) // 2, (h_px - fitted.height) // 2))
    return canvas


def _prepare(photo, slot, target_dpi, workdir, do_enhance=True,
             background=(1.0, 1.0, 1.0), smart_crop=True, look="vivid"):
    """Produce the exact JPEG that goes into one slot. Returns (path, notes)."""
    w_px = max(1, int(round(slot.w / 25.4 * target_dpi)))
    h_px = max(1, int(round(slot.h / 25.4 * target_dpi)))
    target_aspect = w_px / float(h_px)

    img = ingest.load(photo)                       # full size, upright
    upscale = max(w_px / photo.width, h_px / photo.height)
    notes = []

    if smart_crop:
        # Find the crop on a small copy, then apply it to the full-size pixels.
        scale = min(1.0, CROP_ANALYSIS_EDGE / float(max(img.size)))
        if scale < 1.0:
            small = img.resize((max(8, int(img.width * scale)),
                                max(8, int(img.height * scale))), Image.BILINEAR)
        else:
            small = img
        box = crop.find(np.asarray(small.convert("RGB"), dtype=np.uint8),
                        target_aspect, face_boxes=photo.face_boxes)
        inv = 1.0 / (scale if scale > 0 else 1.0)
        full_box = (int(box[0] * inv), int(box[1] * inv),
                    int(box[2] * inv), int(box[3] * inv))
        x, y, cw, ch = full_box
        cw = max(8, min(cw, img.width))
        ch = max(8, min(ch, img.height))
        x = max(0, min(x, img.width - cw))
        y = max(0, min(y, img.height - ch))
        photo.crop_box = (x, y, cw, ch)
        img = img.crop((x, y, x + cw, y + ch))
        img = img.resize((w_px, h_px), Image.LANCZOS if cw > w_px else Image.BICUBIC)
        if photo.face_boxes:
            notes.append("cropped around %d face%s" % (
                len(photo.face_boxes), "" if len(photo.face_boxes) == 1 else "s"))
    else:
        focus = None
        if photo.scores.get("face_cx", -1) >= 0:
            focus = (photo.scores["face_cx"], photo.scores["face_cy"])
        img = cover_crop(img, w_px, h_px, focus=focus)

    # Resampling up always softens; give those a touch more bite.
    extra = 25 if upscale > 1.15 else 0
    img, more = enhance.apply(img, photo, enabled=do_enhance, extra_sharpen=extra,
                              look=look)
    notes.extend(more)

    if img.mode != "RGB":
        img = img.convert("RGB")

    out = os.path.join(workdir, "slot_%04d_%s.jpg" % (
        photo.seq, os.path.splitext(photo.name)[0][:40].replace(os.sep, "_")))
    img.save(out, "JPEG", quality=JPEG_QUALITY, optimize=True,
             progressive=False, subsampling=1, dpi=(int(target_dpi), int(target_dpi)))
    return out, notes


def _cover_image(style, hero, pool, page_w, page_h, target_dpi, workdir,
                 look="vivid", named=(), gap_seconds=240, src=None, name="_cover"):
    """Compose the cover as one JPEG. Returns (path, text layout, style used).

    Bands, scrims and blur are painted into the pixels rather than laid over
    them as PDF shapes - they survive flattening at the print shop, and the
    title never lands as white-on-white over a bright sky.
    """
    img, text, used = covers.compose(style, hero, pool, page_w, page_h, target_dpi,
                                    look=look, named=named,
                                    gap_seconds=gap_seconds, src=src)
    out = os.path.join(workdir, name + ".jpg")
    img.save(out, "JPEG", quality=JPEG_QUALITY, optimize=True,
             dpi=(int(target_dpi), int(target_dpi)))
    return out, text, used


# --- blended pages ----------------------------------------------------------
#
# The printed-album look this replaces - photos in white margins, white lines
# between them, insets in white frames - reads as a scrapbook. Here the whole
# page is composed as one picture: photos run to the paper edge, neighbours
# cross-fade over a few millimetres instead of meeting at a white line, any
# space a photo does not fill is a blurred, darkened wash of the page's own
# photos, and insets float on a soft shadow with no frame.

BLEND_MM = 6.0            # widest cross-fade between two photos
BLEND_SHARE = 0.16        # ...and never more than this share of the slot
EDGE_EPS_MM = 0.5         # an edge this close to the paper edge is a paper edge


def _ramp(n, width):
    """0 -> 1 over `width` pixels, smoothstepped, then flat at 1."""
    x = np.clip(np.arange(n, dtype=np.float32) / max(1.0, width), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _backdrop(paths_and_areas, W, H):
    """A blurred, darkened wash of the page's largest photo, filling the page."""
    path = max(paths_and_areas, key=lambda t: t[1])[0]
    with Image.open(path) as im:
        small = im.convert("RGB")
        small.thumbnail((max(16, W // 10), max(16, H // 10)))
    k = max(W / float(small.width), H / float(small.height))
    small = small.resize((max(1, int(small.width * k / 10) + 1),
                          max(1, int(small.height * k / 10) + 1)), Image.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(max(2, small.width // 18)))
    a = np.asarray(small, dtype=np.float32) * 0.55
    big = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize(
        (small.width * 10, small.height * 10), Image.BICUBIC)
    left, top = (big.width - W) // 2, (big.height - H) // 2
    return big.crop((left, top, left + W, top + H))


def _soft_shadow(canvas, box, radius, strength=0.78, offset=(0, 0)):
    """Darken a blurred rectangle under an inset - a shadow with no frame."""
    x0, y0, x1, y1 = box
    pad = radius * 3
    mask = Image.new("L", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), 0)
    ImageDraw.Draw(mask).rectangle((pad, pad, pad + x1 - x0, pad + y1 - y0),
                                   fill=int(255 * strength))
    mask = mask.filter(ImageFilter.GaussianBlur(radius))
    black = Image.new("RGB", mask.size, (0, 0, 0))
    canvas.paste(black, (x0 - pad + offset[0], y0 - pad + offset[1]), mask)


PAPER_L_LIGHT = 93.0      # LAB lightness of the paper on ordinary pages
PAPER_L_DARK = 14.0       # ...and on dim, night-time pages
PAPER_CHROMA = 6.0        # a tint of the photos' colour, never a colour
INSET_EDGE_MM = 0.7       # paper-coloured hairline round an inset photo
DIM_PAGE = 90.0           # median brightness below which a page is "night"


def _paper_colour(page, paths):
    """A flat paper colour for this page, drawn from its own photos.

    The dominant colour of the page (weighted towards what the eye notices,
    as the cover accent is) desaturated to a faint tint: a warm cream beside
    sunset photos, a cool grey beside the sea. Dim chapters get a near-black
    in the same hue, so night photos are not framed in glaring light paper.
    Returns ((r, g, b) 0-255, is_dark).
    """
    samples = []
    for path in paths:
        with Image.open(path) as im:
            samples.append(np.asarray(im.convert("RGB").resize((32, 32)),
                                      dtype=np.float32).reshape(-1, 3))
    a = np.concatenate(samples) / 255.0
    mx, mn = a.max(axis=1), a.min(axis=1)
    weight = (mx - mn) / (mx + 1e-6) * mx + 1e-3
    mean = (a * weight[:, None]).sum(axis=0) / weight.sum()
    lab = cv2.cvtColor(mean.reshape(1, 1, 3).astype(np.float32), cv2.COLOR_RGB2LAB)[0, 0]
    chroma = float(np.hypot(lab[1], lab[2])) or 1.0
    k = min(1.0, PAPER_CHROMA / chroma)
    bright = [(p.scores or {}).get("mean_brightness", 128.0) for p in page.photos]
    dark = bool(bright) and float(np.median(bright)) < DIM_PAGE
    out = np.array([[[PAPER_L_DARK if dark else PAPER_L_LIGHT,
                      lab[1] * k, lab[2] * k]]], dtype=np.float32)
    rgb = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)[0, 0]
    return tuple(int(round(float(np.clip(c, 0, 1)) * 255)) for c in rgb), dark


def _draw_opener_text(c, pw, ph, title, subtitle):
    """A chapter's name across the middle of its opening photo."""
    serif, light = _register_display_fonts()
    name = (title or subtitle or "").upper()
    if not name:
        return
    size = 40.0
    while size > 18 and _spaced_width(c, name, serif, size, size * 0.28) > pw * 0.8:
        size -= 1
    y = ph * 0.48
    _draw_spaced(c, pw / 2.0, y, name, serif, size, size * 0.28,
                 Color(1, 1, 1), shadow=0.3)
    if title and subtitle:
        ss = max(9.0, size * 0.26)
        _draw_spaced(c, pw / 2.0, y - size * 0.5 - ss, subtitle.upper(), light, ss,
                     ss * 0.32, Color(0.93, 0.93, 0.94), shadow=0.3)


PAPER_ROUTE = (0.965, 0.952, 0.925)     # warm paper for the map
INK_ROUTE = (0.24, 0.21, 0.19)


def _draw_route_page(c, pw, ph, stop_list, title, subtitle):
    """Where the trip went: a line through the stops, named, on plain paper.

    No map tiles and no internet - the shape of the journey is the picture.
    """
    from . import route
    serif, light = _register_display_fonts()
    c.setFillColor(Color(*PAPER_ROUTE))
    c.rect(-2, -2, pw + 4, ph + 4, stroke=0, fill=1)
    ink = Color(*INK_ROUTE)
    soft = Color(*[0.45 * v + 0.55 * 0.62 for v in INK_ROUTE])

    head = covers.display_title(title).upper()
    size = 26.0
    while size > 14 and _spaced_width(c, head, serif, size, size * 0.3) > pw * 0.8:
        size -= 1
    _draw_spaced(c, pw / 2.0, ph * 0.88, head, serif, size, size * 0.3, ink)
    _draw_spaced(c, pw / 2.0, ph * 0.88 - size * 0.9, "THE ROUTE", light, 8.5,
                 8.5 * 0.4, soft)

    # Reportlab measures up from the bottom; the projection's box is in the
    # same space, so north stays up.
    pts = route.project(stop_list, (pw * 0.16, ph * 0.14, pw * 0.84, ph * 0.72))
    c.setStrokeColor(ink)
    c.setLineWidth(1.3)
    c.setDash(4, 3)
    path = c.beginPath()
    path.moveTo(*pts[0])
    for x, y in pts[1:]:
        path.lineTo(x, y)
    c.drawPath(path, stroke=1, fill=0)
    c.setDash()

    for i, ((x, y), stop) in enumerate(zip(pts, stop_list)):
        r = 4.0 if i in (0, len(pts) - 1) else 2.8
        c.setFillColor(ink)
        c.circle(x, y, r, stroke=0, fill=1)
        label = (stop["label"] or "").upper()
        right = x < pw * 0.7
        tx = x + 8 if right else x - 8
        for text, font, fs, col, dy in ((label, light, 10.0, ink, 1.5),
                                        (stop["when"].upper(), light, 7.5, soft, -10.0)):
            if not text:
                continue
            w = _spaced_width(c, text, font, fs, fs * 0.25)
            _draw_spaced(c, (tx + w / 2.0) if right else (tx - w / 2.0), y + dy,
                         text, font, fs, fs * 0.25, col)
    if subtitle:
        _draw_spaced(c, pw / 2.0, ph * 0.07, subtitle.upper(), light, 8.0, 8.0 * 0.35, soft)
    c.showPage()


def _blend_page(page, page_w, page_h, dpi, workdir, prepare, float_photos=False,
                paper=False):
    """Compose one page as a single picture. Returns (path, [(photo, notes)]).

    `prepare(photo, slot)` returns the finished JPEG path and notes for a
    photo in a slot, exactly as for a bordered page. With `float_photos`,
    photos inside the page keep their own edges and float on a soft shadow
    over the blurred wash, instead of cross-fading into their neighbours.
    With `paper`, they sit crisp on a flat paper colour taken from the
    photos - no shadow, no blur - and insets get a paper-coloured edge.
    Returns (path, done, paper_colour or None, is_dark).
    """
    float_photos = float_photos or paper
    W = max(1, int(round(page_w / 25.4 * dpi)))
    H = max(1, int(round(page_h / 25.4 * dpi)))
    px = dpi / 25.4
    done, laid = [], []

    for slot in page.slots:
        if slot.photo is None:
            continue
        if slot.frame or (float_photos and not slot.bleed):
            path, notes = prepare(slot.photo, slot)
            laid.append((slot, path, None))
            done.append((slot, notes))
            continue
        # Grow each edge that is inside the paper by half a blend, so two
        # neighbours overlap by a whole one and can cross-fade.
        b = min(BLEND_MM, BLEND_SHARE * min(slot.w, slot.h))
        inner = {"l": slot.x > EDGE_EPS_MM, "t": slot.y > EDGE_EPS_MM,
                 "r": slot.x + slot.w < page_w - EDGE_EPS_MM,
                 "b": slot.y + slot.h < page_h - EDGE_EPS_MM}
        grown = Slot(slot.x - (b / 2 if inner["l"] else 0),
                     slot.y - (b / 2 if inner["t"] else 0),
                     slot.w + b / 2 * (inner["l"] + inner["r"]),
                     slot.h + b / 2 * (inner["t"] + inner["b"]))
        path, notes = prepare(slot.photo, grown)
        laid.append((grown, path, (inner, b)))
        done.append((slot, notes))

    if not laid:
        return None, done, None, False
    tint, dark = None, False
    if paper:
        tint, dark = _paper_colour(page, [p for _, p, _ in laid])
        canvas = Image.new("RGB", (W, H), tint)
    else:
        canvas = _backdrop([(p, s.w * s.h) for s, p, _ in laid], W, H)

    # Full photos first, insets last so they sit on top.
    for slot, path, fade in sorted(laid, key=lambda t: t[2] is None):
        with Image.open(path) as im:
            img = im.convert("RGB")
        x0, y0 = int(round(slot.x * px)), int(round(slot.y * px))
        img = img.resize((max(1, int(round(slot.w * px))),
                          max(1, int(round(slot.h * px)))), Image.LANCZOS) \
            if img.size != (int(round(slot.w * px)), int(round(slot.h * px))) else img
        if fade is None and paper:
            if slot.frame:
                e = max(2, int(round(INSET_EDGE_MM * px)))
                canvas.paste(Image.new("RGB", (img.width + 2 * e, img.height + 2 * e),
                                       tint), (x0 - e, y0 - e))
            canvas.paste(img, (x0, y0))
            continue
        if fade is None:
            r = max(4, int(4.0 * px))
            _soft_shadow(canvas, (x0, y0, x0 + img.width, y0 + img.height), r,
                         offset=(int(r * 0.3), int(r * 0.5)))
            canvas.paste(img, (x0, y0))
            continue
        inner, b = fade
        w_px = b * px
        mx = np.ones(img.width, dtype=np.float32)
        my = np.ones(img.height, dtype=np.float32)
        if inner["l"]:
            mx = np.minimum(mx, _ramp(img.width, w_px))
        if inner["r"]:
            mx = np.minimum(mx, _ramp(img.width, w_px)[::-1])
        if inner["t"]:
            my = np.minimum(my, _ramp(img.height, w_px))
        if inner["b"]:
            my = np.minimum(my, _ramp(img.height, w_px)[::-1])
        mask = Image.fromarray((np.outer(my, mx) * 255).astype(np.uint8), "L")
        canvas.paste(img, (x0, y0), mask)

    if page.kind == "opener":
        # A soft shade across the middle for the chapter title to sit on.
        canvas = covers._title_band(canvas, 0.52, spread=0.15, strength=0.42, edge=0.0)
    out = os.path.join(workdir, "page_%04d.jpg" % page.index)
    canvas.save(out, "JPEG", quality=JPEG_QUALITY, optimize=True,
                dpi=(int(dpi), int(dpi)))
    return out, done, tint, dark


# Where an inset may sit, as the top-left corner in page fractions. Both sides
# at three heights: the inset always hugs a margin, so it reads as a deliberate
# placement rather than something dropped in the middle of the picture.
INSET_CANDIDATES = (
    (0.055, 0.055), (0.575, 0.055),      # top corners
    (0.055, 0.320), (0.575, 0.320),      # mid height
    (0.055, 0.585), (0.575, 0.585),      # bottom corners
)


def choose_inset_side(bg_path, insets, page_w, page_h):
    """Put a stacked pair of insets on the left or right, whichever covers
    less of the finished background - people found or not."""
    try:
        with Image.open(bg_path) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    except Exception:
        return
    H, W = rgb.shape[:2]
    boxes = facedet.detect(rgb)
    imp, _ = crop.importance_map(rgb, [(x / W, y / H, w / W, h / H, sc)
                                       for x, y, w, h, sc in boxes])
    mh, mw = imp.shape

    def covered(xs):
        total = 0.0
        for ins, x in zip(insets, xs):
            x0, y0 = int(x / page_w * mw), int(ins.y / page_h * mh)
            x1 = min(mw, int((x + ins.w) / page_w * mw))
            y1 = min(mh, int((ins.y + ins.h) / page_h * mh))
            total += float(imp[max(0, y0):y1, max(0, x0):x1].sum())
        return total

    here = [s.x for s in insets]
    mirror = [page_w - s.x - s.w for s in insets]
    if covered(mirror) < covered(here):
        for s, x in zip(insets, mirror):
            s.x = x


def choose_inset_position(bg_path, w_frac, h_frac):
    """Pick where an inset should sit over a finished background page.

    Face boxes alone are not enough to answer this: on the page that prompted
    it, a child stood in the lower left with her head turned, no face was
    detected there, and the inset was dropped straight onto her. So this works
    from the same importance map the cropper uses - detail, colour, local
    contrast and skin tone - with any faces that ARE found stamped on top, and
    then puts the inset wherever the picture underneath is quietest.
    """
    try:
        with Image.open(bg_path) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None

    # Faces found on the finished page, in that page's own coordinates.
    boxes = facedet.detect(rgb)
    H, W = rgb.shape[:2]
    norm = [(x / W, y / H, w / W, h / H, sc) for x, y, w, h, sc in boxes]

    imp, _ = crop.importance_map(rgb, norm)
    mh, mw = imp.shape
    total = float(imp.sum()) or 1.0

    best, best_cost = None, None
    for (px, py) in INSET_CANDIDATES:
        if px + w_frac > 0.985 or py + h_frac > 0.985:
            continue
        x0, y0 = int(px * mw), int(py * mh)
        x1 = min(mw, int((px + w_frac) * mw))
        y1 = min(mh, int((py + h_frac) * mh))
        if x1 <= x0 or y1 <= y0:
            continue
        covered = float(imp[y0:y1, x0:x1].sum()) / total
        if best_cost is None or covered < best_cost:
            best, best_cost = (px, py), covered
    return best


def _draw_cover(c, page_w, page_h, title, subtitle, detail, fonts,
                bg=(0.09, 0.10, 0.12), image_path=None, text=None, label=None):
    regular, bold = fonts
    if image_path:
        # Drawn a little past every edge. Placed exactly on the page, viewers
        # anti-alias the boundary into a white hairline, and a print shop's
        # trim can drift by a millimetre and show paper.
        over = 1.5 * MM_TO_PT
        c.drawImage(image_path, -over, -over, width=page_w * MM_TO_PT + 2 * over,
                    height=page_h * MM_TO_PT + 2 * over, preserveAspectRatio=False)
    else:
        c.setFillColor(Color(*bg))
        c.rect(0, 0, page_w * MM_TO_PT, page_h * MM_TO_PT, stroke=0, fill=1)

    if text and text.get("display"):
        _draw_display_title(c, page_w, page_h, title, subtitle, text)
        _draw_label(c, page_h, label, bold)
        c.showPage()
        return

    # With no picture the title sits near the optical centre of a dark page.
    text = text or dict(covers.LIGHT_TEXT, y=0.56, size=34,
                        sub=(0.78, 0.78, 0.80), rule=(0.45, 0.45, 0.48),
                        detail=(0.55, 0.55, 0.58))
    cx = page_w * MM_TO_PT / 2.0
    size = text.get("size", 34)
    while size > 14 and c.stringWidth(title, bold, size) > page_w * MM_TO_PT * 0.80:
        size -= 1

    # `y` names the middle of the text block, so each style only has to say
    # where its clear space is: rule, title, subtitle and detail are centred
    # on it. The block runs from 1.25 x size above the baseline to 0.62 x size
    # plus the detail line below it.
    title_y = page_h * MM_TO_PT * text["y"] - size * 0.315 + 10

    c.setFillColor(Color(*text["fg"]))
    c.setFont(bold, size)
    c.drawCentredString(cx, title_y, title)

    if subtitle:
        c.setFont(regular, 13)
        c.setFillColor(Color(*text["sub"]))
        c.drawCentredString(cx, title_y - size * 0.62 - 3, subtitle)

    # A thin rule above the title, clear of the ascenders. Placed by the font's
    # actual cap height rather than a guess, or it draws through the lettering.
    c.setStrokeColor(Color(*text["rule"]))
    c.setLineWidth(0.7)
    rule_y = title_y + size * 1.25
    c.line(cx - 22 * MM_TO_PT, rule_y, cx + 22 * MM_TO_PT, rule_y)

    if detail:
        c.setFont(regular, 9)
        c.setFillColor(Color(*text["detail"]))
        c.drawCentredString(cx, title_y - size * 0.62 - 20, detail)

    _draw_label(c, page_h, label, bold)
    c.showPage()


def _draw_label(c, page_h, label, bold):
    """A tag in the corner of a preview page, naming the style to ask for."""
    if not label:
        return
    c.setFillColor(Color(0, 0, 0, alpha=0.6))
    c.roundRect(10, page_h * MM_TO_PT - 32, c.stringWidth(label, bold, 11) + 16,
                22, 4, stroke=0, fill=1)
    c.setFillColor(white)
    c.setFont(bold, 11)
    c.drawString(18, page_h * MM_TO_PT - 25, label)


def render(pages, out_path, page_w, page_h, title="Album", subtitle="", detail="",
           target_dpi=300.0, do_enhance=True, cover=True, page_numbers=True,
           background=(1.0, 1.0, 1.0), workdir=None, progress=None,
           smart_crop=True, cover_photo=None, look="vivid", captions="auto",
           cover_style=covers.DEFAULT_STYLE, cover_pool=(), cover_named=(),
           cover_gap=240, page_style="paper", route_stops=None):
    """Write the PDF. Returns a dict of what happened."""
    if not HAS_REPORTLAB:
        raise RuntimeError("reportlab is not installed - cannot write a PDF")

    tmp = workdir or tempfile.mkdtemp(prefix="album-")
    owns_tmp = workdir is None
    os.makedirs(tmp, exist_ok=True)

    fonts = _register_fonts()
    regular, bold = fonts

    pw_pt, ph_pt = page_w * MM_TO_PT, page_h * MM_TO_PT
    c = rl_canvas.Canvas(out_path, pagesize=(pw_pt, ph_pt))
    c.setTitle(title)
    c.setAuthor("photoalbum")
    c.setSubject(subtitle or "Event album")

    style_used = None
    if cover:
        cover_img, text = None, None
        if cover_photo is not None:
            try:
                cover_img, text, style_used = _cover_image(
                    cover_style, cover_photo, cover_pool or [cover_photo],
                    page_w, page_h, target_dpi, tmp, look=look,
                    named=cover_named, gap_seconds=cover_gap)
            except Exception:
                cover_img = None      # a plain cover is better than no album
        _draw_cover(c, page_w, page_h, title, subtitle, detail, fonts,
                    image_path=cover_img, text=text)

    if route_stops:
        _draw_route_page(c, pw_pt, ph_pt, route_stops, title, subtitle)

    all_notes = {}
    prepared = {}
    skipped_photos = []
    placed = 0
    total = sum(len(p.photos) for p in pages)

    for page in pages:
        page_look = page.look or look
        c.setFillColor(Color(*background))
        c.rect(0, 0, pw_pt, ph_pt, stroke=0, fill=1)

        # On an overlaid page the background has to be rendered before the
        # inset can be positioned - the decision needs the finished pixels.
        bg_slot = next((s for s in page.slots if s.bleed and s.photo), None)
        inset_slots = [s for s in page.slots if s.frame and s.photo]
        if bg_slot is not None and inset_slots:
            bg_path, bg_notes = _prepare(bg_slot.photo, bg_slot, target_dpi, tmp,
                                         do_enhance, background=background,
                                         smart_crop=smart_crop, look=page_look)
            prepared[id(bg_slot)] = (bg_path, bg_notes)
            if len(inset_slots) == 1:
                ins = inset_slots[0]
                spot = choose_inset_position(bg_path, ins.w / page_w, ins.h / page_h)
                if spot:
                    ins.x, ins.y = spot[0] * page_w, spot[1] * page_h
            else:
                # A stacked pair moves as one, to whichever side hides less.
                choose_inset_side(bg_path, inset_slots, page_w, page_h)

        if page_style in ("blend", "float", "paper"):
            def prep(photo, slot, _look=page_look):
                key = id(slot)
                if key in prepared:
                    return prepared.pop(key)
                return _prepare(photo, slot, target_dpi, tmp, do_enhance,
                                background=background, smart_crop=smart_crop,
                                look=_look)
            # The inset decision above used the background at its own size;
            # keep that file for the grown background slot too.
            if bg_slot is not None and id(bg_slot) in prepared:
                prepared.pop(id(bg_slot))
            try:
                path, done, tint, dark = _blend_page(
                    page, page_w, page_h, target_dpi, tmp, prep,
                    float_photos=(page_style == "float"),
                    paper=(page_style == "paper"))
            except (FileNotFoundError, OSError):
                path, done, tint, dark = None, [], None, False
                skipped_photos.extend(s.photo.name for s in page.slots if s.photo)
            if path:
                over = 1.5 * MM_TO_PT
                c.drawImage(path, -over, -over, width=pw_pt + 2 * over,
                            height=ph_pt + 2 * over, preserveAspectRatio=False)
            if path and page.kind == "opener":
                _draw_opener_text(c, pw_pt, ph_pt, page.title, page.subtitle)
            for slot, notes in done:
                if notes:
                    all_notes[slot.photo.name] = notes
                placed += 1
                if progress:
                    progress(placed, total, slot.photo.name)
            if captions and captions != "off":
                text = next((t for t in (places.caption(p, captions)
                                         for p in page.photos if p) if t), "")
                if text and tint is not None and not page.is_bleed:
                    # On the paper margin: small spaced capitals in a tone of
                    # the paper, the way an editorial caption sits.
                    _, light = _register_display_fonts()
                    ink = (Color(0.82, 0.82, 0.84) if dark else
                           Color(*[max(0.0, c / 255.0 * 0.42) for c in tint]))
                    _draw_spaced(c, pw_pt / 2.0, 6.0 * MM_TO_PT, text.upper(),
                                 light, 6.5, 6.5 * 0.28, ink)
                elif text and tint is None:
                    # Over the picture, so white with a faint shadow.
                    _draw_spaced(c, pw_pt / 2.0, 6.0 * MM_TO_PT, text, regular,
                                 6.8, 0.4, Color(1, 1, 1, alpha=0.9), shadow=0.45)
            c.showPage()
            continue

        for slot in page.slots:
            if slot.photo is None:
                continue
            try:
                if id(slot) in prepared:
                    path, notes = prepared.pop(id(slot))
                else:
                    path, notes = _prepare(slot.photo, slot, target_dpi, tmp,
                                           do_enhance, background=background,
                                           smart_crop=smart_crop, look=page_look)
            except (FileNotFoundError, OSError):
                # A photo removed from the inbox mid-run leaves a gap on the
                # page rather than losing the whole album.
                skipped_photos.append(slot.photo.name)
                continue
            if notes:
                all_notes[slot.photo.name] = notes
            slot.photo.enhanced_path = path

            # reportlab measures from the bottom-left; slots are top-left.
            x_pt = slot.x * MM_TO_PT
            y_pt = (page_h - slot.y - slot.h) * MM_TO_PT
            w_pt, h_pt = slot.w * MM_TO_PT, slot.h * MM_TO_PT

            if slot.frame:
                f = slot.frame * MM_TO_PT
                if slot.shadow:
                    # A soft drop shadow, faked with a few offset grey rects -
                    # reportlab has no blur, and this reads convincingly enough
                    # once printed.
                    for step in range(5, 0, -1):
                        g = 0.62 + step * 0.055
                        off = step * 0.9
                        c.setFillColor(Color(g, g, g))
                        c.rect(x_pt - f + off * 0.5, y_pt - f - off,
                               w_pt + 2 * f, h_pt + 2 * f, stroke=0, fill=1)
                c.setFillColor(white)
                c.rect(x_pt - f, y_pt - f, w_pt + 2 * f, h_pt + 2 * f,
                       stroke=0, fill=1)

            c.drawImage(path, x_pt, y_pt, width=w_pt, height=h_pt,
                        preserveAspectRatio=False, anchor="c")
            placed += 1
            if progress:
                progress(placed, total, slot.photo.name)

        # A quiet line in the bottom margin saying where and when. Skipped on
        # full-bleed pages, which have no margin to put it in.
        if captions and captions != "off" and not page.is_bleed:
            # Use the first photo on the page that actually has something to
            # say. Reading only the first slot meant one GPS-less frame at the
            # top of a page silenced the caption for everything beside it.
            text = ""
            for candidate in page.photos:
                if candidate is None:
                    continue
                text = places.caption(candidate, captions)
                if text:
                    break
            if text:
                c.setFont(regular, 6.8)
                c.setFillColor(Color(0.52, 0.52, 0.55))
                c.drawCentredString(pw_pt / 2.0, 7.0 * MM_TO_PT, text)

        if page_numbers and not page.is_bleed:
            c.setFont(regular, 7.5)
            c.setFillColor(Color(0.55, 0.55, 0.57))
            c.drawCentredString(pw_pt / 2.0, 6.5 * MM_TO_PT, str(page.index + 1))

        c.showPage()

    c.save()

    if owns_tmp:
        shutil.rmtree(tmp, ignore_errors=True)

    return {"pages_written": len(pages) + (1 if cover else 0),
            "cover_style": style_used,
            "photos_placed": placed, "enhancements": all_notes,
            "vanished": skipped_photos,
            "size_bytes": os.path.getsize(out_path) if os.path.exists(out_path) else 0}


def render_covers(out_path, page_w, page_h, title, subtitle, detail, hero, pool,
                  styles=covers.STYLES, dpi=110.0, look="vivid", named=(),
                  gap_seconds=240):
    """Write one page per cover style, so the choice can be made by eye.

    Rendered well below print resolution: this is for choosing, not printing.
    Returns the styles in page order (a style that fell back is left out).
    """
    tmp = tempfile.mkdtemp(prefix="album-covers-")
    fonts = _register_fonts()
    c = rl_canvas.Canvas(out_path, pagesize=(page_w * MM_TO_PT, page_h * MM_TO_PT))
    c.setTitle(title + " - cover styles")
    src = covers.new_source(page_w, page_h, dpi)
    shown = []
    try:
        for style in styles:
            path, text, used = _cover_image(style, hero, pool, page_w, page_h, dpi,
                                            tmp, look=look, named=named,
                                            gap_seconds=gap_seconds, src=src,
                                            name="_cover_" + style)
            if used != style:
                continue
            _draw_cover(c, page_w, page_h, title, subtitle, detail, fonts,
                        image_path=path, text=text, label=style)
            shown.append(style)
        c.save()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return shown
