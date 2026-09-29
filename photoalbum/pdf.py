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

from .deps import np, Image, ImageDraw, ImageFont, HAS_REPORTLAB
from . import ingest, enhance, crop, cover as covers, faces as facedet, places

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


# Where an inset may sit, as the top-left corner in page fractions. Both sides
# at three heights: the inset always hugs a margin, so it reads as a deliberate
# placement rather than something dropped in the middle of the picture.
INSET_CANDIDATES = (
    (0.055, 0.055), (0.575, 0.055),      # top corners
    (0.055, 0.320), (0.575, 0.320),      # mid height
    (0.055, 0.585), (0.575, 0.585),      # bottom corners
)


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
        c.drawImage(image_path, 0, 0, width=page_w * MM_TO_PT,
                    height=page_h * MM_TO_PT, preserveAspectRatio=False)
    else:
        c.setFillColor(Color(*bg))
        c.rect(0, 0, page_w * MM_TO_PT, page_h * MM_TO_PT, stroke=0, fill=1)

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

    if label:
        # A tag in the corner of a preview page, naming the style to ask for.
        c.setFillColor(Color(0, 0, 0, alpha=0.6))
        c.roundRect(10, page_h * MM_TO_PT - 32, c.stringWidth(label, bold, 11) + 16,
                    22, 4, stroke=0, fill=1)
        c.setFillColor(white)
        c.setFont(bold, 11)
        c.drawString(18, page_h * MM_TO_PT - 25, label)
    c.showPage()


def render(pages, out_path, page_w, page_h, title="Album", subtitle="", detail="",
           target_dpi=300.0, do_enhance=True, cover=True, page_numbers=True,
           background=(1.0, 1.0, 1.0), workdir=None, progress=None,
           smart_crop=True, cover_photo=None, look="vivid", captions="auto",
           cover_style=covers.DEFAULT_STYLE, cover_pool=(), cover_named=(),
           cover_gap=240):
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
            for ins in inset_slots:
                spot = choose_inset_position(bg_path, ins.w / page_w, ins.h / page_h)
                if spot:
                    ins.x, ins.y = spot[0] * page_w, spot[1] * page_h

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
