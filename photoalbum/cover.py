"""Cover designs.

The cover is the one page everyone sees, so it gets its own module. Each
style composes the whole page as a single picture at print resolution - the
photos, the bands, the blur - and says where the title should go. The title
itself is drawn afterwards as real PDF text so it stays crisp at any size.

    classic   one photo, full bleed, darkened at the foot for the title
    strips    five or six photos cut into tall vertical bars, title below
    collage   a wall of photos with a frosted, blurred band for the title
    split     one large photo over a coloured title band and a row of three
    duotone   one photo in two inks drawn from its own colours, large title
    frame     one photo matted on a light page, like a gallery print

Everything is deterministic: the same folder always gives the same cover.
Multi-photo styles need enough usable photos; with too few they fall back to
`classic` rather than print a half-empty design.
"""
import colorsys

from .deps import np, Image, ImageFilter
from . import ingest, enhance, crop, dedupe
from .curate import SCENE_BITS

STYLES = ("classic", "strips", "collage", "split", "duotone", "frame")
DEFAULT_STYLE = "strips"

CREAM = (246, 243, 236)
CROP_ANALYSIS_EDGE = 900


def _px(mm, dpi):
    return max(1, int(round(mm / 25.4 * dpi)))


# --- choosing photos -------------------------------------------------------

def _quality(p):
    """How fit a photo is to be printed on a cover, before shape is considered.

    The same disqualifiers as the single-photo cover: it is the first thing
    anyone sees, so softness and poor exposure count heavily.
    """
    m = p.scores or {}
    value = p.score
    sharp = m.get("sharpness", 0.0)
    if sharp < 0.72:
        value -= 45.0 * (0.72 - sharp) / 0.72
    if p.technical < 0.80:
        value -= 25.0
    mp = m.get("effective_mp", p.megapixels)
    if mp < 2.0:
        value -= 20.0 * (1.0 - mp / 2.0)
    return value


def _shape_loss(p, aspect):
    """Fraction of the photo thrown away to fit a tile of `aspect` (w/h)."""
    a = p.aspect or 1.0
    return 1.0 - min(a, aspect) / max(a, aspect)


def _face_misfit(p, aspect):
    """How badly the people in a photo overflow a tile of `aspect` (0 = fine).

    A tall strip keeps a narrow column of a wide photo. One face fits; a row
    of six does not, and whichever column is chosen someone is cut in half.
    Only faces comparable in size to the largest count - a stranger in the
    background is not a reason to reject the frame.
    """
    boxes = list(p.face_boxes or [])
    if not boxes:
        return 0.0
    a = p.aspect or 1.0
    biggest = max(w * h for (_x, _y, w, h, _s) in boxes) or 1e-6
    boxes = [b for b in boxes if b[2] * b[3] >= 0.3 * biggest]
    if aspect < a:                    # tile is narrower: width is what is lost
        keep = aspect / a
        span = max(x + w for (x, _y, w, _h, _s) in boxes) - min(b[0] for b in boxes)
    else:                             # tile is flatter: height is lost
        keep = a / aspect
        span = max(y + h for (_x, y, _w, h, _s) in boxes) - min(b[1] for b in boxes)
    span *= 1.0 + 2 * crop.FACE_PAD * 0.5     # hair and chin come too
    return float(np.clip((span - keep) / max(span, 1e-6), 0.0, 1.0))


def _same_scene(a, b):
    return (a.dhash is not None and b.dhash is not None
            and dedupe.hamming(a.dhash, b.dhash) <= SCENE_BITS)


def pick_set(pool, n, tile_aspect, first=None, named=(), gap_seconds=240,
             avoid=()):
    """Choose `n` photos that work together on one cover.

    Taking the top n by score gives five frames of the same moment. So this
    picks greedily, and each pick is marked down for sharing a chapter or a
    few minutes with one already chosen, for looking like the same scene,
    and for tipping the set too far towards either people or places. `avoid`
    are photos already elsewhere on the cover: never picked, and marked down
    against in the same way. Photos named in cover.txt go first, in the order
    written. Returned in the order they were taken.
    """
    chosen = []
    by_name = {p.name.lower(): p for p in pool}
    for name in named:
        p = by_name.get(name)
        if p is not None and p not in chosen:
            chosen.append(p)
    if first is not None and first not in chosen:
        chosen.insert(0, first)
    chosen = chosen[:n]

    avoid = list(avoid)
    rest = [p for p in pool if p not in chosen and p not in avoid]
    while len(chosen) < n and rest:
        people = sum(1 for c in chosen if (c.scores or {}).get("faces", 0))

        def value(p):
            v = _quality(p) - 40.0 * _shape_loss(p, tile_aspect)
            v -= 80.0 * _face_misfit(p, tile_aspect)
            for c in chosen + avoid:
                if _same_scene(p, c):
                    v -= 35.0
                if p.segment == c.segment:
                    v -= 8.0
                if p.taken and c.taken and \
                        abs((p.taken - c.taken).total_seconds()) < gap_seconds:
                    v -= 22.0
            has_people = bool((p.scores or {}).get("faces", 0))
            half = len(chosen) / 2.0
            if has_people and people > half + 0.5:
                v -= 25.0
            if not has_people and (len(chosen) - people) > half + 0.5:
                v -= 25.0
            return v

        best = max(rest, key=value)
        chosen.append(best)
        rest.remove(best)

    # The pool arrives in the order the photos were taken; keep to it.
    order = {id(p): i for i, p in enumerate(pool)}
    chosen.sort(key=lambda p: order.get(id(p), -1))
    return chosen


# --- turning a photo into a tile -------------------------------------------

class _Source:
    """Loads each photo once, at no more pixels than the page can use."""

    def __init__(self, long_edge):
        self.long_edge = long_edge
        self.cache = {}

    def get(self, photo):
        key = id(photo)
        if key not in self.cache:
            self.cache[key] = ingest.load(photo, max_long_edge=self.long_edge)
        return self.cache[key]


def _fit(src, photo, w, h, look):
    """The photo cropped to exactly w x h pixels, graded, faces kept whole."""
    img = src.get(photo)
    scale = min(1.0, CROP_ANALYSIS_EDGE / float(max(img.size)))
    small = img if scale >= 1.0 else img.resize(
        (max(8, int(img.width * scale)), max(8, int(img.height * scale))),
        Image.BILINEAR)
    box = crop.find(np.asarray(small.convert("RGB"), dtype=np.uint8),
                    w / float(h), face_boxes=photo.face_boxes)
    inv = 1.0 / scale
    x, y, cw, ch = (int(v * inv) for v in box)
    cw, ch = max(8, min(cw, img.width)), max(8, min(ch, img.height))
    x, y = max(0, min(x, img.width - cw)), max(0, min(y, img.height - ch))
    tile = img.crop((x, y, x + cw, y + ch))
    tile = tile.resize((w, h), Image.LANCZOS if cw > w else Image.BICUBIC)
    tile, _ = enhance.apply(tile, photo, enabled=True, look=look,
                            extra_sharpen=25 if cw < w * 0.87 else 0)
    return tile.convert("RGB")


def _accent(img, value=0.22, max_sat=0.55):
    """A deep colour taken from the photo itself, for bands and inks.

    Pixels are weighted by how saturated and bright they are, so the result
    is the colour you notice in the picture rather than its muddy average.
    """
    a = np.asarray(img.resize((48, 48), Image.BILINEAR), dtype=np.float32) / 255.0
    mx, mn = a.max(axis=2), a.min(axis=2)
    weight = (mx - mn) / (mx + 1e-6) * mx + 1e-4
    mean = (a * weight[..., None]).sum(axis=(0, 1)) / weight.sum()
    h, s, _v = colorsys.rgb_to_hsv(*[float(c) for c in mean])
    r, g, b = colorsys.hsv_to_rgb(h, min(max_sat, s), value)
    return (int(r * 255), int(g * 255), int(b * 255))


def _unit(rgb):
    return tuple(c / 255.0 for c in rgb)


def _ramp_darken(img, start=0.42, strength=0.90):
    """Darken from `start` (fraction of height) down to the foot of the page."""
    a = np.asarray(img, dtype=np.float32)
    y = np.linspace(0.0, 1.0, a.shape[0], dtype=np.float32)
    ramp = np.clip((y - start) / (1.0 - start), 0.0, 1.0) ** 1.05
    a *= (1.0 - strength * ramp)[:, None, None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# Text over a photo or a dark band, and text on a light page.
LIGHT_TEXT = {"fg": (1.0, 1.0, 1.0), "sub": (0.88, 0.88, 0.90),
              "rule": (0.78, 0.78, 0.80), "detail": (0.80, 0.80, 0.82)}


def _dark_text(ink):
    ink = _unit(ink)
    soft = tuple(0.45 * c + 0.55 * 0.42 for c in ink)
    return {"fg": ink, "sub": soft, "rule": soft, "detail": soft}


# --- the styles ------------------------------------------------------------
#
# Each takes (W, H, hero, pool, ctx) and returns (image, text) where text says
# where the title's baseline sits (fraction of the height, from the bottom),
# its largest size in points, and its colours. None means "not enough photos".

def _classic(W, H, hero, pool, ctx):
    img = _fit(ctx["src"], hero, W, H, ctx["look"])
    img = _ramp_darken(img)
    return img, dict(LIGHT_TEXT, y=0.175, size=34)


def _strips(W, H, hero, pool, ctx):
    n = 5 if W <= H else 6
    photo_h = int(H * 0.76)
    gap = max(2, int(W * 0.012))
    strip_w = (W - gap * (n - 1)) // n
    named = ctx["named"]
    photos = pick_set(pool, n, strip_w / float(photo_h),
                      first=hero if named else None, named=named,
                      gap_seconds=ctx["gap"])
    if len(photos) < n:
        return None

    page = Image.new("RGB", (W, H), CREAM)
    x = 0
    for i, p in enumerate(photos):
        # The last strip takes up the rounding so the right edge is flush.
        w = W - x if i == n - 1 else strip_w
        page.paste(_fit(ctx["src"], p, w, photo_h, ctx["look"]), (x, 0))
        x += w + gap
    ink = _accent(page.crop((0, 0, W, photo_h)), value=0.20)
    band_mid = 1.0 - (photo_h + (H - photo_h) * 0.46) / H
    return page, dict(_dark_text(ink), y=band_mid, size=34)


def _justified_rows(photos, rows, W, H, gap):
    """Lay photos in rows of equal height, widths following their shapes."""
    per_row = [len(photos) // rows + (1 if i < len(photos) % rows else 0)
               for i in range(rows)]
    row_h = (H - gap * (rows - 1)) / float(rows)
    cells, k, y = [], 0, 0.0
    for count in per_row:
        group = photos[k:k + count]
        k += count
        # Clamp shapes so one panorama cannot squeeze its row-mates to slivers.
        aspects = [float(np.clip(p.aspect or 1.0, 0.62, 1.7)) for p in group]
        free = W - gap * (count - 1)
        x = 0.0
        for j, (p, a) in enumerate(zip(group, aspects)):
            w = free * a / sum(aspects)
            x1 = W if j == count - 1 else int(round(x + w))
            cells.append((p, int(round(x)), int(round(y)), x1,
                          int(round(y + row_h))))
            x = x1 + gap
        y += row_h + gap
    return cells


def _collage(W, H, hero, pool, ctx):
    # An odd number of rows, so the frosted band can be exactly the middle
    # one: a band cutting across two rows leaves slivers of photo either side.
    rows = 5 if W <= H else 3
    n = min(13 if W <= H else 10, len(pool))
    if n < 8:
        return None
    photos = pick_set(pool, n, 1.0, first=hero, named=ctx["named"],
                      gap_seconds=ctx["gap"] / 2.0)
    # The featured photo leads the top row. Left in time order it could fall
    # in the middle row, under the frosted band, and be blurred out.
    photos.remove(hero)
    photos.insert(0, hero)
    gap = max(2, int(W * 0.006))
    page = Image.new("RGB", (W, H), (18, 18, 20))
    for p, x0, y0, x1, y1 in _justified_rows(photos, rows, W, H, gap):
        page.paste(_fit(ctx["src"], p, x1 - x0, y1 - y0, ctx["look"]), (x0, y0))

    # The frosted band: that part of the wall blurred to a soft wash and
    # knocked back, so the photos stay recognisable behind the lettering.
    row_h = (H - gap * (rows - 1)) / float(rows)
    mid_row = rows // 2
    top = int(round(mid_row * (row_h + gap))) - gap
    bot = int(round(mid_row * (row_h + gap) + row_h)) + gap
    band = page.crop((0, top, W, bot)).filter(ImageFilter.GaussianBlur(W * 0.018))
    a = np.asarray(band, dtype=np.float32) * 0.52 + 14.0
    page.paste(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), (0, top))
    line = max(1, int(H * 0.0012))
    edge = Image.new("RGB", (W, line), (235, 235, 238))
    page.paste(edge, (0, top))
    page.paste(edge, (0, bot - line))
    mid = 1.0 - (top + (bot - top) * 0.52) / H
    return page, dict(LIGHT_TEXT, y=mid, size=38)


def _split(W, H, hero, pool, ctx):
    n = 3 if W <= H else 4
    hero_h, band_h = int(H * 0.58), int(H * 0.17)
    gap = max(2, int(W * 0.008))
    row_y = hero_h + band_h
    tile_w = (W - gap * (n - 1)) // n
    photos = pick_set(pool, n, tile_w / float(H - row_y),
                      named=[m for m in ctx["named"] if m != hero.name.lower()],
                      gap_seconds=ctx["gap"], avoid=[hero])
    if len(photos) < n:
        return None

    top = _fit(ctx["src"], hero, W, hero_h, ctx["look"])
    ink = _accent(top, value=0.24)
    page = Image.new("RGB", (W, H), ink)
    page.paste(top, (0, 0))
    x = 0
    for i, p in enumerate(photos):
        w = W - x if i == n - 1 else tile_w
        page.paste(_fit(ctx["src"], p, w, H - row_y, ctx["look"]), (x, row_y))
        x += w + gap
    mid = 1.0 - (hero_h + band_h * 0.50) / H
    return page, dict(LIGHT_TEXT, y=mid, size=34)


def _duotone(W, H, hero, pool, ctx):
    img = _fit(ctx["src"], hero, W, H, "natural")
    dark = np.array(_accent(img, value=0.10, max_sat=0.70), dtype=np.float32)
    light = np.array((250, 240, 222), dtype=np.float32)
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    g = np.clip((g - 0.04) / 0.92, 0.0, 1.0) ** 1.1
    a = dark * (1.0 - g[..., None]) + light * g[..., None]
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    img = _ramp_darken(img, start=0.50, strength=0.70)
    return img, dict(LIGHT_TEXT, y=0.16, size=46)


def _frame(W, H, hero, pool, ctx):
    side, top = int(W * 0.11), int(H * 0.09)
    photo_w = W - 2 * side
    photo_h = int(H * (0.62 if W <= H else 0.60))
    page = Image.new("RGB", (W, H), CREAM)
    img = _fit(ctx["src"], hero, photo_w, photo_h, ctx["look"])
    # A hairline keyline, the way a print is matted.
    line = max(1, int(W * 0.0012))
    page.paste(Image.new("RGB", (photo_w + 2 * line, photo_h + 2 * line),
                         (200, 196, 188)), (side - line, top - line))
    page.paste(img, (side, top))
    ink = _accent(img, value=0.20)
    below = top + photo_h
    mid = 1.0 - (below + (H - below) * 0.46) / H
    return page, dict(_dark_text(ink), y=mid, size=32)


_BUILDERS = {"classic": _classic, "strips": _strips, "collage": _collage,
             "split": _split, "duotone": _duotone, "frame": _frame}


def compose(style, hero, pool, page_w, page_h, dpi, look="vivid", named=(),
            gap_seconds=240, src=None):
    """Build the cover picture. Returns (PIL image, text layout, style used).

    A multi-photo style without enough photos quietly becomes `classic`.
    """
    W, H = _px(page_w, dpi), _px(page_h, dpi)
    # Headroom over the page's long edge: a wide photo on a tall page is
    # cropped to a sliver of its width, and that sliver still has to fill it.
    src = src or _Source(int(max(W, H) * 1.5))
    ctx = {"src": src, "look": look, "named": list(named), "gap": gap_seconds}
    builder = _BUILDERS.get(style, _BUILDERS[DEFAULT_STYLE])
    out = builder(W, H, hero, list(pool), ctx)
    if out is None:
        style, out = "classic", _classic(W, H, hero, pool, ctx)
    img, text = out
    return img, text, style if style in _BUILDERS else DEFAULT_STYLE


def new_source(page_w, page_h, dpi):
    """A photo loader that can be shared across several covers."""
    return _Source(max(_px(page_w, dpi), _px(page_h, dpi)))
