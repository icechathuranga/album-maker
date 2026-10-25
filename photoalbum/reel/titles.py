"""Title text for reels, in the same lettering as the album covers.

Rendered by PIL as a transparent full-frame PNG, because ffmpeg's drawtext
cannot letter-space - and the spaced capitals are the look. A soft, blurred
shade sits under the letters so they read over a bright sky; on a phone
screen that is what keeps white text legible.
"""
from ..deps import Image, ImageDraw, ImageFilter, ImageFont
from .. import fonts

TRACK_TITLE = 0.30      # letter spacing, share of the size (as the cover)
TRACK_SUB = 0.32


def _font(paths, size):
    path = fonts.first(paths)
    return ImageFont.truetype(path, size) if path else ImageFont.load_default()


def _spaced_width(font, text, tracking):
    return sum(font.getlength(ch) for ch in text) + tracking * max(0, len(text) - 1)


def _draw_spaced(draw, cx, y, text, font, tracking, fill):
    x = cx - _spaced_width(font, text, tracking) / 2.0
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += font.getlength(ch) + tracking


def _fit(text, W, size, limit, min_size):
    while size > min_size:
        f = _font(fonts.SERIF, size)
        if _spaced_width(f, text, size * TRACK_TITLE) <= W * limit:
            return f, size
        size -= 2
    return _font(fonts.SERIF, min_size), min_size


def render(title, subtitle, W, H, style="overlay", y=None):
    """A WxH RGBA image with the title laid out for `style`.

    overlay  large, across the upper-middle of the frame (over the hook)
    top      smaller, high in the frame, clear of faces lower down
    Both keep out of the bottom fifth and right edge, where TikTok and Reels
    put their buttons and captions. `y` (0-1) places the block's centre -
    the editor sets it away from the faces in the opening shot.
    """
    title = (title or "").upper()
    sub = (subtitle or "").upper()
    big = style == "overlay"
    centre_y = H * (y if y is not None else (0.40 if big else 0.15))
    # Sizes follow the frame's short side, so a 16:9 title is no larger than
    # a 9:16 one; on a wide frame the title spans less of the width.
    base = min(W, H)
    limit = 0.84 if W <= H else 0.62
    size0 = int(base * (0.095 if big else 0.065))
    words = title.split()
    lines = [title]
    font, size = _fit(title, W, size0, limit, int(base * 0.045))
    # Two lines at a size a phone can read beat one line of small type.
    if len(words) > 1 and size < int(base * 0.085):
        cut = min(range(1, len(words)), key=lambda i: abs(
            len(" ".join(words[:i])) - len(" ".join(words[i:]))))
        lines = [" ".join(words[:cut]), " ".join(words[cut:])]
        font, size = _fit(max(lines, key=len), W, size0, limit, int(base * 0.04))

    sub_font = _font(fonts.LIGHT, max(int(base * 0.03), int(size * 0.28)))
    sub_size = sub_font.size
    lead = size * 1.15
    block = lead * len(lines) + (sub_size * 2.4 if sub else 0)
    top = centre_y - block / 2.0

    text_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shade = Image.new("L", (W, H), 0)
    td, sd = ImageDraw.Draw(text_layer), ImageDraw.Draw(shade)
    y = top
    for ln in lines:
        _draw_spaced(td, W / 2.0, y, ln, font, size * TRACK_TITLE, (255, 255, 255, 255))
        _draw_spaced(sd, W / 2.0, y, ln, font, size * TRACK_TITLE, 255)
        y += lead
    if sub:
        y += sub_size * 0.6
        tw = _spaced_width(sub_font, sub, sub_size * TRACK_SUB)
        _draw_spaced(td, W / 2.0, y, sub, sub_font, sub_size * TRACK_SUB,
                     (240, 240, 242, 255))
        _draw_spaced(sd, W / 2.0, y, sub, sub_font, sub_size * TRACK_SUB, 255)
        # Hairlines either side of the subtitle, as on the album cover.
        ry = y + sub_size * 0.55
        gap, margin = sub_size * 1.1, W * 0.08
        if W / 2.0 - tw / 2.0 - gap > margin + 10:
            for x0, x1 in ((margin, W / 2.0 - tw / 2.0 - gap),
                           (W / 2.0 + tw / 2.0 + gap, W - margin)):
                td.line([(x0, ry), (x1, ry)], fill=(235, 235, 238, 230),
                        width=max(1, W // 900))

    shade = shade.filter(ImageFilter.GaussianBlur(max(4, size // 5)))
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow.putalpha(shade.point(lambda v: int(v * 0.62)))
    return Image.alpha_composite(shadow, text_layer)


def caption(text, W, H, y=0.11):
    """A small spaced-capitals line high in the frame: "DAY 2 · KATARAGAMA"."""
    text = (text or "").upper()
    size = max(14, int(min(W, H) * 0.038))
    font = _font(fonts.LIGHT, size)
    track = size * TRACK_SUB
    while size > 12 and _spaced_width(font, text, track) > W * 0.86:
        size -= 1
        font = _font(fonts.LIGHT, size)
        track = size * TRACK_SUB
    top = H * y - size / 2.0
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shade = Image.new("L", (W, H), 0)
    _draw_spaced(ImageDraw.Draw(layer), W / 2.0, top, text, font, track, (255, 255, 255, 255))
    _draw_spaced(ImageDraw.Draw(shade), W / 2.0, top, text, font, track, 255)
    # Thin letters over a bright sky need more shade than the big title.
    shade = shade.filter(ImageFilter.MaxFilter(5)).filter(
        ImageFilter.GaussianBlur(max(4, size // 2)))
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow.putalpha(shade.point(lambda v: int(min(255, v * 0.9))))
    return Image.alpha_composite(shadow, layer)
