"""Colour balance: find the real cast, correct part of it, match a chapter.

The old correction averaged the whole frame, so a red dress or a green lawn
counted as a "cast" and a yellow wedding hall with colourful guests did not.
This measures only what ought to be grey - walls, shirts, tablecloths, sky
haze: bright-enough pixels with very little colour - and asks which way they
lean.

Three rules keep it from doing harm:

  * With too little grey in the frame (a sunset, a stage in coloured light)
    there is nothing to measure against, and the photo is left alone.
  * Only part of the cast is removed. A hall under tungsten light should
    still look warm in the album, just not orange.
  * Light belongs to a place, so a chapter of the day - one venue, one light -
    gets one correction. Two photos side by side on a page then match, which
    per-photo balancing cannot promise.

Skin is checked last: if a correction would push faces towards green or blue
it is backed off, because that is the colour error people notice first.
"""
import math

from .deps import np, cv2, HAS_CV2, Image

NEUTRAL_CHROMA = 18.0     # LAB a/b distance from grey that still counts as grey
NEUTRAL_L = (25.0, 90.0)  # too dark is noise, too bright is clipped
MIN_NEUTRAL = 0.03        # share of the frame that must be grey to measure
MIN_CAST = 2.5            # smaller than this nobody sees it
STRENGTH = 0.60           # share of the cast removed
MAX_SHIFT = 10.0          # most LAB units moved, whatever the cast
MAX_GAIN = 0.15           # most any channel is scaled
CHAPTER_AGREE = 8.0       # a photo this far from its chapter is judged alone
SKIN_HUE = (22.0, 80.0)   # degrees in LAB a/b: where skin of every tone sits


def _lab(rgb01):
    """RGB in 0..1 (any shape ending in 3) to float LAB: L 0-100, a/b signed."""
    a = np.asarray(rgb01, dtype=np.float32).reshape(-1, 1, 3)
    return cv2.cvtColor(a, cv2.COLOR_RGB2LAB).reshape(-1, 3)


def _rgb(lab):
    a = np.asarray(lab, dtype=np.float32).reshape(-1, 1, 3)
    return cv2.cvtColor(a, cv2.COLOR_LAB2RGB).reshape(-1, 3)


def measure(rgb, face_boxes=None):
    """What colour the grey parts of this frame lean towards.

    Returns a dict for photo.scores: cast_l/cast_a/cast_b (the mean LAB of the
    grey pixels), neutral_frac, and skin_r/g/b (mean colour of the faces'
    centres, -1 when there are none). Runs on the analysis-size copy.
    """
    out = {"cast_l": -1.0, "cast_a": 0.0, "cast_b": 0.0, "neutral_frac": 0.0,
           "skin_r": -1.0, "skin_g": -1.0, "skin_b": -1.0}
    if not HAS_CV2:
        return out
    small = rgb
    if max(rgb.shape[:2]) > 600:
        k = 600.0 / max(rgb.shape[:2])
        small = cv2.resize(rgb, (max(1, int(rgb.shape[1] * k)),
                                 max(1, int(rgb.shape[0] * k))),
                           interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    grey = ((L > NEUTRAL_L[0]) & (L < NEUTRAL_L[1])
            & (np.hypot(a, b) < NEUTRAL_CHROMA))
    frac = float(grey.mean())
    out["neutral_frac"] = frac
    if frac >= MIN_NEUTRAL:
        out["cast_l"] = float(L[grey].mean())
        out["cast_a"] = float(a[grey].mean())
        out["cast_b"] = float(b[grey].mean())

    # The middle of each face: cheeks and nose, clear of hair and background.
    H, W = rgb.shape[:2]
    samples = []
    for (fx, fy, fw, fh, _s) in face_boxes or []:
        x0, x1 = int((fx + fw * 0.3) * W), int((fx + fw * 0.7) * W)
        y0, y1 = int((fy + fh * 0.35) * H), int((fy + fh * 0.75) * H)
        if x1 - x0 >= 3 and y1 - y0 >= 3:
            samples.append(rgb[y0:y1, x0:x1].reshape(-1, 3))
    if samples:
        skin = np.concatenate(samples).astype(np.float32).mean(axis=0) / 255.0
        out["skin_r"], out["skin_g"], out["skin_b"] = (float(v) for v in skin)
    return out


def _gains(L, a, b, amount):
    """Per-channel RGB gains that move grey (L, a, b) `amount` of the way to neutral."""
    src = _rgb([[L, a, b]])[0]
    dst = _rgb([[L, a * (1.0 - amount), b * (1.0 - amount)]])[0]
    g = dst / np.maximum(src, 1e-4)
    g = g / g.mean() * (dst.mean() / max(src.mean(), 1e-4))   # keep brightness
    return tuple(float(np.clip(v, 1.0 - MAX_GAIN, 1.0 + MAX_GAIN)) for v in g)


def _skin_ok(skin_rgb, gains):
    """Is skin still skin-coloured after these gains?"""
    lab = _lab(np.clip(np.array(skin_rgb) * np.array(gains), 0.0, 1.0))[0]
    hue = math.degrees(math.atan2(lab[2], lab[1]))
    return lab[1] > 3.0 and SKIN_HUE[0] <= hue <= SKIN_HUE[1]


def _plan_one(s, a, b):
    """Gains and a note for one photo, correcting towards cast (a, b)."""
    cast = math.hypot(a, b)
    if cast < MIN_CAST:
        return None
    amount = min(STRENGTH, MAX_SHIFT / cast)
    skin = (s["skin_r"], s["skin_g"], s["skin_b"]) if s.get("skin_r", -1) >= 0 else None
    skin_was_ok = skin is not None and _skin_ok(skin, (1.0, 1.0, 1.0))
    for _ in range(3):
        gains = _gains(s["cast_l"], a, b, amount)
        if not skin_was_ok or _skin_ok(skin, gains):
            break
        amount *= 0.5                     # faces going green: back off
    else:
        return None
    # Plain numbers only: photo.scores is rounded and written to album.json.
    return {"wb_r": gains[0], "wb_g": gains[1], "wb_b": gains[2],
            "wb_hue": math.degrees(math.atan2(b, a)) % 360,
            "wb_from": cast, "wb_to": cast * (1.0 - amount)}


def _describe(hue):
    for limit, name in ((25, "magenta"), (70, "orange"), (110, "yellow"),
                        (160, "yellow-green"), (215, "green"), (290, "blue"),
                        (335, "purple"), (360, "magenta")):
        if hue < limit:
            return name
    return "colour"


def balance(photos):
    """Decide the correction for every photo, one light per chapter.

    Stores wb_r/wb_g/wb_b (channel gains) plus wb_hue/wb_from/wb_to (for the
    report) in each photo's scores, or removes them when it should be left as
    it is. Call after curation, once chapters (photo.segment) are final.
    """
    chapters = {}
    for p in photos:
        s = p.scores or {}
        for k in PLAN_KEYS:
            s.pop(k, None)
        if s.get("cast_l", -1) >= 0:
            chapters.setdefault(p.segment, []).append(p)

    for seg, members in chapters.items():
        med_a = float(np.median([p.scores["cast_a"] for p in members]))
        med_b = float(np.median([p.scores["cast_b"] for p in members]))
        for p in members:
            s = p.scores
            a, b = s["cast_a"], s["cast_b"]
            # One lamp-lit table in a sunlit garden is its own light.
            if math.hypot(a - med_a, b - med_b) <= CHAPTER_AGREE:
                a, b = med_a, med_b
            plan = _plan_one(s, a, b)
            if plan:
                s.update(plan)


PLAN_KEYS = ("wb_r", "wb_g", "wb_b", "wb_hue", "wb_from", "wb_to")


def note(scores):
    """The report line for a planned correction."""
    return "colour balance: %s cast %.1f -> %.1f" % (
        _describe(scores["wb_hue"]), scores["wb_from"], scores["wb_to"])


def apply(img, scores):
    """Apply the planned correction, if any. Returns (image, note or None)."""
    scores = scores or {}
    if "wb_r" not in scores:
        return img, None
    gains = (scores["wb_r"], scores["wb_g"], scores["wb_b"])
    if all(abs(g - 1.0) < 0.004 for g in gains):
        return img, None
    arr = np.asarray(img, dtype=np.float32) * np.array(gains, dtype=np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), note(scores)
