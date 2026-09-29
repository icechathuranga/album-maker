"""Conservative, conditional print preparation.

Every correction here is applied only when the measurements say the photo
needs it, and every one is capped. The goal is a print that looks like the
photo the user remembers taking - not a filtered version of it.

Two passes run, in order:

  correction   measured faults only - underexposure, flat contrast, grain -
               each applied only when the numbers say so, and each capped.

  the look     a deliberate grade on top: vibrance, an S-curve, a touch of
               warmth and local clarity. This is the difference between a
               photo that has been fixed and one that has been finished.

Deliberately NOT done:
  * full grey-world white balance - it neutralises sunsets, candlelight and
    stage lighting. Only a fraction of a cast measured on grey surfaces is
    corrected, one light per chapter (see colour.py), so a warm room is
    cooled but a sunset stays a sunset.
  * aggressive upscaling - invented detail looks worse in print than a
    smaller, honest reproduction, so the layout demotes small photos instead
  * flat saturation boosts - skin goes orange long before a dull sky improves,
    which is why the look uses vibrance instead
"""
from .deps import np, cv2, HAS_CV2, Image, ImageEnhance, ImageFilter
from . import colour

# ---------------------------------------------------------------------------
# Looks
#
# A "look" is the deliberate grade applied on top of the corrective work: the
# difference between a photo that is merely fixed and one that looks like it
# was finished. Values are kept moderate because these are printed - screens
# forgive heavy saturation, ink does not, and skin is the first thing to go.
# ---------------------------------------------------------------------------
LOOKS = {
    # nothing beyond the corrective pass
    "natural": {"vibrance": 0.00, "saturation": 1.00, "scurve": 1.00,
                "warmth": 0.00, "clarity": 0.00},

    # the everyday choice: punchier colour, deeper contrast, still believable
    "vivid":   {"vibrance": 0.34, "saturation": 1.06, "scurve": 1.22,
                "warmth": 0.02, "clarity": 0.18},

    # golden, holiday-postcard feel
    "warm":    {"vibrance": 0.24, "saturation": 1.04, "scurve": 1.14,
                "warmth": 0.075, "clarity": 0.12},

    # muted highlights, lifted blacks - a printed-film look
    "film":    {"vibrance": 0.12, "saturation": 0.96, "scurve": 1.06,
                "warmth": 0.035, "clarity": 0.06,
                "lift": 0.055, "rolloff": 0.05},

    "mono":    {"vibrance": 0.0, "saturation": 0.0, "scurve": 1.26,
                "warmth": 0.0, "clarity": 0.22},

    # portraits: gentle contrast and eased highlights, kind to skin
    "soft":    {"vibrance": 0.18, "saturation": 1.00, "scurve": 1.08,
                "warmth": 0.02, "clarity": 0.00, "rolloff": 0.06},

    # daylight scenery: a little cooler, more local contrast
    "crisp":   {"vibrance": 0.30, "saturation": 1.04, "scurve": 1.20,
                "warmth": -0.03, "clarity": 0.24},

    # evening and lamplight: warmer than `warm`, highlights kept soft
    "golden":  {"vibrance": 0.26, "saturation": 1.05, "scurve": 1.16,
                "warmth": 0.10, "clarity": 0.10, "rolloff": 0.03},
}
DEFAULT_LOOK = "vivid"
AUTO = "auto"

# Caps. Nothing in this module may exceed these regardless of measurements.
MAX_BRIGHTEN = 1.55
MAX_LEVELS_SHIFT = 0.14      # fraction of the tonal range a level may move
CLAHE_CLIP = 2.0
MAX_SATURATION = 1.12
MAX_SHARPEN_PERCENT = 65


def _auto_levels(img, black_pct=0.5, white_pct=99.6, strength=1.0):
    """Stretch the tonal range to the histogram's real ends, gently.

    Uses percentiles rather than absolute min/max so a single blown specular
    highlight or one dead pixel cannot dictate the whole curve.
    """
    a = np.asarray(img, dtype=np.float32)
    lum = a.mean(axis=2)
    lo = float(np.percentile(lum, black_pct))
    hi = float(np.percentile(lum, white_pct))
    if hi - lo < 8:
        return img, False

    # Cap how far the endpoints may travel, so a low-contrast-on-purpose
    # photo (fog, backlight, mood lighting) is not forced to full range.
    max_shift = 255.0 * MAX_LEVELS_SHIFT
    lo = max(0.0, min(lo, max_shift)) * strength
    hi = min(255.0, max(hi, 255.0 - max_shift))
    if hi <= lo + 1:
        return img, False

    scale = 255.0 / (hi - lo)
    if abs(scale - 1.0) < 0.02 and lo < 1.0:
        return img, False

    out = np.clip((a - lo) * scale, 0, 255).astype(np.uint8)
    return Image.fromarray(out), True


def _local_contrast(img, clip=CLAHE_CLIP):
    """CLAHE on the L channel only - lifts detail without shifting colour."""
    if not HAS_CV2:
        return img, False
    a = np.asarray(img, dtype=np.uint8)
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
    l, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a_ch, b_ch))
    return Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)), True


def _brighten(img, factor):
    factor = float(np.clip(factor, 1.0, MAX_BRIGHTEN))
    if factor <= 1.01:
        return img, False
    return ImageEnhance.Brightness(img).enhance(factor), True


def _saturate(img, factor):
    factor = float(np.clip(factor, 1.0, MAX_SATURATION))
    if factor <= 1.01:
        return img, False
    return ImageEnhance.Color(img).enhance(factor), True


def _sharpen(img, percent):
    """Unsharp mask. Applied last, after any resize, and only when measured soft."""
    percent = int(np.clip(percent, 0, MAX_SHARPEN_PERCENT))
    if percent <= 0:
        return img, False
    radius = 1.4 if max(img.size) > 2000 else 1.0
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent,
                                              threshold=3)), True


def _denoise(img, sigma):
    """Only for genuinely grainy frames - denoising costs fine detail."""
    if not HAS_CV2 or sigma < 7.0:
        return img, False
    a = np.asarray(img, dtype=np.uint8)
    strength = float(np.clip((sigma - 6.0) * 1.1, 1.0, 5.0))
    out = cv2.fastNlMeansDenoisingColored(a, None, strength, strength, 7, 21)
    return Image.fromarray(out), True


def _tone_recover(img, shadows=0.0, highlights=0.0):
    """Open up crushed shadows and ease clipped highlights through one curve.

    Unlike the film look's lift, black stays black: the curve only bends in
    the shadows' upper part, and in the highlights just below white, so
    detail comes back without the picture going grey.
    """
    if shadows <= 0.001 and highlights <= 0.001:
        return img, False
    x = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    y = x + shadows * 4.0 * x * (1.0 - x) ** 3 - highlights * 4.0 * x ** 3 * (1.0 - x)
    lut = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(lut[np.asarray(img, dtype=np.uint8)]), True


def _vibrance(img, amount):
    """Saturation that spares already-vivid colours - and therefore skin.

    A flat blanket saturation boost turns faces orange long before a dull sky
    improves. Vibrance scales each pixel by how unsaturated it already is, so
    the muted parts of the frame come up and the strong colours stay put.
    """
    if amount <= 0.001 or not HAS_CV2:
        return img, False
    rgb = np.asarray(img, dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0
    hsv[:, :, 1] = np.clip(sat * (1.0 + amount * (1.0 - sat)), 0.0, 1.0) * 255.0
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return Image.fromarray(out), True


def _scurve(img, strength, lift=0.0, rolloff=0.0):
    """Gentle S-curve contrast through a lookup table.

    Steepening the midtones while easing both ends is what gives a print
    depth; a straight contrast multiplier just clips the ends off instead.
    """
    if strength <= 1.001 and lift <= 0.001:
        return img, False
    x = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    a = float(strength)
    xa = np.power(np.clip(x, 1e-6, 1.0), a)
    inv = np.power(np.clip(1.0 - x, 1e-6, 1.0), a)
    y = xa / (xa + inv)
    if lift:
        y = lift + (1.0 - lift) * y          # milky blacks, film style
    if rolloff:
        y = y * (1.0 - rolloff) + rolloff * x
    lut = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(np.asarray(lut)[np.asarray(img, dtype=np.uint8)]), True


def _warmth(img, amount):
    """Shift towards amber - a little goes a very long way on skin."""
    if abs(amount) < 0.002:
        return img, False
    a = np.asarray(img, dtype=np.float32)
    a[:, :, 0] *= (1.0 + amount)
    a[:, :, 2] *= (1.0 - amount * 0.85)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), True


def _clarity(img, amount):
    """Local-contrast punch: a wide-radius unsharp mask, not edge sharpening."""
    if amount <= 0.005:
        return img, False
    radius = max(6, int(min(img.size) * 0.012))
    blurred = img.filter(ImageFilter.GaussianBlur(radius))
    a = np.asarray(img, dtype=np.float32)
    b = np.asarray(blurred, dtype=np.float32)
    out = np.clip(a + (a - b) * amount, 0, 255).astype(np.uint8)
    return Image.fromarray(out), True


# What `auto` may choose for a chapter, by what its photos show. Two or more
# per kind so neighbouring chapters can differ; mono is never chosen for you -
# a random black-and-white chapter reads as a mistake, not a style.
AUTO_CANDIDATES = {
    "dim":    ("film", "warm"),       # dark or grainy: film forgives both
    "lamp":   ("warm", "golden"),     # warm light: go with it, not against it
    "people": ("vivid", "soft"),
    "scenic": ("vivid", "crisp"),
    "mixed":  ("vivid", "warm"),
}


def _chapter_kind(photos):
    """Which kind of light and subject a chapter is, from its measurements."""
    def med(key, default):
        vals = [(p.scores or {}).get(key, default) for p in photos]
        return float(np.median(vals)) if vals else default
    people = np.mean([1.0 if (p.scores or {}).get("faces", 0) else 0.0
                      for p in photos]) if photos else 0.0
    grey_b = [(p.scores or {}).get("cast_b", 0.0) for p in photos
              if (p.scores or {}).get("cast_l", -1) >= 0]
    if med("mean_brightness", 128.0) < 90 or med("raw_noise_sigma", 0.0) > 5.5:
        return "dim"
    # Measured before correction. Sunlit sand and brown earth read about 7;
    # halls under tungsten and evening light read 8 and up.
    if grey_b and float(np.median(grey_b)) > 8.0:
        return "lamp"
    if people >= 0.6:
        return "people"
    if people <= 0.4 and med("mean_brightness", 128.0) > 115:
        return "scenic"
    return "mixed"


def _seeded(text):
    """A stable number from text. Python's hash() changes between runs."""
    import zlib
    return zlib.crc32(text.encode("utf-8"))


def auto_looks(pages, seed=""):
    """Give every page one look, chosen per chapter of the day.

    Looks vary the way a photographer's edit does - with the light and the
    subject - and the pick among suitable looks is seeded from the event, so
    it seems random but the same folder always gives the same album. Every
    photo on a page shares its look, and two chapters in a row avoid
    repeating one. Returns {chapter: look} and sets page.look.
    """
    chapters = {}
    for pg in pages:
        for p in pg.photos:
            chapters.setdefault(p.segment, []).append(p)

    chosen, previous = {}, None
    for seg in sorted(chapters):
        options = list(AUTO_CANDIDATES[_chapter_kind(chapters[seg])])
        if previous in options and len(options) > 1:
            options.remove(previous)
        look = options[_seeded("%s:%d" % (seed, seg)) % len(options)]
        chosen[seg] = previous = look

    for pg in pages:
        segs = [p.segment for p in pg.photos]
        if segs:
            # The chapter most of the page comes from; ties go to the first.
            main = max(segs, key=lambda s: (segs.count(s), -segs.index(s)))
            pg.look = chosen.get(main)
    return chosen


def grade(img, look=DEFAULT_LOOK):
    """Apply a named look. Runs after the corrective pass, before sharpening."""
    spec = LOOKS.get(look)
    if not spec or look == "natural":
        return img, []
    notes = []

    if spec.get("scurve", 1.0) > 1.0 or spec.get("lift"):
        img, done = _scurve(img, spec.get("scurve", 1.0),
                            spec.get("lift", 0.0), spec.get("rolloff", 0.0))
        if done:
            notes.append("contrast curve")
    if spec.get("vibrance"):
        img, done = _vibrance(img, spec["vibrance"])
        if done:
            notes.append("vibrance")
    sat = spec.get("saturation", 1.0)
    if abs(sat - 1.0) > 0.01:
        img = ImageEnhance.Color(img).enhance(sat)
        notes.append("saturation x%.2f" % sat if sat < 1.0 else "colour")
    if spec.get("warmth"):
        img, done = _warmth(img, spec["warmth"])
        if done:
            notes.append("warmth")
    if spec.get("clarity"):
        img, done = _clarity(img, spec["clarity"])
        if done:
            notes.append("clarity")
    return img, notes


def plan(photo):
    """Decide what this photo needs, from the metrics already measured."""
    s = photo.scores or {}
    actions = {
        "levels": False, "local_contrast": False, "brighten": 1.0,
        "saturate": 1.0, "sharpen": 0, "denoise": 0.0,
        "shadows": 0.0, "highlights": 0.0,
    }
    mean = s.get("mean_brightness", 128.0)
    crushed = s.get("crushed_pct", 0.0)
    blown = s.get("blown_pct", 0.0)

    # Underexposed: lift, but never so far that we amplify shadow noise badly.
    if mean < 95:
        actions["brighten"] = min(MAX_BRIGHTEN, 1.0 + (95 - mean) / 130.0)

    # Detail lost in the shadows comes back, unless lifting would just show
    # grain; a hard edge around blown highlights is softened. Neither can
    # invent what was never recorded, so both stay small.
    if crushed > 10 and s.get("raw_noise_sigma", 0.0) < 6.0:
        actions["shadows"] = min(0.14, 0.04 + (crushed - 10) * 0.004)
    if blown > 4:
        actions["highlights"] = min(0.10, 0.03 + (blown - 4) * 0.004)

    # Flat contrast, and not already clipping at either end.
    if s.get("contrast", 1.0) < 0.62 and blown < 6 and crushed < 12:
        actions["levels"] = True
    if s.get("contrast", 1.0) < 0.50 and blown < 10:
        actions["local_contrast"] = True

    # Dull colour, but only where there is colour to recover.
    if s.get("colorfulness", 1.0) < 0.55:
        actions["saturate"] = 1.0 + (0.55 - s.get("colorfulness", 0.55)) * 0.30

    # Slightly soft frames benefit from a little bite; already-crisp ones do not,
    # and truly blurred ones cannot be rescued so we do not try.
    sharp = s.get("sharpness", 1.0)
    if 0.30 <= sharp < 0.72:
        actions["sharpen"] = int(30 + (0.72 - sharp) * 80)
    # Soft at full size but printed small enough to hide it: a little more
    # bite finishes the job. Still capped by MAX_SHARPEN_PERCENT.
    if s.get("sharp_scale", 1.0) < 1.0:
        actions["sharpen"] = max(actions["sharpen"], 45)

    if s.get("raw_noise_sigma", 0.0) > 7.0:
        actions["denoise"] = s.get("raw_noise_sigma", 0.0)

    return actions


def apply(img, photo, enabled=True, extra_sharpen=0, look=DEFAULT_LOOK):
    """Run the plan against a loaded image. Returns (image, list-of-changes)."""
    changes = []
    if not enabled:
        if extra_sharpen:
            img, done = _sharpen(img, extra_sharpen)
            if done:
                changes.append("resize sharpen")
        return img, changes

    actions = plan(photo)

    if actions["denoise"]:
        img, done = _denoise(img, actions["denoise"])
        if done:
            changes.append("denoise")
    img, done = colour.apply(img, photo.scores)
    if done:
        changes.append(done)
    if actions["shadows"] or actions["highlights"]:
        img, done = _tone_recover(img, actions["shadows"], actions["highlights"])
        if done:
            changes.append("recovered %s" % " and ".join(
                n for n, v in (("shadows", actions["shadows"]),
                               ("highlights", actions["highlights"])) if v))
    if actions["brighten"] > 1.0:
        img, done = _brighten(img, actions["brighten"])
        if done:
            changes.append("brighten x%.2f" % actions["brighten"])
    if actions["levels"]:
        img, done = _auto_levels(img)
        if done:
            changes.append("auto levels")
    if actions["local_contrast"]:
        img, done = _local_contrast(img)
        if done:
            changes.append("local contrast")
    if actions["saturate"] > 1.0:
        img, done = _saturate(img, actions["saturate"])
        if done:
            changes.append("saturation x%.2f" % actions["saturate"])

    # The look goes on after the corrections, so it grades a photo that is
    # already exposed and balanced rather than fighting one that is not.
    img, look_notes = grade(img, look)
    changes.extend(look_notes)

    # Sharpening always comes last, once the pixels are at final size.
    total_sharpen = max(actions["sharpen"], extra_sharpen)
    if total_sharpen:
        img, done = _sharpen(img, total_sharpen)
        if done:
            changes.append("sharpen %d%%" % total_sharpen)

    return img, changes
