"""Classical image-quality scoring - no neural nets, no network, deterministic.

Every metric returns 0..1. The composite score is a weighted sum x100 so it
reads like a percentage in the report.
"""
import math

from .deps import np, cv2, HAS_CV2, Image
from . import ingest, colour
from . import faces as facedet

# Analysis resolution. Laplacian variance is resolution-dependent, so every
# photo is measured at the same long edge or big files score artificially high.
ANALYSIS_EDGE = 1024
TILE_GRID = 6           # 6x6 tiles
TILE_PERCENTILE = 80    # shallow depth-of-field: judge by the sharpest region

# Technical quality answers "is this frame usable?". It is a gate, not the
# ranking: a razor-sharp photo of an empty wall must still lose to a slightly
# soft photo of someone laughing.
TECHNICAL_WEIGHTS = {
    "sharpness": 0.40,
    "exposure": 0.24,
    "contrast": 0.16,
    "noise": 0.10,
    "compression": 0.10,
}

# Content answers "is this frame worth printing?" - this is what ranks.
CONTENT_WEIGHTS = {
    "faces": 0.52,        # people are the subject of an event album
    "isolation": 0.20,    # a subject that pops out of its background
    "colorfulness": 0.16,
    "interest": 0.12,     # tonal variety - a flat frame is a boring frame
}

# Blend used for the final score. Content leads; technical mostly gates.
CONTENT_MIX = 0.62

# Ceiling on the content score for a photo with nobody in it. People do carry
# most event photographs, but this cannot be so low that scenery never competes.
NO_FACE_CAP = 0.84

# Below this technical score a frame is unusable in print at any size.
TECHNICAL_REJECT = 0.35
# Technical only drags the score down until it reaches this comfort level.
TECHNICAL_GATE = 0.58

# A true thumbnail is not a photograph. This is the ONLY place resolution
# affects selection - everywhere else it decides slot size, not rank.
MIN_MEGAPIXELS = 0.15

DEFAULT_WEIGHTS = TECHNICAL_WEIGHTS  # backwards-compatible alias

def _laplacian(gray):
    """4-neighbour Laplacian; cv2 when present, numpy slicing otherwise."""
    if HAS_CV2:
        return cv2.Laplacian(gray, cv2.CV_32F)
    out = np.zeros_like(gray)
    out[1:-1, 1:-1] = (gray[:-2, 1:-1] + gray[2:, 1:-1] +
                       gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * gray[1:-1, 1:-1])
    return out


def sharpness(gray):
    """Tile-wise Laplacian variance, reported at the TILE_PERCENTILE-th tile.

    A portrait with a sharp face and a deliberately soft background scores on
    the face, not on the average - which is what a human would judge.
    """
    lap = _laplacian(gray)
    h, w = lap.shape
    th, tw = max(1, h // TILE_GRID), max(1, w // TILE_GRID)
    variances = []
    for ty in range(TILE_GRID):
        for tx in range(TILE_GRID):
            y0, x0 = ty * th, tx * tw
            y1 = h if ty == TILE_GRID - 1 else y0 + th
            x1 = w if tx == TILE_GRID - 1 else x0 + tw
            tile = lap[y0:y1, x0:x1]
            if tile.size > 16:
                variances.append(float(tile.var()))
    if not variances:
        return 0.0, 0.0, 1.0
    raw = float(np.percentile(variances, TILE_PERCENTILE))
    median = float(np.median(variances))
    # log-normalise: 8 -> unusable, 400+ -> crisp
    lo, hi = math.log10(8.0), math.log10(400.0)
    norm = (math.log10(max(raw, 1e-3)) - lo) / (hi - lo)
    isolation = raw / max(median, 1e-3)
    return float(np.clip(norm, 0.0, 1.0)), raw, isolation


# --- sharpness at the size it prints ----------------------------------------
#
# Everything above is measured at ANALYSIS_EDGE. Shrinking a 12-16 MP phone
# frame to 1024 px hides motion blur of a few pixels completely, so a photo
# that is visibly smeared at A4 scored as "sharp" and was printed full page.
#
# The test is how fast fine detail falls away as the frame is enlarged. Per
# halving of size, crisp photos lose 1.5-3x their Laplacian variance; smeared
# ones lose 4-6x. Measured on 426 printed photos and checked by eye at 100%:
# at 3.2 or below every crop was crisp, above 3.6 every crop was soft.
# Being a ratio, it does not care how much texture the scene has or how hard
# a given phone smooths its output.
PRINT_SHARPNESS = {"strict": 3.0, "normal": 3.4}
BLURRED_AT_HALF = 6.0      # still falling this fast at half size: unprintable
MIN_OCTAVES = 0.4          # too little between sizes to say anything


def _shrink(gray, scale):
    if scale >= 0.999:
        return gray
    h, w = gray.shape
    size = (max(8, int(w * scale)), max(8, int(h * scale)))
    if HAS_CV2:
        return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return np.asarray(Image.fromarray(gray).resize(size, Image.BOX), dtype=np.float32)


def _falloff(ref_var, var, octaves):
    """Detail lost per halving of size between the reference and this scale."""
    if octaves < MIN_OCTAVES:
        return None
    return (ref_var / max(var, 1e-3)) ** (1.0 / octaves)


def _face_region(gray, boxes, pad=0.3):
    """The largest face, padded to take in hair and chin."""
    if not boxes:
        return None
    fx, fy, fw, fh, _ = max(boxes, key=lambda b: b[2] * b[3])
    H, W = gray.shape
    x0, y0 = max(0, int((fx - fw * pad) * W)), max(0, int((fy - fh * pad) * H))
    x1, y1 = min(W, int((fx + fw * (1 + pad)) * W)), min(H, int((fy + fh * (1 + pad)) * H))
    return gray[y0:y1, x0:x1]


def sharp_scale(full_gray, face_boxes=None, threshold=PRINT_SHARPNESS["normal"]):
    """The largest fraction of its own size at which a photo is still crisp.

    Returns (scale, falloff_full, falloff_half, blurred) with scale 1.0, 0.5
    or 0.25. A photo sharp only at half size prints happily at half the size
    its pixels suggest; one still smeared at half size is `blurred`.

    When there are people, the largest face is measured as well and the worse
    of frame and face wins - a crisp background behind a blurred face is the
    frame nobody wants printed.
    """
    H, W = full_gray.shape
    ref_scale = min(1.0, ANALYSIS_EDGE / float(max(H, W)))
    if ref_scale > 0.75:
        return 1.0, None, None, False       # already small: nothing hidden

    ref = _shrink(full_gray, ref_scale)
    ref_var = sharpness(ref)[1]
    face_full = _face_region(full_gray, face_boxes)
    face_ref = _face_region(ref, face_boxes)
    use_face = (face_ref is not None and min(face_ref.shape) >= 64)
    face_ref_var = float(_laplacian(face_ref).var()) if use_face else 0.0

    falloffs = {}
    for scale in (1.0, 0.5):
        octaves = math.log2(scale / ref_scale)
        img = _shrink(full_gray, scale)
        f = _falloff(ref_var, sharpness(img)[1], octaves)
        if use_face and f is not None:
            face = _shrink(face_full, scale)
            f = max(f, _falloff(face_ref_var, float(_laplacian(face).var()), octaves))
        falloffs[scale] = f

    full, half = falloffs[1.0], falloffs[0.5]
    if full is None or full < threshold:
        return 1.0, full, half, False
    if half is None or half < threshold:
        return 0.5, full, half, False
    return 0.25, full, half, (half is not None and half >= BLURRED_AT_HALF)


def exposure(gray):
    """Penalise clipped highlights/shadows and extreme average brightness.

    Party and candle-lit shots are meant to be dark, so this is a soft penalty:
    a dim but well-detailed frame still scores respectably.
    """
    total = gray.size
    blown = float((gray >= 250).sum()) / total
    crushed = float((gray <= 5).sum()) / total
    mean = float(gray.mean())

    # a little clipping is normal (specular highlights); 12%+ is a problem
    clip_pen = min(1.0, (max(0.0, blown - 0.02) / 0.12) ** 0.8) * 0.6
    clip_pen += min(1.0, (max(0.0, crushed - 0.05) / 0.25) ** 0.8) * 0.4

    # brightness comfort band 55..200, tapering rather than cliff-edged
    if mean < 55:
        bright_pen = min(1.0, (55 - mean) / 55.0) * 0.75
    elif mean > 200:
        bright_pen = min(1.0, (mean - 200) / 55.0) * 0.9
    else:
        bright_pen = 0.0

    score = 1.0 - min(1.0, 0.65 * clip_pen + 0.55 * bright_pen)
    return float(np.clip(score, 0.0, 1.0)), {"blown": blown, "crushed": crushed, "mean": mean}


def contrast(gray):
    """Robust dynamic range: the p2..p98 spread, ignoring outlier pixels."""
    lo, hi = np.percentile(gray, [2, 98])
    spread = float(hi - lo)
    return float(np.clip(spread / 180.0, 0.0, 1.0)), spread


def colorfulness(rgb):
    """Hasler & Suesstrunk colourfulness - dull grey frames read as flat."""
    a = rgb.astype(np.float32)
    rg = a[:, :, 0] - a[:, :, 1]
    yb = 0.5 * (a[:, :, 0] + a[:, :, 1]) - a[:, :, 2]
    std = math.sqrt(float(rg.std()) ** 2 + float(yb.std()) ** 2)
    mean = math.sqrt(float(rg.mean()) ** 2 + float(yb.mean()) ** 2)
    raw = std + 0.3 * mean
    return float(np.clip(raw / 70.0, 0.0, 1.0)), raw


def noise(gray):
    """Immerkaer noise sigma. High sigma on a soft image means grain, not detail."""
    k = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], dtype=np.float32)
    h, w = gray.shape
    if h < 5 or w < 5:
        return 1.0, 0.0
    if HAS_CV2:
        conv = cv2.filter2D(gray, cv2.CV_32F, k)
    else:
        conv = np.zeros_like(gray)
        for dy in range(3):
            for dx in range(3):
                conv[1:-1, 1:-1] += k[dy, dx] * gray[dy:h - 2 + dy, dx:w - 2 + dx]
    sigma = float(np.abs(conv[1:-1, 1:-1]).sum()) * math.sqrt(0.5 * math.pi) / (
        6.0 * (w - 2) * (h - 2))
    # 0..2 clean, 8+ grainy
    score = 1.0 - float(np.clip((sigma - 2.0) / 8.0, 0.0, 1.0))
    return score, sigma


def blockiness(gray):
    """Detect 8x8 JPEG compression blocking.

    Heavily re-compressed photos (WhatsApp forwards, screenshots) gain edge
    energy exactly on the 8-pixel grid. Laplacian variance mistakes that for
    detail, so it needs measuring separately.

    Must be given NATIVE-resolution pixels on an 8-aligned crop: resampling
    smears the block grid away and the measurement reads clean every time.
    """
    h, w = gray.shape
    if h < 24 or w < 24:
        return 1.0, 0.0
    dh = np.abs(np.diff(gray, axis=1))          # horizontal neighbour differences
    dv = np.abs(np.diff(gray, axis=0))
    # columns/rows that sit on the 8x8 block boundary vs. everything else
    cols = np.arange(dh.shape[1])
    rows = np.arange(dv.shape[0])
    bnd_h = dh[:, (cols % 8) == 7]
    int_h = dh[:, (cols % 8) != 7]
    bnd_v = dv[(rows % 8) == 7, :]
    int_v = dv[(rows % 8) != 7, :]
    interior = float(int_h.mean() + int_v.mean()) / 2.0
    boundary = float(bnd_h.mean() + bnd_v.mean()) / 2.0
    if interior < 1e-4:
        return 1.0, 0.0
    ratio = boundary / interior      # ~1.0 pristine, 1.9 ~= q30, 2.6 ~= q12
    score = 1.0 - float(np.clip((ratio - 1.50) / 1.20, 0.0, 1.0))
    return score, ratio


def resolution(photo, target_mp=6.0):
    """Print needs pixels. 6 MP fills an A4 page at 300 DPI; below that we taper."""
    mp = photo.megapixels
    return float(np.clip(mp / target_mp, 0.0, 1.0)), mp


# Face crops are measured at this size so a big face and a small one are judged
# on the same footing - raw Laplacian variance is mostly a function of how many
# pixels the face occupies, which tells you nothing about focus.
FACE_PATCH = 140
FACE_DETAIL_EDGE = 2048     # resolution the crops are taken from
MIN_FACE_PX = 40



def face_focus(photo, boxes):
    """Are all the people in this photo in focus, or only the nearest one?

    Returns (consistency, values) where consistency is the middle face's
    sharpness as a fraction of the sharpest face's.

    This is what catches the group selfie where the person holding the phone is
    pin sharp and everyone behind them is a smear - a photo that looks fine on
    screen and disappointing at print size. On real photographs the separation
    is stark: a selfie of six with four blurred faces scores 0.06, while a
    family of four all in focus scores 0.71.
    """
    if not boxes or len(boxes) < 2:
        return 1.0, []
    try:
        img = ingest.load(photo, max_long_edge=FACE_DETAIL_EDGE)
    except Exception:
        return 1.0, []

    W, H = img.size
    grey = img.convert("L")
    measured = []
    for (fx, fy, fw, fh, _score) in boxes:
        x0, y0 = int(max(0, fx * W)), int(max(0, fy * H))
        x1, y1 = int(min(W, (fx + fw) * W)), int(min(H, (fy + fh) * H))
        if x1 - x0 < MIN_FACE_PX or y1 - y0 < MIN_FACE_PX:
            continue                      # too few pixels to judge focus
        patch = grey.crop((x0, y0, x1, y1)).resize((FACE_PATCH, FACE_PATCH),
                                                   Image.BILINEAR)
        a = np.asarray(patch, dtype=np.float32)
        measured.append((float(_laplacian(a).var()), fw * fh))

    values = [v for v, _area in measured]
    if len(measured) < 2:
        return 1.0, values

    measured.sort(key=lambda t: -t[0])
    sharpest = measured[0][0]
    if sharpest <= 1e-6:
        return 1.0, values

    middle = float(np.median(values))
    return float(np.clip(middle / sharpest, 0.0, 1.0)), values


def face_sharpness(gray, boxes):
    """Laplacian variance measured inside the face boxes.

    Tile-percentile sharpness asks "is any part of this frame sharp?", which a
    photo of a motion-blurred person against a crisp background answers yes to.
    When there are people in the shot, their faces are the thing that has to be
    sharp, so those regions get measured on their own.
    """
    if not boxes:
        return None
    H, W = gray.shape
    best = 0.0
    for (fx, fy, fw, fh, _score) in boxes:
        x0, y0 = int(max(0, fx * W)), int(max(0, fy * H))
        x1, y1 = int(min(W, (fx + fw) * W)), int(min(H, (fy + fh) * H))
        if x1 - x0 < 12 or y1 - y0 < 12:
            continue                      # too small to judge reliably
        region = gray[y0:y1, x0:x1]
        best = max(best, float(_laplacian(region).var()))
    if best <= 0.0:
        return None
    lo, hi = math.log10(6.0), math.log10(260.0)
    norm = (math.log10(max(best, 1e-3)) - lo) / (hi - lo)
    return float(np.clip(norm, 0.0, 1.0)), best


def interest(gray):
    """Shannon entropy of the luminance histogram.

    A frame with a full range of tones has something going on in it; a flat
    wall, a lens cap or a washed-out sky does not.
    """
    hist, _ = np.histogram(gray, bins=256, range=(0, 255))
    p = hist.astype(np.float64)
    total = p.sum()
    if total <= 0:
        return 0.0, 0.0
    p /= total
    nz = p[p > 0]
    h = float(-(nz * np.log2(nz)).sum())      # 0..8 bits
    return float(np.clip((h - 4.0) / 3.0, 0.0, 1.0)), h


def detail_spread(peak_var, isolation_ratio):
    """How much real detail there is across the WHOLE frame.

    Subject isolation is the wrong question to ask of a landscape. A jungle, a
    hillside or a sky has no subject popping out of a background - detail is
    spread evenly, which scores near zero on isolation and drags a perfectly
    good photograph down. What actually distinguishes a good wide shot from an
    empty one is whether there is detail everywhere, so this measures the
    typical tile rather than the sharpest.
    """
    median_var = peak_var / max(isolation_ratio, 1e-6)
    lo, hi = math.log10(3.0), math.log10(160.0)
    v = (math.log10(max(median_var, 1e-3)) - lo) / (hi - lo)
    return float(np.clip(v, 0.0, 1.0))


def isolation_score(ratio):
    """How strongly the sharpest region stands out from the rest of the frame.

    Portrait lenses and close subjects give a big ratio; a flat snapshot of a
    room gives ~1. This is the closest classical proxy for "there is a subject".
    """
    lo, hi = math.log10(1.3), math.log10(8.0)
    v = (math.log10(max(ratio, 1e-3)) - lo) / (hi - lo)
    return float(np.clip(v, 0.0, 1.0))


# A frame with one clear subject was almost always aimed and taken on purpose;
# a frame with a crowd of small faces is usually a grab shot. Portraits get a
# lift so the deliberate pictures rise above the incidental ones.
PORTRAIT_BONUS = 0.20
PORTRAIT_MIN_AREA = 0.012      # the subject has to actually fill some frame

# The other kind of deliberate photograph: everyone lined up together. Faces in
# a group shot are individually small, which the size term punishes hard - yet
# the posed group is very often the single most wanted picture of the whole
# trip. It gets its own bonus rather than competing on face size.
GROUP_BONUS = 0.30
GROUP_MIN_FACES = 4

# Above this, the photo reads as everybody posed together - and then everybody
# is expected to be in focus. Below it, soft faces behind a sharp subject are
# depth, not a fault. Calibrated on real examples; see the module notes.
GROUP_FOCUS_THRESHOLD = 0.55


def group_score(n_faces, eye_ratio, total_area):
    """How strongly this reads as a posed group photograph, 0..1."""
    if n_faces < GROUP_MIN_FACES:
        return 0.0
    # More people is a stronger signal, levelling off around a dozen.
    size = float(np.clip((n_faces - GROUP_MIN_FACES + 1) / 8.0, 0.0, 1.0))
    # People looking at the camera is what separates "posed" from "crowd".
    looking = 1.0 if eye_ratio < 0 else float(np.clip(eye_ratio, 0.0, 1.0))
    # And they should occupy a real part of the frame, not be distant specks.
    presence = float(np.clip(total_area / 0.045, 0.0, 1.0))
    return float(np.clip(0.42 * size + 0.34 * looking + 0.24 * presence, 0.0, 1.0))


def face_score(n_faces, largest_area, total_area):
    """Rank on people: how many, how big in frame, and how deliberate.

    Largest-face area dominates - somebody close to the camera is the shot,
    where six tiny faces across a hall usually is not.
    """
    if not n_faces:
        return 0.0
    count_term = min(1.0, math.sqrt(min(n_faces, 6)) / math.sqrt(3.0))
    # 8% of the frame is already a solid portrait
    size_term = float(np.clip(math.sqrt(max(largest_area, 0.0) / 0.08), 0.0, 1.0))
    spread_term = float(np.clip(total_area / 0.20, 0.0, 1.0))
    base = 0.34 * count_term + 0.50 * size_term + 0.16 * spread_term

    # One or two people, filling a decent part of the frame: a portrait, not a
    # crowd scene. Scaled by subject size so a distant lone figure gets little.
    if n_faces <= 2 and largest_area >= PORTRAIT_MIN_AREA:
        weight = float(np.clip(largest_area / 0.05, 0.0, 1.0))
        base += PORTRAIT_BONUS * weight * (1.0 if n_faces == 1 else 0.6)

    return float(np.clip(base, 0.0, 1.0))


def combined_face_score(n_faces, largest_area, total_area, eye_ratio):
    """Face score with the group-photo case handled on its own terms."""
    base = face_score(n_faces, largest_area, total_area)
    grp = group_score(n_faces, eye_ratio, total_area)
    return float(np.clip(base + GROUP_BONUS * grp, 0.0, 1.0)), grp


def analyse(photo, weights=None, detect_faces=True, content_weights=None,
            content_mix=CONTENT_MIX, print_sharpness="normal"):
    """Score one photo in place.

    Produces two independent numbers:
      photo.technical - is the frame usable? (sharp, exposed, clean)
      photo.content   - is it worth printing? (people, subject, life)

    The final score is led by content. Technical only drags it down while the
    frame is below the comfort gate, so a slightly soft photo of somebody
    laughing beats a pin-sharp photo of an empty chair.
    """
    tw = weights or TECHNICAL_WEIGHTS
    cw = content_weights or CONTENT_WEIGHTS
    try:
        # Decoded once at full size: the print-size sharpness check needs
        # the real pixels, and everything else runs on the same reduced copy
        # ingest.load(max_long_edge=...) would have produced.
        full = ingest.load(photo)
        img = full
        if max(full.size) > ANALYSIS_EDGE:
            k = ANALYSIS_EDGE / float(max(full.size))
            img = full.resize((max(1, int(full.width * k)), max(1, int(full.height * k))),
                              Image.BILINEAR)
    except FileNotFoundError:
        photo.status = "unreadable"
        photo.reasons.append("file disappeared while the album was being built")
        photo.score = photo.technical = photo.content = 0.0
        return 0.0
    except Exception as exc:
        photo.status = "unreadable"
        photo.reasons.append("decode failed (%s)" % type(exc).__name__)
        photo.score = photo.technical = photo.content = 0.0
        return 0.0

    rgb = np.asarray(img, dtype=np.uint8)
    gray = np.asarray(img.convert("L"), dtype=np.float32)

    s_sharp, raw_sharp, iso_ratio = sharpness(gray)
    s_expo, expo_raw = exposure(gray)
    s_contrast, raw_contrast = contrast(gray)
    s_color, raw_color = colorfulness(rgb)
    s_noise, raw_noise = noise(gray)
    s_block, raw_block = blockiness(ingest.native_crop_gray(photo))
    s_interest, raw_entropy = interest(gray)
    s_iso = isolation_score(iso_ratio)
    s_detail = detail_spread(raw_sharp, iso_ratio)
    _, raw_mp = resolution(photo)

    if detect_faces:
        boxes = facedet.detect(rgb)
        finfo = facedet.summarise(boxes, rgb.shape[1], rgb.shape[0])
        eyes_open, eyes_judged = facedet.eye_state(rgb, boxes)
    else:
        finfo = {"count": 0, "total_area": 0.0, "largest_area": 0.0,
                 "centre": None, "boxes": []}
        eyes_open, eyes_judged = 0, 0
    n_faces = finfo["count"]
    face_total = finfo["total_area"]
    face_largest = finfo["largest_area"]
    face_centre = finfo["centre"]
    photo.face_boxes = finfo["boxes"]
    # Eye ratio is needed before scoring, because a posed group is defined by
    # people looking at the camera as much as by how many of them there are.
    eye_ratio = (eyes_open / float(eyes_judged)) if eyes_judged else -1.0
    s_faces, s_group = combined_face_score(n_faces, face_largest, face_total,
                                           eye_ratio)

    # How many of the people are actually in focus, not just the nearest one.
    focus_ok, focus_values = face_focus(photo, photo.face_boxes) if n_faces >= 2 \
        else (1.0, [])
    # Only a POSED GROUP is expected to have everyone sharp. A portrait with a
    # softly blurred background is deliberate photography, not a fault, and
    # penalising it threw away good pictures of one person standing in front of
    # other people. The group score is what separates the two cases.
    if s_group > GROUP_FOCUS_THRESHOLD and n_faces >= 3 and focus_ok < 0.30:
        s_faces *= 0.30 + 0.70 * (focus_ok / 0.30)

    # Blinks. A frame where people's eyes are shut is the classic reject in any
    # set of group shots, and there is usually a near-identical frame a second
    # later where they are open - so this feeds ranking AND lets burst
    # de-duplication keep the right one of the pair.
    if eyes_judged:
        # Softened for groups: in a photo of ten people somebody always blinks,
        # and that is not a reason to lose the only picture of everyone.
        floor = 0.50 + 0.34 * s_group
        s_faces *= floor + (1.0 - floor) * eye_ratio

    # When people are present, judge sharpness mostly on their faces.
    fs = face_sharpness(gray, photo.face_boxes) if n_faces else None
    if fs is not None:
        s_face_sharp, raw_face_sharp = fs
        s_sharp = 0.38 * s_sharp + 0.62 * s_face_sharp
    else:
        s_face_sharp, raw_face_sharp = -1.0, -1.0

    technical = (tw["sharpness"] * s_sharp +
                 tw["exposure"] * s_expo +
                 tw["contrast"] * s_contrast +
                 tw["noise"] * s_noise +
                 tw["compression"] * s_block)

    content = (cw["faces"] * s_faces +
               cw["isolation"] * s_iso +
               cw["colorfulness"] * s_color +
               cw["interest"] * s_interest)

    # With no face detected at all, content rests on the weaker signals. They
    # are renormalised so a landmark, a view or a plate of food is judged on
    # its own merits - a trip is not only the people on it, and capping these
    # too hard left an album of 41 photographs without a single one of Thailand.
    if n_faces == 0:
        # Judged as a picture of a place or a thing: detail across the frame,
        # colour, and tonal range. Isolation is deliberately not used here -
        # see detail_spread for why it is the wrong question for scenery.
        denom = cw["isolation"] + cw["colorfulness"] + cw["interest"]
        content = (cw["isolation"] * max(s_detail, s_iso) +
                   cw["colorfulness"] * s_color +
                   cw["interest"] * s_interest) / max(denom, 1e-6) * NO_FACE_CAP

    gate = min(1.0, technical / TECHNICAL_GATE)
    blended = content_mix * content + (1.0 - content_mix) * technical
    total = float(np.clip(blended, 0.0, 1.0)) * gate * 100.0

    photo.technical = float(technical)
    photo.content = float(content)
    photo.scores = {
        "sharpness": s_sharp, "exposure": s_expo, "contrast": s_contrast,
        "colorfulness": s_color, "noise": s_noise, "compression": s_block,
        "interest": s_interest, "isolation": s_iso, "detail": s_detail,
        "faces_score": s_faces,
        "technical": float(technical), "content": float(content), "gate": gate,
        "raw_laplacian_var": raw_sharp, "raw_contrast_spread": raw_contrast,
        "raw_colorfulness": raw_color, "raw_noise_sigma": raw_noise,
        "raw_megapixels": raw_mp, "raw_block_ratio": raw_block,
        "raw_entropy": raw_entropy, "raw_isolation_ratio": iso_ratio,
        "face_sharpness": s_face_sharp, "raw_face_laplacian": raw_face_sharp,
        "group_score": s_group, "face_focus": focus_ok,
        "eyes_open": float(eyes_open), "eyes_judged": float(eyes_judged),
        "eye_ratio": eye_ratio,
        "faces": float(n_faces), "face_area": face_total,
        "face_largest": face_largest,
        "face_cx": face_centre[0] if face_centre else -1.0,
        "face_cy": face_centre[1] if face_centre else -1.0,
        "blown_pct": expo_raw["blown"] * 100.0,
        "crushed_pct": expo_raw["crushed"] * 100.0,
        "mean_brightness": expo_raw["mean"],
    }
    photo.score = total

    threshold = PRINT_SHARPNESS.get(print_sharpness)
    if threshold is not None:
        full_gray = np.asarray(full.convert("L"), dtype=np.float32)
        scale, f_full, f_half, blurred = sharp_scale(full_gray, photo.face_boxes,
                                                     threshold)
    else:
        scale, f_full, f_half, blurred = 1.0, None, None, False
    del full
    photo.scores.update(colour.measure(rgb, photo.face_boxes))
    photo.scores.update({
        "sharp_scale": scale,
        "effective_mp": raw_mp * scale * scale,
        "falloff_full": -1.0 if f_full is None else f_full,
        "falloff_half": -1.0 if f_half is None else f_half,
        "blurred_at_print": 1.0 if blurred else 0.0,
    })

    # Hard rejects - nothing downstream can rescue these.
    if raw_mp < MIN_MEGAPIXELS:
        photo.status = "rejected"
        photo.reasons.append("thumbnail-sized (%.2f MP)" % raw_mp)
    elif technical < TECHNICAL_REJECT:
        photo.status = "rejected"
        photo.reasons.append("technically unusable (quality %.0f%%)" % (technical * 100))

    return total


def describe(photo):
    """Short human phrases explaining the score, for the review report."""
    s = photo.scores
    out = []
    if not s:
        return out
    if s.get("face_sharpness", -1) >= 0 and s["face_sharpness"] < 0.34:
        out.append("faces are soft (face detail %.0f)" % s["raw_face_laplacian"])
    elif s["sharpness"] < 0.30:
        out.append("soft/blurred (laplacian %.0f)" % s["raw_laplacian_var"])
    elif s.get("sharp_scale", 1.0) < 1.0:
        # Width it can print at 170 DPI, the album's default minimum.
        cm = photo.width * s["sharp_scale"] / 170.0 * 2.54
        out.append("soft at full size - sharp up to %.0f cm wide" % cm)
    elif s["sharpness"] > 0.75:
        out.append("sharp")
    if s["blown_pct"] > 8:
        out.append("blown highlights %.0f%%" % s["blown_pct"])
    if s["crushed_pct"] > 15:
        out.append("crushed shadows %.0f%%" % s["crushed_pct"])
    if s["mean_brightness"] < 55:
        out.append("underexposed")
    elif s["mean_brightness"] > 200:
        out.append("overexposed")
    if s["contrast"] < 0.35:
        out.append("flat contrast")
    if s["raw_noise_sigma"] > 6:
        out.append("noisy (sigma %.1f)" % s["raw_noise_sigma"])
    if s.get("eyes_judged", 0) and s.get("eye_ratio", 1) < 0.5:
        out.append("eyes closed (%d of %d open)" % (
            int(s["eyes_open"]), int(s["eyes_judged"])))
    if s["faces"]:
        out.append("%d face%s%s" % (
            int(s["faces"]), "" if s["faces"] == 1 else "s",
            " (close-up)" if s.get("face_largest", 0) > 0.06 else ""))
    if s.get("compression", 1.0) < 0.55:
        out.append("jpeg blocking (ratio %.2f)" % s.get("raw_block_ratio", 0))
    if (s.get("face_focus", 1.0) < 0.30 and s.get("faces", 0) >= 3
            and s.get("group_score", 0) > GROUP_FOCUS_THRESHOLD):
        out.append("group photo but only the nearest person is in focus")
    if s.get("group_score", 0) > 0.45:
        out.append("group photo - %d people" % int(s.get("faces", 0)))
    elif s.get("faces", 0) == 1 and s.get("face_largest", 0) >= PORTRAIT_MIN_AREA:
        out.append("portrait - single subject")
    if s.get("isolation", 0) > 0.6:
        out.append("subject stands out")
    if s.get("faces", 0) == 0 and s.get("detail", 0) > 0.6:
        out.append("detailed scene")
    if s.get("interest", 1) < 0.35:
        out.append("flat/empty frame")
    if s["raw_megapixels"] < 2.0:
        out.append("small file %.1f MP (limits print size)" % s["raw_megapixels"])
    return out
