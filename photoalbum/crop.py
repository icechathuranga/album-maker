"""Content-aware cropping - deciding which part of a frame goes on the page.

A slot on a page has a fixed shape, and most photos do not match it. Cutting
from the centre is the lazy answer: it slices through people standing off to
one side and throws away the part of a scene that made it worth shooting.

Instead this builds a small importance map of the frame - detail, colour,
contrast, and faces weighted far above everything else - then slides every
candidate window of the required shape across it and keeps the best. Scoring
rewards the things that make a print look composed:

  * keeping the important content, and keeping whole faces in particular
  * placing that content near a rule-of-thirds line rather than dead centre
  * not cutting a subject off at the frame edge
  * staying as wide as possible, since throwing pixels away is a cost

It is entirely deterministic - the same photo and slot always give the same
crop - and runs on a 220-pixel thumbnail, so it costs a few milliseconds.
"""
from .deps import np, cv2, HAS_CV2, Image

MAP_EDGE = 220           # importance map resolution
SCALE_STEPS = (1.00, 0.94, 0.88, 0.82, 0.76, 0.70, 0.64)
GRID_STEPS = 12          # candidate positions per axis

FACE_WEIGHT = 6.0        # a face is worth many times a patch of texture
FACE_PAD = 0.28          # grow face boxes by this much - hair and chin matter
CUT_FACE_PENALTY = 3.2   # slicing through a head is the worst outcome
THIRDS_BONUS = 0.16
EDGE_PENALTY = 0.22
ZOOM_PENALTY = 0.30      # discourage cropping tighter than necessary


def importance_map(rgb, face_boxes=None):
    """A small float map of where the interesting content sits.

    Detail, colour and local contrast each contribute; faces are stamped on
    top at a weight nothing else can reach.
    """
    h, w = rgb.shape[:2]
    scale = MAP_EDGE / float(max(h, w))
    mw, mh = max(8, int(round(w * scale))), max(8, int(round(h * scale)))

    if HAS_CV2:
        small = cv2.resize(rgb, (mw, mh), interpolation=cv2.INTER_AREA)
    else:
        small = np.asarray(Image.fromarray(rgb).resize((mw, mh), Image.BILINEAR))

    a = small.astype(np.float32) / 255.0
    lum = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]

    # detail: gradient magnitude
    gx = np.zeros_like(lum)
    gy = np.zeros_like(lum)
    gx[:, 1:-1] = lum[:, 2:] - lum[:, :-2]
    gy[1:-1, :] = lum[2:, :] - lum[:-2, :]
    detail = np.sqrt(gx * gx + gy * gy)

    # colour: distance from grey, so a vivid subject beats a grey wall
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    saturation = mx - mn

    # local contrast over a small window
    if HAS_CV2:
        mean = cv2.blur(lum, (9, 9))
        sq = cv2.blur(lum * lum, (9, 9))
        local = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    else:
        local = np.abs(lum - lum.mean())

    imp = (0.52 * _norm(detail) + 0.26 * _norm(saturation) + 0.22 * _norm(local))

    # Skin-ish hues carry people even when no face box was returned.
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    skin = ((r > 0.30) & (r > g * 1.05) & (r > b * 1.10) &
            (saturation > 0.10) & (saturation < 0.62))
    imp += skin.astype(np.float32) * 0.10

    faces = np.zeros((mh, mw), dtype=np.float32)
    for (fx, fy, fw, fh, _score) in (face_boxes or []):
        # normalised coords -> map pixels, padded to include hair and chin
        px = (fx - fw * FACE_PAD) * mw
        py = (fy - fh * FACE_PAD * 1.4) * mh
        pw = fw * (1 + 2 * FACE_PAD) * mw
        ph = fh * (1 + 2.4 * FACE_PAD) * mh
        x0, y0 = int(max(0, px)), int(max(0, py))
        x1, y1 = int(min(mw, px + pw)), int(min(mh, py + ph))
        if x1 > x0 and y1 > y0:
            faces[y0:y1, x0:x1] = 1.0
            imp[y0:y1, x0:x1] += FACE_WEIGHT

    return imp, faces


def _norm(a):
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-6:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _integral(a):
    return np.pad(a, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)


def _window_sum(ii, x0, y0, x1, y1):
    return float(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])


def _thirds_score(imp_win):
    """How much of the window's importance sits near a thirds intersection."""
    h, w = imp_win.shape
    if h < 6 or w < 6:
        return 0.0
    total = float(imp_win.sum())
    if total <= 1e-6:
        return 0.0
    ys = np.arange(h, dtype=np.float32) / max(1, h - 1)
    xs = np.arange(w, dtype=np.float32) / max(1, w - 1)
    # proximity to 1/3 or 2/3 along each axis
    py = np.minimum(np.abs(ys - 1 / 3.0), np.abs(ys - 2 / 3.0))
    px = np.minimum(np.abs(xs - 1 / 3.0), np.abs(xs - 2 / 3.0))
    wy = np.clip(1.0 - py / 0.22, 0.0, 1.0)
    wx = np.clip(1.0 - px / 0.22, 0.0, 1.0)
    weight = np.outer(wy, wx)
    return float((imp_win * weight).sum() / total)


def find(rgb, target_aspect, face_boxes=None, max_zoom_out=0.64):
    """Best crop of `target_aspect` (w/h). Returns (x, y, w, h) in rgb pixels."""
    H, W = rgb.shape[:2]
    if target_aspect <= 0 or H < 8 or W < 8:
        return (0, 0, W, H)

    imp, facemap = importance_map(rgb, face_boxes)
    mh, mw = imp.shape
    ii = _integral(imp)

    # Face boxes in map pixels, weighted by size RELATIVE to the largest face in
    # the frame. That is what separates "one of the people in this photo" from
    # "a stranger forty metres away": someone standing with the group is a
    # comparable size to everyone else in it, however small they all are, and
    # must not be cropped out to improve the composition.
    faces_px = []
    boxes = list(face_boxes or [])
    if boxes:
        biggest = max(fw * fh for (_fx, _fy, fw, fh, _s) in boxes) or 1e-6
        for (fx, fy, fw, fh, _score) in boxes:
            bx0, by0 = fx * mw, fy * mh
            bx1, by1 = (fx + fw) * mw, (fy + fh) * mh
            weight = float(np.clip((fw * fh) / biggest, 0.0, 1.0)) ** 0.5
            faces_px.append((bx0, by0, bx1, by1, weight))

    src_aspect = mw / float(mh)

    # Largest window of the wanted shape that fits the frame.
    if target_aspect >= src_aspect:
        base_w, base_h = mw, mw / target_aspect
    else:
        base_h, base_w = mh, mh * target_aspect

    best = None
    for s in SCALE_STEPS:
        if s < max_zoom_out:
            break
        cw, ch = base_w * s, base_h * s
        iw, ih = int(round(cw)), int(round(ch))
        if iw < 4 or ih < 4 or iw > mw or ih > mh:
            continue

        xs = _positions(mw - iw, GRID_STEPS)
        ys = _positions(mh - ih, GRID_STEPS)

        for y0 in ys:
            for x0 in xs:
                x1, y1 = x0 + iw, y0 + ih
                content = _window_sum(ii, x0, y0, x1, y1)

                score = content / (iw * ih) ** 0.5   # density, not raw sum

                # Face handling, judged one face at a time.
                #
                # Measuring the total fraction of face pixels kept cannot tell
                # "dropped a bystander entirely" from "sliced someone in half",
                # and those are very different mistakes. A face left out of
                # frame is an editing decision; a face cut down the middle is
                # the error people notice immediately in a printed album.
                score -= _face_penalty(faces_px, x0, y0, x1, y1) * CUT_FACE_PENALTY

                win = imp[y0:y1, x0:x1]
                score += _thirds_score(win) * THIRDS_BONUS * score

                # Important content pressed against the border reads as a
                # mistake, so nudge away from it.
                border = (float(win[0, :].mean()) + float(win[-1, :].mean()) +
                          float(win[:, 0].mean()) + float(win[:, -1].mean())) / 4.0
                score -= border * EDGE_PENALTY

                # All else equal, keep more of the photograph.
                score -= (1.0 - s) * ZOOM_PENALTY * abs(score)

                if best is None or score > best[0]:
                    best = (score, x0, y0, iw, ih)

    if best is None:
        return _centre_box(W, H, target_aspect)

    _, x0, y0, iw, ih = best
    fx = W / float(mw)
    fy = H / float(mh)
    x = int(round(x0 * fx))
    y = int(round(y0 * fy))
    w = int(round(iw * fx))
    h = int(round(ih * fy))

    # Clamp back inside the frame and re-true the aspect after rounding.
    w = max(8, min(w, W))
    h = max(8, min(h, H))
    if w / float(h) > target_aspect:
        w = int(round(h * target_aspect))
    else:
        h = int(round(w / target_aspect))
    x = max(0, min(x, W - w))
    y = max(0, min(y, H - h))
    return (x, y, w, h)


SLICE_OK = 0.90      # a face this fully inside the window counts as kept
SLICE_GONE = 0.04    # below this it is simply not in the picture

# Leaving somebody out of a group photograph is nearly as bad as slicing them.
# It was previously treated as a mild cost, and a crop duly dropped the child on
# the left of a family of four - she ended 0.002 of the frame outside the window.
EXCLUDE_PENALTY = 1.55


def _face_penalty(faces_px, x0, y0, x1, y1):
    """How badly this window treats the faces in the frame. 0 is clean.

    Three outcomes per face:
      kept      - almost entirely inside the window: no penalty
      excluded  - almost entirely outside: a small penalty, and only for faces
                  big enough that leaving them out changes the picture
      sliced    - partly in, partly out: heavily penalised, worst around half,
                  because a head cut down the middle is the one crop error
                  that is obvious in print
    """
    total = 0.0
    for (fx0, fy0, fx1, fy1, weight) in faces_px:
        area = max(1e-6, (fx1 - fx0) * (fy1 - fy0))
        ix0, iy0 = max(fx0, x0), max(fy0, y0)
        ix1, iy1 = min(fx1, x1), min(fy1, y1)
        inside = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        frac = inside / area

        if frac >= SLICE_OK:
            continue
        if frac <= SLICE_GONE:
            total += EXCLUDE_PENALTY * weight       # left out of the picture
            continue
        # Peaks at frac = 0.5 and falls away towards either clean outcome.
        severity = 1.0 - abs(frac - 0.5) * 2.0
        total += (0.45 + 0.55 * severity) * weight
    return total


def _positions(span, steps):
    if span <= 0:
        return [0]
    if span < steps:
        return list(range(span + 1))
    return [int(round(i * span / float(steps))) for i in range(steps + 1)]


def _centre_box(W, H, target_aspect):
    if W / float(H) > target_aspect:
        w = int(round(H * target_aspect))
        return ((W - w) // 2, 0, w, H)
    h = int(round(W / target_aspect))
    return (0, (H - h) // 2, W, h)


def apply(img, box):
    """Crop a PIL image to a (x, y, w, h) box."""
    x, y, w, h = box
    return img.crop((x, y, x + w, y + h))
