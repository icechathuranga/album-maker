"""Follow the people in a landscape clip with a vertical window.

What CapCut and Premiere call auto reframe: a 9:16 window that pans across
a wide frame to keep the subject in it. The camera-operator rules matter
more than the tracking itself - a window that twitches with every detection
looks worse than a fixed crop - so the path is:

  faces at 4 samples a second -> the subject's centre (largest faces count
  most; held through samples where nobody is found) -> median of 5 to drop
  stray detections -> smoothed forwards and backwards (no lag) -> a dead
  zone, so small drift is ignored -> a speed limit -> 30 frames a second.

If the window would barely move, the clip keeps one fixed crop instead.
"""
from ..deps import np
from . import analyse

ASPECT = 9.0 / 16.0
TAU = 0.35            # seconds of smoothing
DEAD_ZONE = 0.025     # share of frame width ignored as drift
MAX_SPEED = 0.25      # frame widths per second
STATIC_RANGE = 0.04   # a path moving less than this is held still
FPS = 30


def _smooth(x, dt, tau):
    a = dt / (tau + dt)
    out = np.array(x, dtype=float)
    for i in range(1, len(out)):
        out[i] = out[i - 1] + a * (out[i] - out[i - 1])
    for i in range(len(out) - 2, -1, -1):
        out[i] = out[i + 1] + a * (out[i] - out[i + 1])
    return out


def track(clip, t_in, t_out, window):
    """Window left edges (fraction of width), one per output frame, or None.

    `window` is the segment's static (x, y, w, h) window; the path moves it
    sideways and keeps its size. Portrait clips, where the window already
    spans the width, stay static.
    """
    win_w = window[2]
    if win_w >= 0.98 or not clip.timeline:
        return None
    tl = clip.timeline
    t = np.array(tl["t"])
    idx = [i for i in range(len(t)) if t_in - 0.5 <= t[i] <= t_out + 0.5]
    if len(idx) < 3:
        return None

    # Follow one person, as a camera operator would: a 9:16 window holds one
    # or two people, and aiming at the middle of a group frames the gap
    # between them. Start on the largest face, then stay with whoever is
    # nearest the last position among the faces that are still prominent.
    centres, last = [], None
    for i in idx:
        boxes = tl["boxes"][i]
        if boxes:
            biggest = max(b[2] * b[3] for b in boxes)
            near = [b for b in boxes if b[2] * b[3] >= 0.5 * biggest]
            if last is None:
                pick = max(near, key=lambda b: b[2] * b[3])
            else:
                pick = min(near, key=lambda b: abs(b[0] + b[2] / 2.0 - last))
            last = pick[0] + pick[2] / 2.0
        centres.append(last)
    first = next((c for c in centres if c is not None), None)
    if first is None:
        return None
    centres = [first if c is None else c for c in centres]
    c = np.array(centres)
    c = np.array([np.median(c[max(0, i - 2):i + 3]) for i in range(len(c))])
    dt = 1.0 / analyse.RATE
    c = _smooth(c, dt, TAU)

    # Dead zone: hold still until the subject drifts clear of it.
    held = [c[0]]
    for v in c[1:]:
        held.append(v if abs(v - held[-1]) > DEAD_ZONE else held[-1])
    c = _smooth(np.array(held), dt, TAU / 2.0)

    # Speed limit, then one value per output frame.
    for i in range(1, len(c)):
        step = MAX_SPEED * dt
        c[i] = min(max(c[i], c[i - 1] - step), c[i - 1] + step)
    ts = t[idx]
    n = max(1, int(round((t_out - t_in) * FPS)))
    frames = np.interp(t_in + np.arange(n) / float(FPS), ts, c)
    left = np.clip(frames - win_w / 2.0, 0.0, 1.0 - win_w)
    if left.max() - left.min() < STATIC_RANGE:
        return None
    return [round(float(v), 5) for v in left]
