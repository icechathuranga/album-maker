"""Find the moments worth showing: short spans of clips, and photos.

A clip's timeline is scored sample by sample - people, sharpness, exposure,
movement without shake, and sound (a cheer, a laugh) - then the peaks are
picked out. Each moment keeps the whole span around its peak that stays
good, so the editor can trim it to the length the cut needs without
straying into a blurred pan or across a cut.

A long recording is treated the way an editor treats it: as a beginning,
a middle and an end. A six-minute jeep recovery is three beats on screen -
the first good wide shot, the moment of most action, the loudest moment
(the cheer as it climbs out) - not twelve near-identical holds. The longer
the recording the more it may give, up to five.
"""
import datetime

from ..deps import np
from . import analyse

MIN_SPAN = 0.75          # seconds; anything shorter is not a moment
PEAK_GAP = 2.0           # seconds between two moments of a short clip
LONG_GAP = 20.0          # ...and of a long one
SPAN_REACH = 5.0         # a moment reaches at most this far either side of its peak
SKIP_HEAD = 0.3          # the first instant of a clip is usually a fumble
SHAKE_FULL = 0.02        # jitter (share of width) that counts as full shake
OWN_TWIN_BITS = 14       # two moments of one clip must differ at least this much
CLOSE_UP = 0.06          # largest face area above this is a close-up
WIDE = 0.02              # ...below this (or no face) is a wide shot


class Moment:
    __slots__ = ("kind", "src", "centre", "lo", "hi", "score", "people", "motion",
                 "loud", "onset", "speech", "sharp", "segment", "hash", "taken",
                 "faces", "largest", "tag", "shake", "action", "low_res", "colour",
                 "bright")

    def __init__(self, kind, src):
        self.kind, self.src = kind, src
        self.centre = self.lo = self.hi = 0.0
        self.score = self.people = self.motion = self.loud = 0.0
        self.onset = self.speech = 0.0
        self.sharp = 1.0
        self.segment = getattr(src, "segment", 0)
        self.hash = getattr(src, "dhash", None)
        self.taken = getattr(src, "taken", None)
        self.faces = 0
        self.largest = 0.0
        self.tag = ""
        self.shake = None            # measured at full frame rate, later
        self.action = None           # time of the strongest action, for the cut
        self.low_res = False
        self.colour = 0.5
        self.bright = 128.0          # mean brightness 0-255

    @property
    def span(self):
        return self.hi - self.lo if self.kind == "clip" else 99.0

    @property
    def wide(self):
        return self.faces == 0 or self.largest < WIDE

    @property
    def close_up(self):
        return self.largest > CLOSE_UP

    def __repr__(self):
        return "<Moment %s %s @%.1f [%.1f-%.1f] %.2f %s>" % (
            self.kind, self.src.name, self.centre, self.lo, self.hi, self.score, self.tag)


def goodness(tl):
    """Per-sample score 0-1, and a mask of samples that must not be used."""
    sharp = np.array(tl["sharp"])
    expo = np.array(tl["expo"])
    people = np.array(tl["face_score"])
    motion = np.clip(np.array(tl["motion"]) / 0.12, 0, 1)
    shake = np.clip(np.array(tl["jitter"]) / SHAKE_FULL, 0, 1)
    loud = np.clip((np.array(tl["rms"]) + 50.0) / 40.0, 0, 1)
    onset = np.array(tl["onset"])
    colour = np.array(tl["colour"])
    if max(tl["faces"]) > 0:
        g = (0.45 * people + 0.15 * sharp + 0.10 * expo
             + 0.10 * motion * (1 - shake) + 0.10 * loud + 0.10 * onset)
    else:
        # Nobody in the clip: judged as scenery, capped as the album caps it.
        g = 0.84 * (0.35 * colour + 0.25 * sharp + 0.15 * expo
                    + 0.15 * motion * (1 - shake) + 0.10 * loud)
    t = np.array(tl["t"])
    bad = (sharp < 0.25) | (expo < 0.30) | (np.array(tl["jitter"]) > 0.03) | (t < SKIP_HEAD)
    return g, bad, people, motion, loud


def _cap(duration):
    """How many moments a clip of this length may give. Long recordings are
    the richest material, so they give the most: three, plus one for every
    45 seconds, up to twelve (a 6-minute clip gives eleven) - and fewer for
    short clips. Moments stay at least 20 s apart and visually distinct."""
    return int(min(12, 3 + duration // 45, 1 + duration // 20))


def _candidates(clip, g, bad, t, gap):
    """Every local peak worth considering, best first, spaced by `gap`."""
    smooth = np.convolve(np.where(bad, 0.0, g), np.ones(3) / 3.0, mode="same")
    cuts = set(i for i, c in enumerate(clip.timeline["cut"]) if c)
    n = len(t)
    step = 1.0 / analyse.RATE
    out, taken = [], []
    for i in np.argsort(-smooth, kind="stable"):
        if smooth[i] < 0.2 or len(out) >= 24:
            break
        if bad[i] or any(abs(t[i] - c) < gap for c in taken):
            continue
        floor = 0.5 * smooth[i]
        lo = hi = i
        while lo - 1 >= 0 and not bad[lo - 1] and smooth[lo - 1] >= floor and lo not in cuts:
            lo -= 1
        while hi + 1 < n and not bad[hi + 1] and smooth[hi + 1] >= floor and (hi + 1) not in cuts:
            hi += 1
        # A moment is its own few seconds, not the whole good stretch round
        # it: otherwise several moments of a long steady recording share one
        # span and end up cut from the same footage.
        while t[lo] < t[i] - SPAN_REACH and lo < i:
            lo += 1
        while t[hi] > t[i] + SPAN_REACH and hi > i:
            hi -= 1
        lo_t, hi_t = t[lo], min(clip.duration - 0.1, t[hi] + step)
        if hi_t - lo_t < MIN_SPAN:
            continue
        out.append((int(i), int(lo), int(hi), float(lo_t), float(hi_t)))
        taken.append(t[i])
    return out


def _make(clip, tl, g, people, motion, loud, i, lo, hi, lo_t, hi_t):
    t = np.array(tl["t"])
    m = Moment("clip", clip)
    m.centre, m.lo, m.hi, m.tag = float(t[i]), lo_t, hi_t, "best"
    core = (t >= t[i] - 0.5) & (t <= t[i] + 0.5)
    speech = np.array(tl["speech"])
    onset = np.array(tl["onset"])
    rms = np.array(tl["rms"])
    m.score = float(g[core].mean())
    m.people = float(people[core].mean())
    m.motion = float(motion[core].mean())
    m.loud = float(loud[core].mean() + 0.5 * onset[core].max())
    m.onset = float(onset[core].max())
    m.speech = float(speech[core].mean())
    m.sharp = float(np.array(tl["sharp"])[core].mean())
    m.colour = float(np.array(tl["colour"])[core].mean())
    # The darker of the peak second and the whole span: a moment of light in
    # a night drive still plays mostly in the dark.
    span_b = np.array(tl["bright"])[lo:hi + 1]
    m.bright = float(min(np.array(tl["bright"])[core].mean(), np.median(span_b)))
    m.faces = int(max(tl["faces"][lo:hi + 1]))
    m.largest = float(max(tl["largest"][lo:hi + 1]))
    m.hash = tl["hash"][i]
    # Where the action is: the strongest sound onset in the span, or failing
    # that the peak of movement. The editor cuts just before it.
    span = range(lo, hi + 1)
    hits = [k for k in span if onset[k]]
    if hits:
        k = max(hits, key=lambda k: rms[k] - (rms[k - 1] if k else rms[k]))
    else:
        k = max(span, key=lambda k: motion[k])
    m.action = float(t[k])
    # A static talking head - someone speaking to the camera, nothing moving
    # - is the dullest shot in a travel film, unless there is a laugh in it.
    if m.speech > 0.5 and m.motion < 0.1 and m.largest > CLOSE_UP and not m.onset:
        m.score *= 0.6
    # 1080p landscape footage is upscaled 1.8x to fill a vertical frame and
    # looks visibly softer beside 4K or portrait shots.
    m.low_res = clip.width > clip.height and clip.height < 1500
    if m.low_res:
        m.score *= 0.85
    if clip.taken:
        m.taken = clip.taken + datetime.timedelta(seconds=m.centre)
    return m


def _apart(a, b, gap):
    return abs(a.centre - b.centre) >= gap and analyse._hamming(a.hash, b.hash) > OWN_TWIN_BITS


def clip_moments(clip):
    tl = clip.timeline
    if not tl or not tl["t"]:
        return []
    g, bad, people, motion, loud = goodness(tl)
    t = np.array(tl["t"])
    gap = PEAK_GAP if clip.duration < 60 else LONG_GAP
    cands = _candidates(clip, g, bad, t, gap)
    if not cands:
        return []
    made = [_make(clip, tl, g, people, motion, loud, *c) for c in cands]
    cap = max(1, _cap(clip.duration))

    # Beginning, middle and end: the first good wide shot, the most action,
    # the loudest moment (a cheer rather than talk) - then the best of the rest.
    picks = []

    def add(m, tag):
        if m is not None and m not in picks and all(_apart(m, p, gap) for p in picks):
            m.tag = tag
            picks.append(m)

    if cap >= 2:
        wides = [m for m in made if m.wide and m.score >= 0.3]
        add(min(wides, key=lambda m: m.centre) if wides else None, "establish")
        add(max(made, key=lambda m: m.motion), "action")
        cheers = [m for m in made if m.onset and m.speech < 0.4] or \
                 [m for m in made if m.onset]
        add(max(cheers, key=lambda m: m.loud) if cheers else None, "sound")
    for m in sorted(made, key=lambda m: -m.score):
        if len(picks) >= cap:
            break
        add(m, "best")
    return sorted(picks[:cap], key=lambda m: m.centre)


def photo_moment(photo):
    s = photo.scores or {}
    m = Moment("photo", photo)
    # A reel is a colourful medium: a grey still is a weaker shot here than
    # in a printed album.
    m.score = photo.score / 100.0 + 0.15 * (s.get("colorfulness", 0.5) - 0.5)
    m.people = s.get("faces_score", 0.0)
    m.faces = int(s.get("faces", 0))
    m.largest = s.get("face_largest", 0.0)
    m.colour = s.get("colorfulness", 0.5)
    m.sharp = s.get("sharpness", 1.0)
    m.bright = s.get("mean_brightness", 128.0)
    m.shake = 0.0
    return m


def collect(clips, photos):
    moments = []
    for c in clips:
        moments.extend(clip_moments(c))
    moments.extend(photo_moment(p) for p in photos)
    return moments


def measure_shake(moments, cache, workers=6):
    """Hand-shake at the full frame rate, for every clip moment.

    The analysis samples four times a second, too slowly to see shake at
    3-10 Hz. So each moment's middle two seconds are decoded at 30 fps on a
    small copy and the frame-to-frame jitter measured. Cached by clip and
    time, so it is paid once. Returns True if anything new was measured.
    """
    from concurrent.futures import ThreadPoolExecutor

    todo = []
    for m in moments:
        if m.kind != "clip":
            continue
        key = "%s|%d|%.2f" % (m.src.rel, int(m.src.mtime), m.centre)
        if key in cache:
            m.shake = cache[key]
        else:
            todo.append((m, key))

    def one(item):
        m, key = item
        return key, analyse.measure_shake(m.src, max(m.lo, m.centre - 1.0),
                                          min(2.0, m.span))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, value in pool.map(one, todo):
            cache[key] = value
    for m, key in todo:
        m.shake = cache[key]
    return bool(todo)
