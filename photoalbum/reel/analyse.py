"""Watch each clip once and write down, four times a second, what is in it.

Every sample is scored with the album's own measures - faces, sharpness,
exposure, colour - at the album's analysis size, so a moment in a video and
a photo are judged on one scale. On top come the things only video has:
motion, camera shake, cuts inside the clip, and what the sound is doing.

The result is cached next to the reels, keyed by the file's path, size and
modification time, so running make-reel again (or making variants) does not
watch the footage twice.
"""
import hashlib
import json
import os

from ..deps import np, cv2
from .. import colour, faces, quality
from . import ffm

VERSION = 1
RATE = 4.0              # samples per second
EDGE = 1024             # analysis long edge, as quality.ANALYSIS_EDGE
CUT_BITS = 22           # dHash distance between samples that means a cut
AUDIO_RATE = 16000


def _dhash(gray):
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _hamming(a, b):
    return bin(a ^ b).count("1")


def _cache_path(cache_dir, clip):
    key = "%s|%d|%d|%d" % (clip.rel, clip.bytes, int(clip.mtime), VERSION)
    return os.path.join(cache_dir, hashlib.sha1(key.encode()).hexdigest() + ".json")


def _audio_track(clip, n):
    """Loudness, onsets and a speech-band share, one value per sample."""
    out = {"rms": [-90.0] * n, "onset": [0.0] * n, "speech": [0.0] * n}
    if not clip.has_audio:
        return out
    pcm = ffm.decode_audio(clip.path, AUDIO_RATE)
    hop = int(AUDIO_RATE / RATE)
    if len(pcm) < hop:
        return out
    freqs = np.fft.rfftfreq(hop, 1.0 / AUDIO_RATE)
    voice = (freqs >= 300) & (freqs <= 3400)
    rms, speech = [], []
    for i in range(n):
        seg = pcm[i * hop:(i + 1) * hop]
        if len(seg) < hop // 2:
            rms.append(-90.0)
            speech.append(0.0)
            continue
        rms.append(float(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-6)))
        spec = np.abs(np.fft.rfft(seg, n=hop)) ** 2
        speech.append(float(spec[voice].sum() / (spec.sum() + 1e-9)))
    onset = []
    for i, r in enumerate(rms):
        before = rms[max(0, i - 2):i] or [r]
        onset.append(1.0 if r - float(np.mean(before)) > 6.0 and r > -40 else 0.0)
    return {"rms": rms, "onset": onset, "speech": speech}


CHUNK = 30.0            # seconds of footage per parallel job
FIELDS = ("t", "sharp", "expo", "colour", "faces", "face_score", "largest",
          "boxes", "motion", "shift", "cut", "bright", "hash")


def _scan_span(path, t0, dur, size, hdr):
    """Score the samples in [t0, t0 + dur). Runs in a worker process.

    Decoding starts one sample early so the first sample has a neighbour to
    measure motion and cuts against; that extra sample is not returned.
    """
    lead = 1.0 / RATE if t0 > 0 else 0.0
    tl = {k: [] for k in FIELDS}
    prev_small, prev_hash = None, None
    best = (-1.0, None, None)
    for j, rgb in enumerate(ffm.decode_frames(path, t0 - lead, dur + lead, RATE, size,
                                              hdr=hdr)):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        small = cv2.resize(gray, (160, max(2, int(160 * size[1] / size[0]))),
                           interpolation=cv2.INTER_AREA)
        h = _dhash(gray)
        if lead and j == 0:
            prev_small, prev_hash = small, h
            continue
        sharp = quality.sharpness(gray)[0]
        expo, raw = quality.exposure(gray)
        col = quality.colorfulness(rgb)[0]
        boxes = faces.detect(rgb)
        info = faces.summarise(boxes, size[0], size[1])
        fscore = quality.combined_face_score(info["count"], info["largest_area"],
                                             info["total_area"], -1.0)[0]
        motion, shift = 0.0, 0.0
        if prev_small is not None:
            motion = float(np.mean(np.abs(small - prev_small)) / 255.0)
            (dx, dy), _ = cv2.phaseCorrelate(prev_small, small)
            shift = float(np.hypot(dx, dy) / small.shape[1])
        cut = 1.0 if prev_hash is not None and _hamming(h, prev_hash) > CUT_BITS else 0.0
        prev_small, prev_hash = small, h
        k = len(tl["t"])
        tl["t"].append(round(t0 + k / RATE, 3))
        tl["sharp"].append(round(sharp, 3))
        tl["expo"].append(round(expo, 3))
        tl["colour"].append(round(col, 3))
        tl["faces"].append(info["count"])
        tl["face_score"].append(round(fscore, 3))
        tl["largest"].append(round(info["largest_area"], 4))
        tl["boxes"].append([[round(v, 4) for v in b] for b in info["boxes"]])
        tl["motion"].append(round(motion, 4))
        tl["shift"].append(round(shift, 4))
        tl["cut"].append(cut)
        tl["bright"].append(round(raw["mean"], 1))
        tl["hash"].append(h)
        value = 0.6 * fscore + 0.4 * sharp
        if value > best[0]:
            best = (value, rgb.copy(), info["boxes"])
        if len(tl["t"]) >= int(round(dur * RATE)):
            break
    return tl, best


def _finish(clip, parts, cache_path):
    """Join a clip's chunks, add shake and sound, set scores, write the cache."""
    tl = {k: [] for k in FIELDS}
    best = (-1.0, None, None)
    for part, b in parts:
        for k in FIELDS:
            tl[k].extend(part[k])
        if b[0] > best[0]:
            best = b
    n = len(tl["t"])
    if n == 0:
        clip.status = "unreadable"
        clip.reasons.append("no frames could be decoded")
        clip.timeline, clip.scores = None, {}
        return clip
    # Camera shake: how far each frame's shift strays from its neighbours'.
    shifts = np.array(tl["shift"])
    med = np.array([np.median(shifts[max(0, i - 2):i + 3]) for i in range(n)])
    tl["jitter"] = [round(float(v), 4) for v in np.abs(shifts - med)]
    tl.update(_audio_track(clip, n))
    # Clip-level numbers in the album's vocabulary, for looks and colour.
    scores = {
        "mean_brightness": float(np.median(tl["bright"])),
        "faces": float(max(tl["faces"])),
        "faces_score": float(max(tl["face_score"])),
        "colorfulness": float(np.median(tl["colour"])),
        "sharpness": float(np.median(tl["sharp"])),
        "raw_noise_sigma": 0.0,
    }
    if best[1] is not None:
        scores.update(colour.measure(best[1], best[2]))
    clip.timeline, clip.scores = tl, scores
    clip.dhash = tl["hash"][int(np.argmax(tl["face_score"]))]
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump({"timeline": tl, "scores": scores, "dhash": clip.dhash}, fh)
    return clip


def _load_cached(clip, path):
    if path and os.path.exists(path):
        with open(path) as fh:
            data = json.load(fh)
        clip.timeline, clip.scores = data["timeline"], data["scores"]
        clip.dhash = data.get("dhash")
        return True
    return False


def analyse_clips(clips, cache_dir=None, workers=None, progress=None):
    """Analyse many clips at once, in 30-second chunks across worker processes.

    Processes rather than threads: the face detector is one shared OpenCV
    object per process and is not safe to call from two threads at once.
    Each ffmpeg decoder uses about two cores, so half the CPUs are enough.
    """
    from concurrent.futures import ProcessPoolExecutor

    todo = []
    for c in clips:
        path = _cache_path(cache_dir, c) if cache_dir else None
        if not _load_cached(c, path):
            todo.append((c, path))
    if not todo:
        return
    jobs = []
    for ci, (c, _p) in enumerate(todo):
        size = ffm.fit_size(c.width, c.height, EDGE)
        t = 0.0
        while t < c.duration - 0.05:
            jobs.append((ci, t, min(CHUNK, c.duration - t), size, c.hdr, c.path))
            t += CHUNK
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_scan_span, path, t0, dur, size, hdr): (ci, t0)
                   for ci, t0, dur, size, hdr, path in jobs}
        from concurrent.futures import as_completed
        for done, fut in enumerate(as_completed(futures), 1):
            results[futures[fut]] = fut.result()
            if progress:
                progress(done, len(jobs), todo[futures[fut][0]][0].name)
    for ci, (c, path) in enumerate(todo):
        parts = [results[k] for k in sorted(k for k in results if k[0] == ci)]
        _finish(c, parts, path)


def analyse_clip(clip, cache_dir=None):
    """One clip: (clip, was_cached)."""
    path = _cache_path(cache_dir, clip) if cache_dir else None
    if _load_cached(clip, path):
        return clip, True
    analyse_clips([clip], cache_dir)
    return clip, False


SHAKE_EDGE = 320        # small copy for measuring shake at full frame rate


def measure_shake(clip, t0, dur):
    """Hand-shake over [t0, t0 + dur]: how far each frame jumps from the
    smooth path of its neighbours, as a share of the frame width.

    Measured at 30 frames a second, because shake lives at 3-10 Hz - far
    above what four samples a second can see. A deliberate pan moves every
    frame by the same amount and so does not count; a shaking hand does.
    """
    size = ffm.fit_size(clip.width, clip.height, SHAKE_EDGE)
    prev, shifts = None, []
    for rgb in ffm.decode_frames(clip.path, t0, dur, 30, size, hdr=clip.hdr):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if prev is not None:
            (dx, dy), _ = cv2.phaseCorrelate(prev, gray)
            shifts.append((dx, dy))
        prev = gray
    if len(shifts) < 6:
        return 0.0
    s = np.array(shifts)
    smooth = np.array([np.median(s[max(0, i - 3):i + 4], axis=0) for i in range(len(s))])
    jitter = np.hypot(*(s - smooth).T) / float(size[0])
    return round(float(np.percentile(jitter, 75)), 5)
