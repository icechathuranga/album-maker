"""Fit a song to the video's length, the way a music editor extends a track.

A song shorter than the video is not looped in a small piece. It plays from
its start; at a bar line in its second half it jumps back to an earlier bar
in the middle of the song and carries on from there to its own ending, so
the extra time is a long, natural stretch of the song and the film still
ends on the song's real ending. If one jump is not enough, the same middle
stretch comes round again.

The jump is placed where it will be heard least:
  * both ends are downbeats (the first beat of a bar), and the stretch
    between them is a whole number of bars, so the beat never stumbles;
  * of all such pairs, the one whose music sounds most alike just before and
    just after the two points wins - the same instruments, the same energy -
    measured on a band spectrum of every beat;
  * the join is a one-beat equal-power crossfade.

`until` cuts the song earlier than its natural end (the same jump then fills
any time after it). The result is written once as a WAV in the event's cache
and used as the music for the edit, so the cuts follow its beats throughout.
"""
import hashlib
import math
import os

from ..deps import np
from . import audio, ffm

SR = audio.SR
FEAT_RATE = 22050
BANDS = 24
LOOK = 8               # beats compared either side of a jump


def _features(path, beats):
    """A normalised band spectrum for every beat of the song."""
    pcm = ffm.decode_audio(path, FEAT_RATE)
    edges = np.geomspace(60, 9000, BANDS + 1)
    feats = []
    for i in range(len(beats)):
        a = int(beats[i] * FEAT_RATE)
        b = int(beats[i + 1] * FEAT_RATE) if i + 1 < len(beats) else a + 4096
        seg = pcm[a:b]
        if len(seg) < 256:
            feats.append(np.zeros(BANDS + 1))
            continue
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        f = np.fft.rfftfreq(len(seg), 1.0 / FEAT_RATE)
        bands = np.array([spec[(f >= lo) & (f < hi)].mean() if np.any((f >= lo) & (f < hi))
                          else 0.0 for lo, hi in zip(edges[:-1], edges[1:])])
        bands = np.log1p(bands)
        loud = float(np.sqrt(np.mean(seg ** 2)))
        feats.append(np.concatenate([bands / (np.linalg.norm(bands) + 1e-9), [loud]]))
    return np.array(feats)


def _alike(feats, i, j):
    """How alike the music is around beats i and j (higher is better)."""
    n = len(feats)
    if i - LOOK < 0 or j - LOOK < 0 or i + LOOK >= n or j + LOOK >= n:
        return -1.0
    a = np.concatenate([feats[i - LOOK:i, :-1].ravel(), feats[i:i + LOOK, :-1].ravel()])
    b = np.concatenate([feats[j - LOOK:j, :-1].ravel(), feats[j:j + LOOK, :-1].ravel()])
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    la, lb = feats[i - 2:i + 2, -1].mean(), feats[j - 2:j + 2, -1].mean()
    level = abs(math.log((la + 1e-6) / (lb + 1e-6)))
    return cos - 0.3 * level


def plan(path, analysis, end, total):
    """Choose the jump. Returns dict(cut, jump_to, repeats, score) in seconds,
    or None when the song (up to `end`) already covers `total`."""
    beats = analysis["beats"]
    period = (beats[-1] - beats[0]) / max(1, len(beats) - 1)
    end_i = max(i for i in range(len(beats)) if beats[i] <= end + 0.05)
    extra = total + 2.5 - beats[end_i]          # a little over, for the fade
    if extra <= 0:
        return None
    down = analysis.get("downbeat", 0)
    bars = list(range(down, end_i, 4))
    # One jump back covers at most half the song; more extra time repeats it.
    span_bars = len(bars)
    need_bars = int(math.ceil(extra / period / 4.0))
    repeats = max(1, int(math.ceil(need_bars / max(1.0, span_bars * 0.5))))
    jump_bars = int(math.ceil(need_bars / float(repeats)))
    feats = _features(path, beats)
    best = None
    for c in bars:
        j = c - 4 * jump_bars
        # The cut late in the song but before its ending; the return point
        # well clear of the intro.
        if j < bars[0] + 8 or c > end_i - 16 or c < 0.4 * end_i:
            continue
        sc = _alike(feats, c, j)
        if best is None or sc > best[0]:
            best = (sc, c, j)
    if best is None:                              # very short song: any bars
        c = bars[-1] if bars else end_i
        j = max(down, c - 4 * jump_bars)
        best = (0.0, c, j)
    sc, c, j = best
    return {"cut": round(beats[c], 3), "jump_to": round(beats[j], 3), "repeats": repeats,
            "score": round(sc, 3), "end": round(beats[end_i], 3),
            "crossfade": round(period, 3)}


def extend(path, analysis, until, total, cache_dir):
    """Return (wav path, info) for the song fitted to `total` seconds, ending
    at `until` (its natural end when None)."""
    end = until if until else analysis["duration"]
    info = plan(path, analysis, end, total)
    key = "%s|%d|%d|%.2f|%.2f|v2" % (os.path.abspath(path), os.path.getsize(path),
                                     int(os.path.getmtime(path)), end, total)
    out = os.path.join(cache_dir, "music",
                       "fit_%s.wav" % hashlib.sha1(key.encode()).hexdigest()[:16])
    if os.path.exists(out):
        return out, info
    src = audio._read_music(path, 0.0, end + 2.0)
    e0 = min(src.shape[1], int(round(end * SR)))
    if info is None:
        pieces = [src[:, :e0]]
    else:
        xf = int(round(info["crossfade"] * SR))
        c0, j0 = int(round(info["cut"] * SR)), int(round(info["jump_to"] * SR))
        fade_in = np.sin(np.linspace(0, np.pi / 2, xf))
        fade_out = np.cos(np.linspace(0, np.pi / 2, xf))
        tail = src[:, c0:c0 + xf]                    # what the song would play next

        def joined(seg):
            seg = seg.copy()
            n = min(xf, seg.shape[1], tail.shape[1])
            seg[:, :n] = seg[:, :n] * fade_in[:n] + tail[:, :n] * fade_out[:n]
            return seg

        pieces = [src[:, :c0]]
        for _ in range(info["repeats"] - 1):
            pieces.append(joined(src[:, j0:c0]))     # the middle comes round again
        pieces.append(joined(src[:, j0:e0]))         # ...then on to the end
    out_pcm = np.concatenate(pieces, axis=1)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".part.wav"
    ffm.run([ffm.FFMPEG, "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2",
             "-i", "-", "-c:a", "pcm_s16le", tmp],
            input_bytes=np.ascontiguousarray(out_pcm.T).astype(np.float32).tobytes())
    os.replace(tmp, out)
    return out, info
