"""Fit a song to the video: stop it at a chosen point, and if the video runs
longer, fill the rest by repeating the song's last section before that point.

The repeat is made the way an editor loops music by hand:
  * the stop point is moved to the nearest downbeat at or before the time
    asked for, so it falls on the first beat of a bar;
  * the repeated section is a whole number of bars ending exactly there, so
    every repeat lands on the beat it left;
  * each join is a one-beat equal-power crossfade in which the song's own
    continuation fades out while the repeat fades in - the ear hears the
    phrase come round again, not a cut.
The result is written once as a WAV in the event's cache and then treated as
the music for the edit, so the cuts follow its beats across the joins too.
"""
import hashlib
import os

from ..deps import np
from . import audio, ffm

SR = audio.SR
LOOP_BEATS = (32, 16, 64)      # preferred repeat lengths, in beats


def _downbeats(analysis):
    beats = analysis["beats"]
    return [i for i in range(analysis.get("downbeat", 0), len(beats), 4)]


def extend(path, analysis, until, total, cache_dir):
    """Return (wav path, info) for `path` cut at `until` seconds and repeated
    to at least `total` seconds (plus a little for the fade)."""
    key = "%s|%d|%d|%.2f|%.2f|v1" % (os.path.abspath(path), os.path.getsize(path),
                                     int(os.path.getmtime(path)), until, total)
    out = os.path.join(cache_dir, "music",
                       "fit_%s.wav" % hashlib.sha1(key.encode()).hexdigest()[:16])
    beats = analysis["beats"]
    period = (beats[-1] - beats[0]) / max(1, len(beats) - 1)
    downs = _downbeats(analysis)
    # The stop point: the last downbeat at or before `until`.
    cut_i = max((i for i in downs if beats[i] <= until + 0.05), default=downs[-1])
    cut = beats[cut_i]
    # The repeated section: whole bars ending at the stop point, as long as
    # the song before it allows, preferring 32 beats.
    loop_beats = next((n for n in LOOP_BEATS if cut_i - n >= downs[0]), 4)
    loop_start = beats[cut_i - loop_beats]
    info = {"cut": round(cut, 3), "loop_start": round(loop_start, 3),
            "loop_beats": loop_beats, "crossfade": round(period, 3)}
    if os.path.exists(out):
        return out, info

    xf = int(round(period * SR))                      # one beat
    need = int((total + 3.0) * SR)
    src = audio._read_music(path, 0.0, cut + period * 2 + 1.0)
    a0, c0 = int(round(loop_start * SR)), int(round(cut * SR))
    body = src[:, :c0]
    pieces = [body]
    length = body.shape[1]
    fade_in = np.sin(np.linspace(0, np.pi / 2, xf)) ** 1
    fade_out = np.cos(np.linspace(0, np.pi / 2, xf))
    tail = src[:, c0:c0 + xf]                          # what the song would play next
    while length < need:
        loop = src[:, a0:c0].copy()
        n = min(xf, loop.shape[1], tail.shape[1])
        # Equal-power: the song's own continuation out, the repeat in.
        loop[:, :n] = loop[:, :n] * fade_in[:n] + tail[:, :n] * fade_out[:n]
        pieces.append(loop)
        length += loop.shape[1]
    out_pcm = np.concatenate(pieces, axis=1)[:, :need]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".part.wav"
    ffm.run([ffm.FFMPEG, "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2",
             "-i", "-", "-c:a", "pcm_s16le", tmp],
            input_bytes=np.ascontiguousarray(out_pcm.T).astype(np.float32).tobytes())
    os.replace(tmp, out)
    return out, info
