"""The soundtrack, mixed the way an editor mixes phone audio.

Phone sound swings wildly - a quiet temple at -41 LUFS, an engine at -11 -
so every clip is levelled on its own before anything is joined, with the
rumble below 90 Hz taken out. Then, on one timeline:

  * clips overlap by 0.3 s with equal-power fades instead of butting
    together, so there are no clicks and no holes;
  * the next shot's sound can start early (a J-cut) when it has a voice or
    a cheer worth leading with;
  * under a photo, the previous clip's sound carries on quietly (an L-cut),
    so a still never cuts to dead air;
  * music is ducked gently (3:1 above -24 dB, 50 ms attack, 800 ms release)
    under the moments whose own sound is kept, and fades over its last
    two seconds, which the editor has placed on a phrase boundary.
Loudness is then set to -14 LUFS in two passes (measured, then applied as
one linear gain), so nothing pumps.
"""
import json
import os
import re

from ..deps import np
from . import ffm

SR = 48000
TARGET_RMS_DB = -20.0     # every clip levelled to this before mixing
MAX_GAIN_DB = 12.0
CROSSFADE = 0.3
AMBIENCE_DB = -10.0       # the previous clip's sound under a photo
LOUDNESS = "I=-14:TP=-1.5:LRA=11"


def _read(path, t0, dur):
    """Stereo float32 (2, n) of a file's sound from t0, high-passed."""
    if dur <= 0:
        return np.zeros((2, 0), dtype=np.float32)
    try:
        raw = ffm.run([ffm.FFMPEG, "-v", "error", "-ss", "%.3f" % max(0.0, t0),
                       "-i", path, "-t", "%.3f" % dur, "-vn",
                       "-af", "highpass=f=90", "-ac", "2", "-ar", str(SR),
                       "-f", "f32le", "-"]).stdout
    except ffm.FfmpegError:
        return np.zeros((2, 0), dtype=np.float32)
    a = np.frombuffer(raw, dtype=np.float32)
    return a[: len(a) // 2 * 2].reshape(-1, 2).T.copy()


def _level(a):
    """Gain that brings this piece to the target level (ignoring silence)."""
    if a.shape[1] < SR // 10:
        return a
    mono = np.abs(a).mean(axis=0)
    win = SR // 20
    n = len(mono) // win
    if n == 0:
        return a
    blocks = np.sqrt((mono[: n * win].reshape(n, win) ** 2).mean(axis=1))
    loud = blocks[blocks > np.percentile(blocks, 30)]
    rms = float(np.sqrt((loud ** 2).mean())) if len(loud) else 0.0
    if rms < 1e-5:
        return a
    gain_db = np.clip(TARGET_RMS_DB - 20 * np.log10(rms), -MAX_GAIN_DB, MAX_GAIN_DB)
    return a * (10 ** (gain_db / 20.0))


def _fade(a, fade_in, fade_out):
    """Equal-power fades, in place."""
    n = a.shape[1]
    fi, fo = min(n, int(fade_in * SR)), min(n, int(fade_out * SR))
    if fi > 1:
        a[:, :fi] *= np.sin(np.linspace(0, np.pi / 2, fi))
    if fo > 1:
        a[:, n - fo:] *= np.cos(np.linspace(0, np.pi / 2, fo))
    return a


def _place(track, a, start):
    i0 = int(round(start * SR))
    if i0 < 0:
        a, i0 = a[:, -i0:], 0
    i1 = min(track.shape[1], i0 + a.shape[1])
    if i1 > i0:
        track[:, i0:i1] += a[:, : i1 - i0]


def _duck(music, voice):
    """Lower the music where the kept natural sound is loud: 3:1 above -24 dB."""
    hop = SR // 100                                     # 10 ms
    n = voice.shape[1] // hop
    if n == 0:
        return music
    env = np.sqrt((np.abs(voice[:, : n * hop]).mean(axis=0).reshape(n, hop) ** 2).mean(axis=1))
    env_db = 20 * np.log10(env + 1e-6)
    want = np.clip((env_db + 24.0) * (1 - 1 / 3.0), 0.0, 12.0)     # dB of reduction
    red = np.zeros(n)
    att, rel = np.exp(-1 / 5.0), np.exp(-1 / 80.0)                  # 50 ms, 800 ms
    for i in range(1, n):
        c = att if want[i] > red[i - 1] else rel
        red[i] = c * red[i - 1] + (1 - c) * want[i]
    gain = 10 ** (-np.repeat(red, hop) / 20.0)
    m = music.copy()
    k = min(m.shape[1], len(gain))
    m[:, :k] *= gain[:k]
    return m


def mix(recipe, sources, music_path, out_wav):
    """Write the finished (not yet loudness-normalised) soundtrack."""
    segs = recipe["segments"]
    starts, t = [], 0.0
    for s in segs:
        starts.append(t)
        t += max(1, int(round(s["dur"] * 30))) / 30.0
    total = t
    nat = np.zeros((2, int(total * SR) + SR), dtype=np.float32)

    for s, t0 in zip(segs, starts):
        snd = s.get("sound")
        if s["kind"] == "clip" and snd in ("own", "up"):
            clip = sources[s["src"]]
            if not clip.has_audio:
                continue
            lead = float(s.get("jcut", 0.0))
            a0 = max(0.0, s["in"] - lead)
            lead = s["in"] - a0
            a = _read(clip.path, a0, (s["out"] - a0) + CROSSFADE)
            a = _fade(_level(a), max(lead, 0.12), CROSSFADE)
            _place(nat, a, t0 - lead)
        elif s["kind"] == "photo" and s.get("ambience"):
            amb = s["ambience"]
            clip = sources.get(amb["src"])
            if clip is None or not clip.has_audio:
                continue
            a = _read(clip.path, amb["from"], s["dur"] + CROSSFADE)
            a = _fade(_level(a) * 10 ** (AMBIENCE_DB / 20.0), 0.15, CROSSFADE)
            _place(nat, a, t0)

    out = nat
    if recipe["audio"]["mode"] == "music" and music_path:
        m = _read_music(music_path, recipe["audio"]["offset"], total)
        m = _fade(_level(m), 0.05, recipe["audio"].get("fade", 2.0))
        music = np.zeros_like(nat)
        _place(music, m, 0.0)
        out = nat + _duck(music, nat)
    out = out[:, : int(total * SR)]
    _fade(out, 0.0, 1.0)                     # with the picture's fade out
    peak = float(np.abs(out).max()) if out.size else 0.0
    if peak > 0.98:
        out *= 0.98 / peak
    ffm.run([ffm.FFMPEG, "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2",
             "-i", "-", "-c:a", "pcm_s16le", out_wav],
            input_bytes=np.ascontiguousarray(out.T).astype(np.float32).tobytes())
    return out_wav


def _read_music(path, offset, total):
    raw = ffm.run([ffm.FFMPEG, "-v", "error", "-ss", "%.3f" % offset, "-i", path,
                   "-t", "%.3f" % (total + 0.2), "-vn", "-ac", "2", "-ar", str(SR),
                   "-f", "f32le", "-"]).stdout
    a = np.frombuffer(raw, dtype=np.float32)
    return a[: len(a) // 2 * 2].reshape(-1, 2).T.copy()


def loudness_filter(wav):
    """Measure the mix, then return a one-gain loudnorm filter for it."""
    proc = ffm.run([ffm.FFMPEG, "-hide_banner", "-i", wav, "-af",
                    "loudnorm=%s:print_format=json" % LOUDNESS, "-f", "null", "-"])
    text = proc.stderr.decode("utf-8", "replace")
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", text, re.S)
    if not m:
        return "loudnorm=%s" % LOUDNESS
    d = json.loads(m.group(0))
    return ("loudnorm=%s:measured_I=%s:measured_TP=%s:measured_LRA=%s:"
            "measured_thresh=%s:offset=%s:linear=true" % (
                LOUDNESS, d["input_i"], d["input_tp"], d["input_lra"],
                d["input_thresh"], d["target_offset"]))
