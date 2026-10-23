"""Find the beat of a music track, offline, with numpy alone.

The method is the textbook one and is deterministic:
  1. an onset envelope - how much new energy arrives in each short frame
     (positive spectral flux on a log-compressed spectrogram);
  2. tempo from the envelope's autocorrelation, between 60 and 200 BPM,
     gently preferring the tempos most pop music sits at;
  3. the beat phase that lines a comb of that tempo up with the onsets;
  4. downbeats - which of every four beats carries the most weight.
Cuts then land on these times.
"""
from ..deps import np
from . import ffm

RATE = 22050
FRAME = 2048
HOP = 512
BPM_RANGE = (60.0, 200.0)
PRIOR_BPM = 120.0


def _envelopes(pcm):
    """(full, bass) onset envelopes. Bass is where kicks - the beat - live."""
    n = 1 + max(0, (len(pcm) - FRAME) // HOP)
    if n < 8:
        z = np.zeros(0, dtype=np.float32)
        return z, z
    window = np.hanning(FRAME).astype(np.float32)
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    mag = np.log1p(100.0 * np.abs(np.fft.rfft(pcm[idx] * window, axis=1)))
    freqs = np.fft.rfftfreq(FRAME, 1.0 / RATE)

    def flux(m):
        f = np.concatenate([[0.0], np.maximum(0.0, np.diff(m, axis=0)).sum(axis=1)])
        # Subtract a moving average so slow swells do not count as onsets.
        smooth = np.convolve(f, np.ones(16) / 16, mode="same")
        e = np.maximum(0.0, f - smooth)
        return (e / (e.max() + 1e-9)).astype(np.float32)

    return flux(mag), flux(mag[:, freqs < 200])


def _prior(bpm):
    return float(np.exp(-0.5 * (np.log2(bpm / PRIOR_BPM) / 0.9) ** 2))


def _comb(env, period, phases=48):
    """Best (mean onset strength on the ticks, phase) for a beat period."""
    t = np.arange(len(env))
    best = (-1.0, 0.0)
    for phase in np.linspace(0, period, phases, endpoint=False):
        ticks = np.arange(phase, len(env) - 1, period)
        if len(ticks) < 4:
            continue
        m = float(np.interp(ticks, t, env).mean())
        if m > best[0]:
            best = (m, phase)
    return best


def _tempo(env, fps):
    """Tempo candidates from the autocorrelation, settled by the comb.

    The autocorrelation alone confuses a tempo with its 2:3 and 1:2 cousins
    (a hi-hat pattern repeats too); the right tempo is the one whose ticks
    actually land on onsets.
    """
    e = env - env.mean()
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    lo = int(fps * 60.0 / BPM_RANGE[1])
    hi = min(len(ac) - 2, int(fps * 60.0 / BPM_RANGE[0]))
    if hi <= lo:
        return PRIOR_BPM
    peaks = [l for l in range(lo + 1, hi) if ac[l] >= ac[l - 1] and ac[l] >= ac[l + 1]]
    peaks.sort(key=lambda l: -ac[l])
    candidates = set()
    for l in peaks[:5]:
        bpm = 60.0 * fps / l
        for k in (1.0, 1.5, 2.0 / 3.0, 2.0, 0.5):
            if BPM_RANGE[0] <= bpm * k <= BPM_RANGE[1]:
                candidates.add(round(bpm * k, 3))
    if not candidates:
        return PRIOR_BPM
    best = max(candidates, key=lambda b: _comb(env, fps * 60.0 / b, 24)[0] * _prior(b))
    # Refine with a fine sweep around the winner.
    fine = np.linspace(best * 0.98, best * 1.02, 41)
    return float(max(fine, key=lambda b: _comb(env, fps * 60.0 / b, 24)[0]))


def analyse(path):
    """{"bpm", "beats": [seconds], "downbeat": index of first downbeat,
    "duration", "energy": per-beat loudness 0-1} - or None if unreadable."""
    pcm = ffm.decode_audio(path, RATE)
    if len(pcm) < RATE * 4:
        return None
    env, bass = _envelopes(pcm)
    fps = RATE / float(HOP)
    # Onsets are a frame or two wide; a little blur lets a comb that is off
    # by half a frame still find them, so a coarse phase search is enough.
    k = np.array([0.25, 0.5, 1.0, 0.5, 0.25], dtype=np.float32)
    both = np.convolve(0.5 * env + 0.5 * bass, k / k.sum(), mode="same")
    bpm = _tempo(both, fps)
    period = fps * 60.0 / bpm
    t = np.arange(len(env))

    # Phase from the bass: hi-hats between the kicks are sharp onsets too,
    # and a cut on the off-beat feels wrong.
    _, best_phase = _comb(bass if bass.max() > 0 else env, period, 96)
    # A frame's flux registers an onset as soon as its window starts to
    # overlap it, a window length (less one hop) before it is heard.
    lag = (FRAME - HOP) / float(RATE)
    beats = np.arange(best_phase, len(env) - 1, period) / fps + lag

    # Downbeat: the one of four beat positions with the most onset weight.
    strength = np.interp(beats * fps, t, env)
    sums = [strength[i::4].sum() for i in range(4)]
    down = int(np.argmax(sums)) if len(beats) >= 8 else 0

    # Loudness per beat, for finding the quiet intro and the full section.
    rms = []
    for i in range(len(beats)):
        a = int(beats[i] * RATE)
        b = int(beats[i + 1] * RATE) if i + 1 < len(beats) else len(pcm)
        seg = pcm[a:b]
        rms.append(float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0)
    rms = np.array(rms)
    energy = rms / (rms.max() + 1e-9)
    return {"bpm": round(float(bpm), 2),
            "beats": [round(float(b), 4) for b in beats],
            "downbeat": down,
            "duration": round(len(pcm) / float(RATE), 3),
            "energy": [round(float(e), 3) for e in energy]}
