"""Thin wrappers round ffmpeg and ffprobe.

Every video frame the reel code looks at comes through `decode_frames`: one
decode path, so rotation, HDR and variable frame rate are handled once.
OpenCV's VideoCapture is deliberately not used - it ignores the rotation
phones record and has no HDR handling.
"""
import json
import shutil
import subprocess

from ..deps import np

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

HDR_TRANSFERS = ("smpte2084", "arib-std-b67")

# Tone-map HDR (PQ or HLG) phone footage to ordinary SDR, so it is not
# printed grey and washed out. zscale needs float frames in between.
HDR_TO_SDR = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
              "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")


class FfmpegError(RuntimeError):
    pass


def available():
    return bool(FFMPEG and FFPROBE)


def run(cmd, input_bytes=None):
    """Run a command; raise FfmpegError with the tail of stderr on failure."""
    proc = subprocess.run(cmd, input=input_bytes, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-6:]
        raise FfmpegError("%s failed: %s" % (cmd[0], " | ".join(tail)))
    return proc


def probe_json(path):
    out = run([FFPROBE, "-v", "error", "-print_format", "json",
               "-show_format", "-show_streams", path]).stdout
    return json.loads(out.decode("utf-8", "replace") or "{}")


def even(n):
    return max(2, int(round(n / 2.0)) * 2)


def fit_size(w, h, long_edge):
    """(w, h) scaled so the long edge is at most long_edge, both even."""
    k = min(1.0, long_edge / float(max(w, h)))
    return even(w * k), even(h * k)


def decode_frames(path, t0, duration, fps, size, hdr=False, extra_vf=None,
                  keyframes=False):
    """Yield RGB uint8 frames of `size` (w, h) from t0 for `duration` seconds.

    ffmpeg applies the phone's rotation itself (autorotate is on by default),
    so `size` must already be the upright display size. `keyframes` decodes
    only key frames - many times faster on 4K HEVC, for when the nearest
    keyframe is close enough (choosing a crop, say).
    """
    w, h = size
    vf = []
    if hdr:
        vf.append(HDR_TO_SDR)
    vf.append("fps=%g" % fps)
    if extra_vf:
        vf.append(extra_vf)
    vf.append("scale=%d:%d:flags=area" % (w, h))
    cmd = [FFMPEG, "-v", "error"] + (["-skip_frame", "nokey"] if keyframes else []) + [
           "-ss", "%.3f" % max(0.0, t0), "-i", path,
           "-t", "%.3f" % duration, "-vf", ",".join(vf), "-an",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def decode_audio(path, rate=16000):
    """The whole soundtrack as mono float32 at `rate`, or an empty array."""
    try:
        out = run([FFMPEG, "-v", "error", "-i", path, "-vn", "-ac", "1",
                   "-ar", str(rate), "-f", "f32le", "-"]).stdout
    except FfmpegError:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(out, dtype=np.float32).copy()
