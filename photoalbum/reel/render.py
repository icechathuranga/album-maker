"""Turn a recipe into an MP4.

Each shot is rendered on its own to a short constant-frame-rate video file
at the output size; the final pass joins them, lays the title and captions
on, and adds the soundtrack mixed by audio.py. Shot files are cached by
their content, so variants that share a moment share its render.

ffmpeg does the decoding (rotation, HDR), stabilising, cropping, colour and
encoding. Python draws the frames only where motion must be sub-pixel
smooth - a window following a person, a drift across scenery, a slow move
on a photo - because ffmpeg's crop and zoompan step visibly.
"""
import hashlib
import json
import os
import subprocess

from ..deps import np, cv2
from .. import enhance, ingest
from . import audio, ffm, looks, titles

PROFILES = {
    "preview": {"size": (540, 960), "seg_preset": "ultrafast", "seg_crf": 20,
                "preset": "veryfast", "crf": 26},
    "final":   {"size": (1080, 1920), "seg_preset": "fast", "seg_crf": 16,
                "preset": "medium", "crf": 19},
}
FPS = 30
DIP_OUT, DIP_IN = 15, 6        # frames: a slow dip out where the day changes
DISSOLVE = 15                  # frames of a photo-to-photo dissolve
END_FADE = 1.0                 # seconds


def _ease(u):
    return u * u * (3.0 - 2.0 * u)


def _key(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _n(seg):
    return max(1, int(round(seg["dur"] * FPS)))


def _finish_filters(seg, cube, gains=True):
    """Colour and the in/out dips, after the picture is framed and scaled."""
    vf = []
    if gains and seg.get("gain"):
        r, g, b = seg["gain"]
        vf.append("colorchannelmixer=rr=%.4f:gg=%.4f:bb=%.4f" % (r, g, b))
    vf.append("lut3d=%s" % cube)
    fx, n = seg.get("fx") or {}, _n(seg)
    if fx.get("in") == "dip":
        vf.append("fade=t=in:s=0:n=%d" % DIP_IN)
    if fx.get("out") == "dip":
        vf.append("fade=t=out:s=%d:n=%d" % (max(0, n - DIP_OUT), DIP_OUT))
    if fx.get("out") == "fade":
        k = int(END_FADE * FPS)
        vf.append("fade=t=out:s=%d:n=%d" % (max(0, n - k), k))
    vf.append("format=yuv420p")
    return vf


def _encode_cmd(out, prof, n, inputs, vf_graph):
    return [ffm.FFMPEG, "-v", "error", "-y"] + inputs + [
        "-filter_complex", vf_graph, "-map", "[v]", "-an", "-frames:v", str(n),
        "-c:v", "libx264", "-preset", prof["seg_preset"], "-crf", str(prof["seg_crf"]),
        "-g", "15", out]


def _stab_prefix(seg, clip, work):
    """Detect camera shake over the shot's source span; return the transform
    filter (or "" when the shot is steady enough)."""
    if not seg.get("stabilise"):
        return ""
    trf = os.path.join(work, "stab_%s.trf" % _key(clip.path, seg["in"], seg["out"]))
    if not os.path.exists(trf):
        ffm.run([ffm.FFMPEG, "-v", "error", "-ss", "%.3f" % seg["in"], "-i", clip.path,
                 "-t", "%.3f" % (seg["out"] - seg["in"]), "-vf",
                 "fps=%d,vidstabdetect=shakiness=6:accuracy=12:result=%s" % (FPS, trf),
                 "-f", "null", "-"])
    return "vidstabtransform=input=%s:smoothing=12:zoom=4:optzoom=0," % trf


def _render_clip(seg, clip, out, prof, cube, work):
    if seg.get("track"):
        return _render_clip_frames(seg, clip, out, prof, cube, work)
    W, H = prof["size"]
    n = _n(seg)
    sp = seg.get("speed", 1.0)
    x, y, w, h = seg.get("window") or _centre_window(clip.width, clip.height)
    cw = ffm.even(min(clip.width, w * clip.width))
    ch = ffm.even(min(clip.height, h * clip.height))
    cx = int(min(max(0, x * clip.width), clip.width - cw))
    cy = int(min(max(0, y * clip.height), clip.height - ch))
    vf = ([ffm.HDR_TO_SDR] if clip.hdr else [])
    if sp != 1.0:
        vf.append("setpts=PTS/%.3f" % sp)
    vf.append("fps=%d" % FPS)
    stab = _stab_prefix(seg, clip, work)
    if stab:
        vf.append(stab.rstrip(","))
    vf += ["crop=%d:%d:%d:%d" % (cw, ch, cx, cy),
           "scale=%d:%d:flags=lanczos" % (W, H), "setsar=1"] + _finish_filters(seg, cube)
    ffm.run(_encode_cmd(out, prof, n, ["-ss", "%.3f" % seg["in"],
                                       "-t", "%.3f" % (seg["out"] - seg["in"] + 0.2),
                                       "-i", clip.path],
                        "[0:v]%s[v]" % ",".join(vf)))


def _centre_window(w, h):
    aspect = 9.0 / 16.0
    if w / float(h) > aspect:
        cw = h * aspect / w
        return [(1 - cw) / 2.0, 0.0, cw, 1.0]
    chh = w / aspect / h
    return [0.0, (1 - chh) / 2.0, 1.0, chh]


def _pipe(out, prof, n, seg, cube):
    """An encoder reading raw RGB frames on stdin, with the shot's colour."""
    W, H = prof["size"]
    inputs = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
              "-r", str(FPS), "-i", "-"]
    vf = "[0:v]%s[v]" % ",".join(_finish_filters(seg, cube))
    return subprocess.Popen(_encode_cmd(out, prof, n, inputs, vf),
                            stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def _close(proc, what):
    try:
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode("utf-8", "replace")
    if proc.wait() != 0:
        raise ffm.FfmpegError("%s failed: %s" % (what, err.strip()[-300:]))


def _render_clip_frames(seg, clip, out, prof, cube, work):
    """A clip whose window moves - following a person, or drifting across
    scenery. Frames are decoded at the size where the window is exactly the
    output, then moved with sub-pixel accuracy."""
    W, H = prof["size"]
    n = _n(seg)
    x, y, w, h = seg.get("window") or _centre_window(clip.width, clip.height)
    scale = W / (w * clip.width)
    dw, dh = ffm.even(clip.width * scale), ffm.even(clip.height * scale)
    track = seg["track"]
    stab = _stab_prefix(seg, clip, work).rstrip(",") or None
    proc = _pipe(out, prof, n, seg, cube)
    last = None
    try:
        frames = ffm.decode_frames(clip.path, seg["in"], n / float(FPS) + 0.2, FPS,
                                   (dw, dh), hdr=clip.hdr, extra_vf=stab)
        for i in range(n):
            src = next(frames, None)
            src = last if src is None else src
            if src is None:
                break
            last = src
            left = track[min(i, len(track) - 1)] * dw
            M = np.array([[1.0, 0, left], [0, 1.0, y * dh]], dtype=np.float64)
            frame = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_REFLECT)
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    except BrokenPipeError:
        pass
    _close(proc, "clip shot")


def _photo_frames(seg, photo, prof):
    """Yield the frames of a photo shot: a gentle drift in or out."""
    W, H = prof["size"]
    n = _n(seg)
    img = ingest.load(photo, max_long_edge=int(max(W, H) * 1.6))
    # The album's corrections (levels, colour balance, tone recovery); the
    # look itself comes from the same colour table as the clips.
    img, _ = enhance.apply(img, photo, enabled=True, look="natural")
    src = np.asarray(img.convert("RGB"), dtype=np.uint8)
    sh, sw = src.shape[:2]
    kb = seg.get("kenburns") or {"start": _centre_window(sw, sh),
                                 "end": _centre_window(sw, sh)}
    a, b = np.array(kb["start"], float), np.array(kb["end"], float)
    for i in range(n):
        u = _ease(i / float(max(1, n - 1)))
        x, y, w, h = a + (b - a) * u
        M = np.array([[w * sw / W, 0, x * sw], [0, h * sh / H, y * sh]], dtype=np.float64)
        yield cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REFLECT)


def _render_photo(seg, photo, out, prof, cube, prev_last=None):
    """A photo shot; with a dissolve in, its first frames blend from the
    last frame of the photo before it."""
    n = _n(seg)
    proc = _pipe(out, prof, n, dict(seg, gain=None), cube)
    last = None
    try:
        for i, frame in enumerate(_photo_frames(seg, photo, prof)):
            if prev_last is not None and i < DISSOLVE:
                k = _ease((i + 1) / float(DISSOLVE + 1))
                frame = cv2.addWeighted(prev_last, 1.0 - k, frame, k, 0.0)
            last = frame
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    except BrokenPipeError:
        pass
    _close(proc, "photo shot")
    return last


def render(recipe, event_dir, sources, out_path, work_dir, profile="preview",
           music_path=None, progress=None):
    """Render the recipe to out_path. `sources` maps rel path -> Clip/Photo."""
    prof = PROFILES[profile]
    W, H = prof["size"]
    os.makedirs(work_dir, exist_ok=True)
    cubes = {}

    def cube_for(look):
        if look not in cubes:
            cubes[look] = looks.build(look, os.path.join(work_dir, "looks"))
        return cubes[look]

    parts, prev_photo_last, prev_key = [], None, None
    for i, seg in enumerate(recipe["segments"]):
        src = sources[seg["src"]]
        look = seg.get("look") or recipe["style"].get("look", "natural")
        dissolve = (seg.get("fx") or {}).get("in") == "dissolve"
        key = _key(seg, profile, look, src.path, os.path.getmtime(src.path), 3,
                   prev_key if dissolve else None)
        out = os.path.join(work_dir, "shot_%s.mp4" % key)
        if seg["kind"] == "photo":
            # A following dissolve needs this shot's last frame, so in that
            # case it is drawn again even when cached; otherwise reused.
            segs_ = recipe["segments"]
            feeds = i + 1 < len(segs_) and (segs_[i + 1].get("fx") or {}).get("in") == "dissolve"
            if not os.path.exists(out) or feeds:
                tmp = out + ".part.mp4"
                prev_photo_last = _render_photo(seg, src, tmp, prof, cube_for(look),
                                                prev_photo_last if dissolve else None)
                os.replace(tmp, out)
            else:
                prev_photo_last = None
        else:
            prev_photo_last = None
            if not os.path.exists(out):
                tmp = out + ".part.mp4"
                _render_clip(seg, src, tmp, prof, cube_for(look), work_dir)
                os.replace(tmp, out)
        prev_key = key
        os.utime(out)               # mark as used, so pruning keeps it
        parts.append(out)
        if progress:
            progress(i + 1, len(recipe["segments"]), seg["src"])

    total = sum(_n(s) for s in recipe["segments"]) / float(FPS)

    # Sound: mixed on its own timeline, then loudness measured and set.
    wav = os.path.join(work_dir, "mix_%s.wav" % _key(recipe["segments"], recipe["audio"]))
    if not os.path.exists(wav):
        audio.mix(recipe, sources, music_path, wav)
    loud = audio.loudness_filter(wav)

    # Text: the title over the opener, a caption at each new day.
    pngs = []
    for k, ov in enumerate(recipe.get("overlays", [])):
        png = os.path.join(work_dir, "text_%s.png" % _key(ov, W, H))
        if not os.path.exists(png):
            if ov["kind"] == "title":
                img = titles.render(ov["text"], ov.get("subtitle"), W, H, y=ov.get("y", 0.4))
            else:
                img = titles.caption(ov["text"], W, H)
            img.save(png)
        pngs.append((png, ov))

    cmd = [ffm.FFMPEG, "-v", "error", "-y"]
    for p in parts:
        cmd += ["-i", p]
    k = len(parts)
    for png, _ov in pngs:
        cmd += ["-loop", "1", "-t", "%.3f" % total, "-i", png]
    cmd += ["-i", wav]
    g = "".join("[%d:v]" % i for i in range(k)) + "concat=n=%d:v=1:a=0[v0];" % k
    last = "v0"
    for j, (png, ov) in enumerate(pngs):
        t0, t1 = ov["t0"], max(ov["t0"] + 0.8, ov["t1"])
        g += ("[%d:v]format=rgba,fade=t=in:st=%.2f:d=0.4:alpha=1,"
              "fade=t=out:st=%.2f:d=0.4:alpha=1[t%d];" % (k + j, t0, t1 - 0.4, j))
        g += "[%s][t%d]overlay=0:0:enable='between(t,%.2f,%.2f)'[v%d];" % (
            last, j, t0, t1, j + 1)
        last = "v%d" % (j + 1)
    g += "[%s]format=yuv420p[v];[%d:a]%s[a]" % (last, k + len(pngs), loud)
    cmd += ["-filter_complex", g, "-map", "[v]", "-map", "[a]",
            "-t", "%.3f" % total, "-r", str(FPS),
            "-c:v", "libx264", "-preset", prof["preset"], "-crf", str(prof["crf"]),
            "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-g", "60", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", out_path]
    ffm.run(cmd)
    return out_path


WORK_LIMIT = 2 * 1024 ** 3     # bytes of cached shots kept between runs


def prune(work_dir, keep_since, limit=WORK_LIMIT):
    """Keep the scratch folder small: when it holds more than `limit`, delete
    the cached files not used since `keep_since` (oldest first) until it is
    back under. Files the last run used are always kept."""
    if not os.path.isdir(work_dir):
        return 0
    files = []
    for name in os.listdir(work_dir):
        path = os.path.join(work_dir, name)
        if os.path.isfile(path):
            st = os.stat(path)
            files.append((st.st_mtime, st.st_size, path))
    total = sum(f[1] for f in files)
    freed = 0
    for mtime, size, path in sorted(files):
        if total <= limit or mtime >= keep_since:
            break
        os.remove(path)
        total -= size
        freed += size
    return freed


def clean(work_dir):
    """Delete the whole scratch folder. Returns the bytes freed."""
    import shutil
    if not os.path.isdir(work_dir):
        return 0
    size = sum(os.path.getsize(os.path.join(d, f))
               for d, _s, fs in os.walk(work_dir) for f in fs)
    shutil.rmtree(work_dir, ignore_errors=True)
    return size


def strip(video, out_jpg, frames=8, height=240):
    """A row of frames across the video, for the comparison page."""
    info = ffm.probe_json(video)
    d = float(info.get("format", {}).get("duration", 0) or 0)
    if d <= 0:
        return None
    ffm.run([ffm.FFMPEG, "-v", "error", "-y", "-i", video, "-vf",
             "fps=%.4f,scale=-2:%d,tile=%dx1" % (frames / d, height, frames),
             "-frames:v", "1", out_jpg])
    return out_jpg
