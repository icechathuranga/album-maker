"""Read what a video file is: upright size, length, frame rate, HDR, audio, time.

A `Clip` carries the same attribute names as `ingest.Photo` where they mean
the same thing (path, name, taken, taken_src, seq, segment, status, scores),
so the album's chapter splitting works on clips and photos together.
"""
import datetime
import os
import re

from .. import ingest
from . import ffm

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".3gp", ".webm", ".mkv", ".avi", ".mts"}


class Clip:
    kind = "clip"

    def __init__(self, path, root):
        self.path = path
        self.rel = os.path.relpath(path, root)
        self.name = os.path.basename(path)
        self.bytes = os.path.getsize(path)
        self.mtime = os.path.getmtime(path)
        self.taken = None
        self.taken_src = ingest.UNKNOWN_TIME
        self.seq = 0
        self.segment = 0
        self.width = self.height = 0          # upright, after rotation
        self.duration = 0.0
        self.fps = 30.0
        self.vfr = False
        self.rotation = 0
        self.codec = ""
        self.hdr = False
        self.has_audio = False
        self.gps = None
        self.place = None
        self.status = "pending"
        self.reasons = []
        self.scores = {}
        self.timeline = None                  # filled by analyse
        self.dhash = None

    @property
    def aspect(self):
        return self.width / float(self.height) if self.height else 1.0

    def __repr__(self):
        return "<Clip %s %dx%d %.1fs>" % (self.name, self.width, self.height, self.duration)


def scan(folder):
    """Video files in an event folder, sorted by name."""
    out = []
    for dirpath, _dirs, files in os.walk(folder):
        for fn in sorted(files):
            if fn.startswith("."):
                continue
            if os.path.splitext(fn)[1].lower() in VIDEO_EXT:
                out.append(Clip(os.path.join(dirpath, fn), folder))
    out.sort(key=lambda c: c.rel)
    for i, c in enumerate(out):
        c.seq = i
    return out


def _fraction(text):
    try:
        num, den = text.split("/")
        return float(num) / float(den) if float(den) else 0.0
    except (ValueError, AttributeError):
        return 0.0


def _rotation(stream):
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            try:
                return int(round(float(side["rotation"]))) % 360
            except ValueError:
                pass
    try:
        return int(stream.get("tags", {}).get("rotate", 0)) % 360
    except ValueError:
        return 0


_ISO6709 = re.compile(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)")


def _gps(tags):
    loc = tags.get("com.apple.quicktime.location.ISO6709") or tags.get("location")
    if not loc:
        return None
    m = _ISO6709.match(loc)
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    if abs(lat) > 90 or abs(lon) > 180 or (lat == 0 and lon == 0):
        return None
    return (lat, lon)


def _parse_time(text):
    """ISO-ish timestamps as phones write them; returns (local datetime, had_tz)."""
    if not text:
        return None, False
    text = text.strip().replace("Z", "+0000")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.datetime.strptime(text, fmt)
        except ValueError:
            continue
        return dt, dt.tzinfo is not None
    return None, False


def probe(clip):
    """Fill in a Clip from ffprobe. Marks it unreadable rather than raising."""
    try:
        info = ffm.probe_json(clip.path)
    except ffm.FfmpegError as exc:
        clip.status = "unreadable"
        clip.reasons.append("not a video ffmpeg can read (damaged or unfinished copy?)")
        return clip
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        clip.status = "unreadable"
        clip.reasons.append("no video stream")
        return clip
    fmt = info.get("format", {})
    tags = dict(fmt.get("tags", {}) or {})
    tags.update(video.get("tags", {}) or {})

    w, h = int(video.get("width", 0)), int(video.get("height", 0))
    clip.rotation = _rotation(video)
    if clip.rotation in (90, 270):
        w, h = h, w
    clip.width, clip.height = w, h
    clip.codec = video.get("codec_name", "")
    clip.hdr = video.get("color_transfer", "") in ffm.HDR_TRANSFERS
    clip.has_audio = any(s.get("codec_type") == "audio" for s in streams)
    avg, nominal = _fraction(video.get("avg_frame_rate")), _fraction(video.get("r_frame_rate"))
    clip.fps = avg or nominal or 30.0
    clip.vfr = bool(avg and nominal and abs(avg - nominal) > 0.5)
    try:
        clip.duration = float(fmt.get("duration") or video.get("duration") or 0.0)
    except ValueError:
        clip.duration = 0.0
    clip.gps = _gps(tags)

    # Capture time: the filename first (phones write local time there and it
    # survives copying), then QuickTime's local creation date, then the
    # container's UTC creation time shown in this machine's zone, then mtime.
    dt, has_time = ingest._filename_datetime(clip.name)
    if dt and has_time:
        clip.taken, clip.taken_src = dt, "filename"
    else:
        local, _ = _parse_time(tags.get("com.apple.quicktime.creationdate"))
        if local:
            clip.taken, clip.taken_src = local.replace(tzinfo=None), "quicktime"
        else:
            utc, had_tz = _parse_time(tags.get("creation_time"))
            if utc and utc.year > 1990:
                if had_tz:
                    utc = utc.astimezone().replace(tzinfo=None)
                clip.taken, clip.taken_src = utc, "container"
            else:
                clip.taken = datetime.datetime.fromtimestamp(clip.mtime)
                clip.taken_src = "file-mtime"

    if clip.duration < 0.8:
        clip.status = "rejected"
        clip.reasons.append("shorter than a second")
    elif clip.width < 240 or clip.height < 240:
        clip.status = "rejected"
        clip.reasons.append("too small (%dx%d)" % (clip.width, clip.height))
    return clip
