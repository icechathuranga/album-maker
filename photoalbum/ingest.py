"""Find photos in an event folder, read their EXIF, load them upright."""
import os
import re
import datetime

from .deps import Image, ImageOps, HAS_HEIF, np

RASTER_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
HEIF_EXT = {".heic", ".heif", ".avif"}
# Raw files need dcraw/rawpy; listed so we can tell the user why they were skipped.
RAW_EXT = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf", ".srw"}

SKIP_DIRS = {"__pycache__", ".git", "@eaDir", ".thumbnails"}


def supported_ext():
    exts = set(RASTER_EXT)
    if HAS_HEIF:
        exts |= HEIF_EXT
    return exts


class Photo:
    """One source image plus everything the pipeline learns about it."""

    __slots__ = ("path", "rel", "name", "ext", "bytes", "taken", "taken_src",
                 "width", "height", "orientation", "camera", "scores", "score",
                 "technical", "content", "hero", "segment", "face_boxes", "crop_box",
                 "gps", "place", "extra_rotation",
                 "reasons", "status", "group", "dhash", "seq", "enhanced_path",
                 "used_slot")

    def __init__(self, path, root):
        self.path = path
        self.rel = os.path.relpath(path, root)
        self.name = os.path.basename(path)
        self.ext = os.path.splitext(path)[1].lower()
        self.bytes = os.path.getsize(path)
        self.taken = None
        self.taken_src = "none"
        self.width = 0
        self.height = 0
        self.orientation = 1
        self.camera = ""
        self.scores = {}
        self.score = 0.0
        self.technical = 0.0
        self.content = 0.0
        self.hero = False
        self.segment = 0
        self.face_boxes = []
        self.crop_box = None
        self.gps = None
        self.place = None
        self.extra_rotation = 0
        self.reasons = []
        self.status = "pending"
        self.group = -1
        self.dhash = None
        self.seq = 0
        self.enhanced_path = None
        self.used_slot = None

    @property
    def megapixels(self):
        return (self.width * self.height) / 1e6

    @property
    def aspect(self):
        return (self.width / self.height) if self.height else 1.0

    @property
    def shape(self):
        a = self.aspect
        if a >= 1.15:
            return "landscape"
        if a <= 0.87:
            return "portrait"
        return "square"

    def to_dict(self):
        return {
            "name": self.name, "rel": self.rel, "status": self.status,
            "score": round(self.score, 1), "technical": round(self.technical * 100, 1),
            "content": round(self.content * 100, 1), "hero": self.hero,
            "segment": self.segment, "width": self.width, "height": self.height,
            "megapixels": round(self.megapixels, 2), "shape": self.shape,
            "taken": self.taken.isoformat(sep=" ") if self.taken else None,
            "taken_source": self.taken_src, "group": self.group,
            "gps": list(self.gps) if self.gps else None,
            "place": (self.place[0] + ", " + self.place[1]) if self.place else None,
            "reasons": self.reasons, "metrics": {k: round(v, 3) for k, v in self.scores.items()},
        }


def scan(folder):
    """Return (photos, skipped) for an event folder, recursing into subfolders."""
    exts = supported_ext()
    photos, skipped = [], []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in sorted(filenames):
            if fn.startswith(".") or fn.startswith("._"):
                continue
            ext = os.path.splitext(fn)[1].lower()
            full = os.path.join(dirpath, fn)
            if ext in exts:
                try:
                    if os.path.getsize(full) < 1024:
                        skipped.append((fn, "file too small to be a photo"))
                        continue
                except OSError:
                    continue
                photos.append(Photo(full, folder))
            elif ext in RAW_EXT:
                skipped.append((fn, "RAW format - export to JPEG first"))
            elif ext in HEIF_EXT:
                skipped.append((fn, "HEIC support not installed (pip install pillow-heif)"))
    return photos, skipped


def _exif_datetime(img):
    """DateTimeOriginal (36867) then DateTimeDigitized (36868) then DateTime (306)."""
    try:
        raw = img.getexif()
    except Exception:
        return None, 1, ""
    if not raw:
        return None, 1, ""

    orientation = 1
    try:
        orientation = int(raw.get(274, 1)) or 1
    except Exception:
        orientation = 1

    make = str(raw.get(271, "") or "").strip()
    model = str(raw.get(272, "") or "").strip()
    camera = (make + " " + model).strip()

    val = None
    try:
        sub = raw.get_ifd(0x8769)  # ExifIFD
    except Exception:
        sub = {}
    for tag in (36867, 36868):
        v = sub.get(tag) if sub else None
        if v:
            val = v
            break
    if not val:
        val = raw.get(306)

    dt = None
    if val:
        s = str(val).strip().rstrip("\x00")
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M",
                    "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.datetime.strptime(s[:len(fmt) + 2].strip(), fmt)
                break
            except ValueError:
                continue
    return dt, orientation, camera


# Cameras and phones stamp the capture time into the filename, and unlike the
# file's modification time that survives copying, syncing and re-downloading.
# It is the best fallback there is when EXIF has been stripped.
_NAME_PATTERNS = (
    # IMG20260301105643 / VID20260301105643
    (re.compile(r"(?:^|[^0-9])(\d{8})(\d{6})(?:[^0-9]|$)"), True),
    # IMG_20260301_105643 / PXL_20260301_105643123 / 20260301-105643
    (re.compile(r"(?:^|[^0-9])(\d{8})[_\-\.](\d{6})"), True),
    # IMG-20260906-WA0027: the date a message arrived, with no time in it
    (re.compile(r"(?:^|[^0-9])(\d{8})(?:[^0-9]|$)"), False),
)


def _filename_datetime(name):
    """Read a capture time out of a filename. Returns (datetime, has_time)."""
    stem = os.path.splitext(name)[0]
    for pattern, has_time in _NAME_PATTERNS:
        m = pattern.search(stem)
        if not m:
            continue
        try:
            date = datetime.datetime.strptime(m.group(1), "%Y%m%d")
        except ValueError:
            continue
        if date.year < 1990 or date > datetime.datetime.now() + datetime.timedelta(days=2):
            continue
        if has_time:
            try:
                t = datetime.datetime.strptime(m.group(2), "%H%M%S").time()
            except ValueError:
                return date, False
            return datetime.datetime.combine(date.date(), t), True
        return date, False
    return None, False


def probe(photo):
    """Fill in dimensions and capture time without decoding full pixels."""
    try:
        with Image.open(photo.path) as img:
            w, h = img.size
            dt, orientation, camera = _exif_datetime(img)
            try:
                from . import places as _places
                photo.gps = _places.gps_from_exif(img.getexif())
            except Exception:
                photo.gps = None
    except Exception as exc:
        photo.status = "unreadable"
        photo.reasons.append("could not be opened (%s)" % type(exc).__name__)
        return False

    photo.orientation = orientation
    photo.camera = camera
    # Orientations 5-8 swap the axes once the image is rotated upright.
    if orientation in (5, 6, 7, 8):
        w, h = h, w
    photo.width, photo.height = w, h

    if dt:
        photo.taken, photo.taken_src = dt, "exif"
        return True

    # No EXIF. Try the filename before falling back to the file's own dates,
    # which usually record when it was copied rather than when it was taken.
    named, has_time = _filename_datetime(photo.name)
    if named and has_time:
        photo.taken, photo.taken_src = named, "filename"
    elif named:
        photo.taken, photo.taken_src = named, "filename-date-only"
    else:
        photo.taken = datetime.datetime.fromtimestamp(os.path.getmtime(photo.path))
        photo.taken_src = "file-mtime"
    return True


OUTLIER_PAD = datetime.timedelta(days=2)


UNKNOWN_TIME = "unknown"


def reconcile_times(photos):
    """Repair timestamps for files that lost their EXIF.

    Copying a folder, downloading from a chat app or exporting from an editor
    strips EXIF and leaves the file modification time set to the moment of the
    copy - often months after the event. Left alone, those photos land in a
    bogus segment of their own and break the album's chronology.

    Where enough EXIF-dated neighbours exist, an out-of-range file is slotted
    back beside the files it sits next to alphabetically, which is almost
    always the order the camera wrote them in.
    """
    dated = [p for p in photos if p.taken_src in ("exif", "filename") and p.taken]
    if len(dated) < 3:
        return 0

    lo = min(p.taken for p in dated) - OUTLIER_PAD
    hi = max(p.taken for p in dated) + OUTLIER_PAD

    by_name = sorted(photos, key=lambda p: p.name.lower())
    fixed = 0
    for i, p in enumerate(by_name):
        # A filename that carried a real time is trusted; only copy dates and
        # bare dates get second-guessed.
        if p.taken_src in ("exif", "filename") or not p.taken:
            continue
        if lo <= p.taken <= hi:
            continue  # mtime is plausible, keep it

        trusted = ("exif", "filename")
        before = next((q for q in reversed(by_name[:i])
                       if q.taken_src in trusted), None)
        after = next((q for q in by_name[i + 1:] if q.taken_src in trusted), None)

        # Interpolating only makes sense when the file sits BETWEEN two dated
        # neighbours - that is evidence it belongs there. A file that sorts
        # outside the whole dated run has no such evidence: messaging apps
        # strip EXIF entirely, and their names ("IMG-20260906-WA0098") carry
        # only the date the message arrived. Guessing put 245 such photos at
        # the start of a trip they were taken during. Better to admit the time
        # is unknown and keep them out of the chronology.
        if before and after:
            p.taken = before.taken + (after.taken - before.taken) / 2
            p.taken_src = "inferred-from-neighbours"
            fixed += 1
        else:
            p.taken_src = UNKNOWN_TIME
    return fixed


def detect_upright(photo, rgb_loader, face_detector):
    """Work out which way up a photo goes when EXIF cannot say.

    Messaging apps strip EXIF wholesale, orientation tag included, so a frame
    the camera stored sideways stays sideways with nothing to flag it. Faces
    settle it: a detector trained on upright faces finds none in a sideways
    photo and several once it is turned the right way.

    Returns the rotation in degrees (0, 90 or 270) that should be applied.
    """
    try:
        img = rgb_loader(photo)
    except Exception:
        return 0

    if face_detector(img):
        return 0                       # already upright, or no faces either way

    best_turn, best_count = 0, 0
    for turn in (90, 270):
        try:
            found = face_detector(img.rotate(turn, expand=True))
        except Exception:
            continue
        if len(found) > best_count:
            best_turn, best_count = turn, len(found)
    return best_turn if best_count else 0


def order(photos):
    """Chronological, with a stable filename tiebreak - albums read as a story.

    Photos whose capture time is genuinely unknown go after everything dated,
    in filename order, rather than being wedged into the timeline at a guess.
    They are still eligible for the album; they simply cannot claim a place in
    the sequence they were never able to prove.
    """
    def key(p):
        unknown = (p.taken_src == UNKNOWN_TIME) or p.taken is None
        return (1 if unknown else 0,
                p.taken or datetime.datetime.max,
                p.name.lower())

    photos.sort(key=key)
    for i, p in enumerate(photos):
        p.seq = i
    return photos


def undated(photos):
    """Photos whose capture time could not be established."""
    return [p for p in photos if p.taken_src == UNKNOWN_TIME or p.taken is None]


def load(photo, max_long_edge=None):
    """Open an image upright in RGB, optionally downscaled for analysis."""
    img = Image.open(photo.path)
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")
    if photo.extra_rotation:
        img = img.rotate(photo.extra_rotation, expand=True)
    if max_long_edge:
        long_edge = max(img.size)
        if long_edge > max_long_edge:
            scale = max_long_edge / float(long_edge)
            new = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            img = img.resize(new, Image.BILINEAR)
    return img


def to_gray(img):
    """Float32 luminance array in 0..255."""
    return np.asarray(img.convert("L"), dtype=np.float32)


def native_crop_gray(photo, size=512):
    """Centre crop at ORIGINAL pixel scale, aligned to the JPEG 8x8 grid.

    Compression-artefact detection only works on unresampled pixels, so this
    deliberately skips the downscale that the other metrics use.
    """
    try:
        with Image.open(photo.path) as img:
            if img.mode != "L":
                img = img.convert("L")
            w, h = img.size
            cw, ch = min(size, w), min(size, h)
            left = ((w - cw) // 2) & ~7      # keep the crop 8-aligned
            top = ((h - ch) // 2) & ~7
            crop = img.crop((left, top, left + cw, top + ch))
            return np.asarray(crop, dtype=np.float32)
    except Exception:
        return np.zeros((16, 16), dtype=np.float32)
