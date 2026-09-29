"""Face detection - the single most important content signal in an event album.

Two detectors, in order of preference:

  YuNet   a 230 KB convolutional model bundled in album/models/. Runs entirely
          offline on the CPU, handles tilted heads, profiles and small faces,
          and returns a confidence score. This is what actually gets used.

  Haar    OpenCV's classic cascades, used only if the model file is missing.
          Kept because it needs no download, but it both misses tilted faces
          and invents faces in textured scenes, so it is a fallback, not a peer.

Whichever runs, the result is the same shape, and `available()` reports honestly
whether detection is working at all - a silently dead detector once made every
photo in a 92-photo folder score zero faces.
"""
import os

from .deps import np, cv2, HAS_CV2

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models")
YUNET_FILE = "face_detection_yunet_2023mar.onnx"

YUNET_SCORE = 0.72       # confidence floor; lower lets in furniture and patterns
YUNET_NMS = 0.30
DETECT_EDGE = 640        # faces are found on a small copy - fast and stable

_detector = None
_backend = None
_haar = None
_eye_cascades = None


def _yunet_path():
    return os.path.join(MODEL_DIR, YUNET_FILE)


def _init():
    """Pick a backend once, then reuse it for the whole run."""
    global _detector, _backend, _haar
    if _backend is not None:
        return _backend

    _backend = "none"
    if not HAS_CV2:
        return _backend

    path = _yunet_path()
    if hasattr(cv2, "FaceDetectorYN") and os.path.exists(path):
        try:
            _detector = cv2.FaceDetectorYN.create(
                path, "", (DETECT_EDGE, DETECT_EDGE),
                score_threshold=YUNET_SCORE, nms_threshold=YUNET_NMS, top_k=500)
            _backend = "yunet"
            return _backend
        except Exception:
            _detector = None

    if hasattr(cv2, "CascadeClassifier"):
        base = getattr(cv2.data, "haarcascades", "")
        cascades = []
        for fn in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"):
            full = os.path.join(base, fn)
            if os.path.exists(full):
                c = cv2.CascadeClassifier(full)
                if not c.empty():
                    cascades.append(c)
        if cascades:
            _haar = cascades
            _backend = "haar"
    return _backend


def _eyes():
    """Haar eye cascades, loaded once.

    These only fire on an OPEN eye - a closed lid has none of the structure
    they were trained on. That limitation is exactly what makes them useful
    here: a face with no eye detected inside it is very probably a blink.
    """
    global _eye_cascades
    if _eye_cascades is not None:
        return _eye_cascades
    _eye_cascades = []
    if HAS_CV2 and hasattr(cv2, "CascadeClassifier"):
        base = getattr(cv2.data, "haarcascades", "")
        for fn in ("haarcascade_eye_tree_eyeglasses.xml", "haarcascade_eye.xml"):
            full = os.path.join(base, fn)
            if os.path.exists(full):
                c = cv2.CascadeClassifier(full)
                if not c.empty():
                    _eye_cascades.append(c)
    return _eye_cascades


def eyes_open(rgb, box):
    """Are this face's eyes open? Returns True, False, or None if unknown.

    Only the upper half of the face box is searched, which is where eyes are
    and which keeps nostrils and teeth from being mistaken for them.
    """
    cascades = _eyes()
    if not cascades:
        return None
    x, y, w, h, _score = box
    if w < 22 or h < 22:
        return None                # too small to judge; do not guess

    H, W = rgb.shape[:2]
    x0 = int(max(0, x + w * 0.06))
    x1 = int(min(W, x + w * 0.94))
    y0 = int(max(0, y + h * 0.16))
    y1 = int(min(H, y + h * 0.62))     # upper face only
    if x1 - x0 < 20 or y1 - y0 < 12:
        return None

    region = rgb[y0:y1, x0:x1]
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

    # Eyes in a group shot are only a dozen pixels across at analysis size,
    # which is below what the cascades can resolve. Enlarging the strip first
    # turns a near-certain miss into a usable answer.
    target_w = 260
    if gray.shape[1] < target_w:
        f = target_w / float(gray.shape[1])
        gray = cv2.resize(gray, (target_w, max(8, int(gray.shape[0] * f))),
                          interpolation=cv2.INTER_CUBIC)
    gray = cv2.equalizeHist(gray)

    min_eye = max(10, int(gray.shape[1] * 0.10))
    for c in cascades:
        try:
            found = c.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3,
                                       minSize=(min_eye, min_eye))
        except Exception:
            continue
        if len(found) >= 1:
            return True
    return False


def available():
    return _init() != "none"


def backend():
    return _init()


def describe():
    b = _init()
    return {"yunet": "YuNet (bundled model)",
            "haar": "Haar cascades (fallback - less accurate)",
            "none": "unavailable"}[b]


def _detect_yunet(rgb):
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _detector.setInputSize((w, h))
    ok, raw = _detector.detect(bgr)
    if raw is None:
        return []
    out = []
    for row in raw:
        x, y, fw, fh = row[0], row[1], row[2], row[3]
        score = float(row[-1])
        if fw <= 1 or fh <= 1:
            continue
        out.append((float(x), float(y), float(fw), float(fh), score))
    return out


def _detect_haar(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    minsize = max(16, int(min(gray.shape) * 0.05))
    boxes = []
    for c in _haar:
        try:
            found = c.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=6,
                                       minSize=(minsize, minsize))
        except Exception:
            continue
        for b in found:
            boxes.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), 0.5))

    # Haar double-reports the same face across cascades; keep the biggest.
    kept = []
    for box in sorted(boxes, key=lambda b: -b[2] * b[3]):
        x, y, w, h, s = box
        if all(abs(x - kx) > kw * 0.5 or abs(y - ky) > kh * 0.5
               for kx, ky, kw, kh, _ in kept):
            kept.append(box)
    return kept


def detect(rgb):
    """Find faces in an RGB uint8 array.

    Returns a list of (x, y, w, h, confidence) in that array's pixel space.
    """
    backend_name = _init()
    if backend_name == "none" or rgb is None or rgb.size == 0:
        return []

    h, w = rgb.shape[:2]
    scale = min(1.0, DETECT_EDGE / float(max(h, w)))
    if scale < 1.0:
        small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = rgb

    try:
        found = (_detect_yunet(small) if backend_name == "yunet"
                 else _detect_haar(small))
    except Exception:
        return []

    if scale < 1.0 and found:
        inv = 1.0 / scale
        found = [(x * inv, y * inv, fw * inv, fh * inv, s) for x, y, fw, fh, s in found]

    # Clamp to the frame - YuNet can return boxes that run off the edge.
    clamped = []
    for x, y, fw, fh, s in found:
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(w), x + fw), min(float(h), y + fh)
        if x1 - x0 > 2 and y1 - y0 > 2:
            clamped.append((x0, y0, x1 - x0, y1 - y0, s))
    return clamped


def eye_state(rgb, boxes, max_faces=6):
    """Open-eye check across the biggest faces in the frame.

    Returns (open_count, judged_count). Only the largest few faces are tested:
    a stranger forty metres away blinking is not a reason to reject a picture,
    and each check costs a cascade pass.
    """
    if not boxes:
        return 0, 0
    ordered = sorted(boxes, key=lambda b: -(b[2] * b[3]))[:max_faces]
    opened = judged = 0
    for box in ordered:
        state = eyes_open(rgb, box)
        if state is None:
            continue
        judged += 1
        if state:
            opened += 1
    return opened, judged


def summarise(boxes, frame_w, frame_h):
    """Reduce a box list to the numbers the scorer and the cropper need."""
    if not boxes or frame_w <= 0 or frame_h <= 0:
        return {"count": 0, "total_area": 0.0, "largest_area": 0.0,
                "centre": None, "boxes": []}
    frame = float(frame_w * frame_h)
    areas = [w * h for _, _, w, h, _ in boxes]
    total = sum(areas) / frame
    largest = max(areas) / frame
    wsum = sum(areas)
    cx = sum((x + w / 2.0) * a for (x, _, w, _, _), a in zip(boxes, areas)) / wsum / frame_w
    cy = sum((y + h / 2.0) * a for (_, y, _, h, _), a in zip(boxes, areas)) / wsum / frame_h
    norm = [(x / frame_w, y / frame_h, w / frame_w, h / frame_h, s)
            for x, y, w, h, s in boxes]
    return {"count": len(boxes), "total_area": float(total),
            "largest_area": float(largest), "centre": (float(cx), float(cy)),
            "boxes": norm}
