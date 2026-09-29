"""Dependency shim.

Everything heavy is optional. If a library is missing the pipeline degrades to
a numpy/PIL fallback instead of dying, so the tool still produces a PDF.
"""
import os
import sys

# vendor/ sits next to the package dir's parent (album/vendor)
_VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor")
if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

import numpy as np  # noqa: E402  (hard requirement)
from PIL import (Image, ImageOps, ImageFilter, ImageDraw, ImageFont,  # noqa: E402,F401
                 ImageEnhance, ImageStat)

# Big files are normal for cameras; don't warn about decompression bombs.
Image.MAX_IMAGE_PIXELS = None

try:
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HAS_HEIF = True
except Exception:
    HAS_HEIF = False

try:
    import piexif
    HAS_PIEXIF = True
except Exception:
    piexif = None
    HAS_PIEXIF = False

try:
    from reportlab.pdfgen import canvas as _rl_canvas
    from reportlab.lib.utils import ImageReader  # noqa: F401
    HAS_REPORTLAB = True
except Exception:
    _rl_canvas = None
    HAS_REPORTLAB = False


def report():
    """Human-readable capability line for the CLI banner.

    Face detection is named explicitly because a silently dead detector once
    scored an entire 92-photo folder as containing no people at all.
    """
    from . import faces
    bits = [
        ("OpenCV", HAS_CV2),
        ("HEIC/HEIF", HAS_HEIF),
        ("EXIF", HAS_PIEXIF),
        ("PDF", HAS_REPORTLAB),
    ]
    line = "  ".join(("+" if ok else "-") + name for name, ok in bits)
    return line + "  |  faces: " + faces.describe()
