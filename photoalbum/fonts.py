"""The display fonts, shared by album covers and reel titles so they match.

Each list is tried in order; the first file that exists wins. Nothing is
downloaded - these are the fonts commonly installed with Ubuntu.
"""
import os

_TT = "/usr/share/fonts/truetype/"

# A light, high-contrast serif for spaced capital titles.
SERIF = [_TT + "fonts-yrsa-rasa/Yrsa-Regular.ttf",
         _TT + "liberation2/LiberationSerif-Regular.ttf",
         _TT + "liberation/LiberationSerif-Regular.ttf",
         _TT + "dejavu/DejaVuSerif.ttf"]

# A light sans for small spaced subtitles and captions.
LIGHT = [_TT + "ubuntu/Ubuntu-L.ttf",
         _TT + "dejavu/DejaVuSans-ExtraLight.ttf",
         _TT + "liberation2/LiberationSans-Regular.ttf"]


def first(paths):
    """The first of these font files present on this machine, or None."""
    return next((p for p in paths if os.path.exists(p)), None)
