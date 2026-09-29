"""Where and when a photo was taken, for the small caption on a page.

Location comes from the GPS tags a phone writes into EXIF, resolved against a
list of world cities bundled in `models/places.tsv`. The lookup is a plain
nearest-neighbour search weighted slightly by population, so a photo taken in a
suburb is captioned with the city everybody has heard of rather than the
nearest hamlet.

Nothing here touches the network. Many phones have location switched off; what
happens then depends on the caption mode - "place" prints nothing at all rather
than putting text on a page for the sake of it, while "auto" falls back to the
day and the part of the day, which is always available.
"""
import math
import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models")
PLACES_FILE = os.path.join(MODEL_DIR, "places.tsv")

# How far a photo may be from a listed city before we stop naming it, in km.
MAX_DISTANCE_KM = 60.0

_places = None

COUNTRY_NAMES = {
    "TH": "Thailand", "LK": "Sri Lanka", "IN": "India", "SG": "Singapore",
    "MY": "Malaysia", "AE": "UAE", "GB": "United Kingdom", "US": "United States",
    "AU": "Australia", "FR": "France", "IT": "Italy", "ES": "Spain",
    "JP": "Japan", "ID": "Indonesia", "VN": "Vietnam", "DE": "Germany",
    "NL": "Netherlands", "CH": "Switzerland", "CA": "Canada", "NZ": "New Zealand",
    "QA": "Qatar", "SA": "Saudi Arabia", "MV": "Maldives", "NP": "Nepal",
}


def _load():
    global _places
    if _places is not None:
        return _places
    _places = []
    try:
        with open(PLACES_FILE, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                try:
                    _places.append((float(parts[0]), float(parts[1]),
                                    parts[2], parts[3], int(parts[4])))
                except ValueError:
                    continue
    except OSError:
        _places = []
    return _places


def available():
    return bool(_load())


def _ratio(value):
    """EXIF rationals arrive as IFDRational, tuple, or plain number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(value[0]) / float(value[1])
        except Exception:
            return 0.0


def gps_from_exif(exif):
    """Pull (latitude, longitude) in signed degrees out of EXIF, or None."""
    if not exif:
        return None
    try:
        gps = exif.get_ifd(0x8825)
    except Exception:
        return None
    if not gps:
        return None

    lat_ref = gps.get(1)
    lat = gps.get(2)
    lon_ref = gps.get(3)
    lon = gps.get(4)
    if not lat or not lon:
        return None

    try:
        latitude = (_ratio(lat[0]) + _ratio(lat[1]) / 60.0 + _ratio(lat[2]) / 3600.0)
        longitude = (_ratio(lon[0]) + _ratio(lon[1]) / 60.0 + _ratio(lon[2]) / 3600.0)
    except Exception:
        return None

    if str(lat_ref).upper().startswith("S"):
        latitude = -latitude
    if str(lon_ref).upper().startswith("W"):
        longitude = -longitude
    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        return None
    if abs(latitude) < 1e-6 and abs(longitude) < 1e-6:
        return None            # null island: the tag exists but holds nothing
    return (latitude, longitude)


def _haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def nearest(lat, lon, max_km=MAX_DISTANCE_KM):
    """Name the place these coordinates are in. Returns (name, country) or None."""
    places = _load()
    if not places:
        return None

    # Cheap bounding box first - a full haversine over 34k rows per photo adds up.
    dlat = max_km / 111.0
    dlon = max_km / max(1.0, 111.0 * math.cos(math.radians(lat)))
    box = [p for p in places
           if abs(p[0] - lat) <= dlat and abs(p[1] - lon) <= dlon]
    if not box:
        return None

    best, best_cost = None, None
    for plat, plon, name, cc, pop in box:
        km = _haversine(lat, lon, plat, plon)
        if km > max_km:
            continue
        # A big city slightly further away is the more useful answer than the
        # village next door - "Bangkok" beats the name of its outer district.
        cost = km / (1.0 + math.log10(max(pop, 1000)) / 3.0)
        if best_cost is None or cost < best_cost:
            best, best_cost = (name, cc), cost
    return best


def label(place):
    """Format a (name, country-code) pair for printing."""
    if not place:
        return ""
    name, cc = place
    country = COUNTRY_NAMES.get(cc, cc)
    return "%s, %s" % (name, country) if country else name


def part_of_day(dt):
    h = dt.hour
    if h < 5:
        return "night"
    if h < 12:
        return "morning"
    if h < 17:
        return "afternoon"
    if h < 21:
        return "evening"
    return "night"


def caption(photo, mode="auto"):
    """The small line printed on a page. Empty string means print nothing."""
    if mode in ("off", None, False):
        return ""

    place = label(getattr(photo, "place", None))

    if mode == "place":
        # Location only. No GPS means no caption at all - falling back to the
        # date here would put text on every page of an album that was asked to
        # stay clean unless there was somewhere real to name.
        return place

    when = ""
    if photo.taken:
        when = "%s %s" % (photo.taken.strftime("%A %-d %B").strip(),
                          part_of_day(photo.taken))
    if mode == "date":
        return when
    # auto: name the place when we know it, otherwise say when it was taken
    return place or when
