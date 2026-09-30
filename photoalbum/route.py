"""The route page: where the trip went, drawn from the photos' GPS.

Travel books open each leg with a map. This draws one without map tiles or
any network: the stops are the chapters of the trip that carry GPS, placed
by a flat projection, joined by a line and named from the bundled city list.
The same folder always draws the same map.
"""
import math

from . import places

MIN_STOPS = 3        # fewer than this is not a route, just a place
MERGE_KM = 5.0       # consecutive chapters closer than this are one stop


def _km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(h)))


def stops(photos):
    """The trip's stops in order: [{"lat", "lon", "label", "when"}] or [].

    One stop per chapter with GPS, at the median of its photos' positions,
    merged with the previous stop when it is the same town. Returns an empty
    list when there are too few distinct stops to call it a route.
    """
    chapters = []
    for p in photos:
        if not getattr(p, "gps", None):
            continue
        if not chapters or chapters[-1][0] != p.segment:
            chapters.append((p.segment, []))
        chapters[-1][1].append(p)

    out = []
    for _seg, group in chapters:
        lats = sorted(p.gps[0] for p in group)
        lons = sorted(p.gps[1] for p in group)
        pos = (lats[len(lats) // 2], lons[len(lons) // 2])
        if out and _km((out[-1]["lat"], out[-1]["lon"]), pos) < MERGE_KM:
            continue
        named = places.nearest(*pos) if places.available() else None
        first = min((p for p in group if p.taken), key=lambda p: p.taken, default=None)
        out.append({"lat": pos[0], "lon": pos[1],
                    "label": named[0] if named else "",
                    "when": first.taken.strftime("%d %b") if first else ""})
    # Two stops with the same name in a row are one visit.
    merged = []
    for s in out:
        if merged and s["label"] and s["label"] == merged[-1]["label"]:
            continue
        merged.append(s)
    return merged if len(merged) >= MIN_STOPS else []


def project(stop_list, box):
    """Place stops inside box (x0, y0, x1, y1), north up, true proportions.

    Equirectangular with longitude scaled by the cosine of the middle
    latitude - accurate enough over the span of one trip.
    """
    x0, y0, x1, y1 = box
    mid = sum(s["lat"] for s in stop_list) / len(stop_list)
    k = math.cos(math.radians(mid))
    xs = [s["lon"] * k for s in stop_list]
    ys = [s["lat"] for s in stop_list]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    scale = min((x1 - x0) / span, (y1 - y0) / span)
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return [(mx + (x - cx) * scale, my + (y - cy) * scale) for x, y in zip(xs, ys)]
