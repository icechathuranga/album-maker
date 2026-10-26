"""Gather an event's clips and photos, scored, for the editor.

Photos reuse the album's verdicts when the album has been made: its
album.json already says which photos are worth printing and why, so a reel
of a 556-photo trip does not re-score everything. Without an album, photos
are scored here with the album's own code and cached.
"""
import hashlib
import json
import os

from .. import config, curate, dedupe, ingest, quality
from . import analyse, probe


def _photo_cache(cache_dir, p):
    key = "photo|%s|%d|%d" % (p.rel, p.bytes, int(os.path.getmtime(p.path)))
    return os.path.join(cache_dir, hashlib.sha1(key.encode()).hexdigest() + ".json")


REEL_MIN_DETAIL = curate.MIN_DETAIL   # the album's motion-blur bar


def photos(event_dir, album_json, cache_dir, skip=(), keep=()):
    """Photos fit for a reel, with .score, .scores and .dhash set."""
    found, _skipped = ingest.scan(event_dir)
    for p in found:
        ingest.probe(p)
    found = [p for p in found if p.status != "unreadable"]
    ingest.reconcile_times(found)
    ingest.order(found)

    verdicts = {}
    if album_json and os.path.exists(album_json):
        with open(album_json) as fh:
            verdicts = {d["name"]: d for d in json.load(fh).get("photos", [])}

    out = []
    for p in found:
        low = p.name.lower()
        if low in skip:
            continue
        d = verdicts.get(p.name)
        if d is not None:
            if d["status"] != "selected" and low not in keep:
                continue
            p.score = float(d["score"])
            p.scores = dict(d.get("metrics", {}))
            p.status = "selected"
            out.append(p)
            continue
        path = _photo_cache(cache_dir, p)
        if os.path.exists(path):
            with open(path) as fh:
                c = json.load(fh)
            p.score, p.technical, p.scores = c["score"], c["technical"], c["scores"]
            p.face_boxes = [tuple(b) for b in c["face_boxes"]]
            p.dhash, p.status = c["dhash"], c["status"]
        else:
            quality.analyse(p)
            p.dhash = dedupe.dhash(p)
            os.makedirs(cache_dir, exist_ok=True)
            with open(path, "w") as fh:
                json.dump({"score": p.score, "technical": p.technical,
                           "scores": p.scores, "face_boxes": p.face_boxes,
                           "dhash": p.dhash, "status": p.status}, fh)
        out.append(p)

    if not verdicts:
        dedupe.group(out)
        out = [p for p in out if p.status not in ("rejected", "duplicate", "unreadable")
               or p.name.lower() in keep]
    # Blurred photos never go in. Without an album the album's own softness
    # rules have not run, so apply them here; and a reel needs a little more:
    # a landscape photo shown vertically uses only a third of its width, so
    # one that is sharp only at half size or less goes soft on the screen.
    sharp = []
    for p in out:
        if p.name.lower() in keep:
            sharp.append(p)
            continue
        why = curate._too_soft(p, curate.SHARPNESS_FLOOR, curate.FACE_SHARPNESS_FLOOR)
        s = p.scores or {}
        # Motion blur smears the whole frame, so even its most detailed parts
        # carry little fine detail. On this trip's 293 photos every blurred
        # one measured under 90 and every good one 140 or more.
        if not why and s.get("raw_laplacian_var", 1e9) < REEL_MIN_DETAIL:
            why = "blurred (too little fine detail anywhere)"
        if not why and (s.get("sharp_scale", 1.0) <= 0.25 or
                        (p.width > p.height and s.get("sharp_scale", 1.0) < 0.5)):
            why = "too soft to fill a phone screen"
        if why:
            p.status = "rejected"
            p.reasons.append(why)
        else:
            sharp.append(p)
    return sharp


def clips(event_dir, cache_dir, progress=None, skip=(), dropped=None):
    """Probed and analysed clips, cached; unreadable or rejected ones dropped
    (and listed in `dropped` as (name, reason) when a list is given)."""
    found = [c for c in probe.scan(event_dir) if c.name.lower() not in skip]
    for c in found:
        probe.probe(c)
    analyse.analyse_clips([c for c in found if c.status == "pending"], cache_dir,
                          progress=progress)
    out = []
    for c in found:
        if c.status == "pending" and c.timeline:
            out.append(c)
        elif dropped is not None:
            dropped.append((c.name, "; ".join(c.reasons) or c.status))
    return out


def chapters(clips_, photos_):
    """Split clips and photos into chapters of the event together."""
    items = list(clips_) + list(photos_)
    for i, it in enumerate(sorted(items, key=lambda x: (x.taken is None, x.taken or 0, x.name))):
        it.seq = i
    return curate.segment(items) if items else 0


def read_lists(event_dir):
    return (config.read_list_file(event_dir, "keep.txt"),
            config.read_list_file(event_dir, "skip.txt"))
