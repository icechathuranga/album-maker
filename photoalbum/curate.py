"""Choose which photos make the album - the editing pass, not the scoring pass.

A photographer with 400 frames does not print the top 40 by score. They cover
the whole event: arrivals, the speeches, the cake, the dancing, the goodbyes.
If the best-lit half hour was the cake cutting, a pure top-N selection returns
twenty cake photos and nothing else.

So selection runs in three steps:
  1. split the day into segments wherever there is a real gap in shooting
  2. give every segment a share of the page budget, so all of them appear
  3. inside a segment, take the highest-content frames, capping how many come
     from any one visually-similar scene
"""
import math

from .deps import np
from . import dedupe, ingest

# How many photos an average page holds once the layout mixes full-page
# heroes with 2-up and 4-up spreads.
PHOTOS_PER_PAGE = 2.4

# A gap this much larger than the typical gap means the event moved on.
GAP_MULTIPLIER = 3.0
MIN_GAP_SECONDS = 240.0     # never split on less than 4 minutes
MAX_SEGMENTS = 14           # beyond this the "chapters" stop meaning anything

SCENE_BITS = 18             # looser than burst detection: "same scene, same place"
MAX_PER_SCENE = 2

HERO_FRACTION = 0.16        # top slice get a full page to themselves

# Two photographs taken moments apart should not each command a full page, even
# when they are different enough to survive burst detection. Consecutive pages
# of near-identical selfies read as padding; the same frames sharing one page
# read as a moment.
HERO_SEPARATION_SECONDS = 240

# Share of the album kept for strong photographs with nobody in them - the
# temple, the street, the view from the hotel, the meal, the thing you stopped
# to photograph. Face-led ranking will otherwise fill every page with people
# and the album stops saying anything about where you actually went.
SCENIC_SHARE = 0.28
# ...and no more than this. People and places cannot be compared on one number
# - whichever category happens to score higher on a given folder would take the
# whole album. Judging them on merit WITHIN a reserved band gives a mixed book
# on any folder, which is what an album of a trip should be.
SCENIC_MAX = 0.42

# A quiet stretch of the event does not earn a page if nothing shot during it
# is worth printing. A photo must clear both floors to be selected at all.
ABSOLUTE_FLOOR = 25.0       # score out of 100
RELATIVE_FLOOR = 0.45       # ...and this fraction of the folder's best score

# Separately from content ranking, a visibly soft or badly exposed frame is not
# printable. Tune this in album.json after seeing the first real run's report.
TECHNICAL_FLOOR = 0.50

# Softness is rejected outright rather than scored down. A blurred picture
# looks worse enlarged on paper than it does on a phone, and no amount of
# interesting content rescues it - so these are gates, not weights.
SHARPNESS_FLOOR = 0.55
FACE_SHARPNESS_FLOOR = 0.50


def _eligible(photos):
    return [p for p in photos if p.status == "pending"]


def segment(photos, gap_multiplier=GAP_MULTIPLIER, min_gap=MIN_GAP_SECONDS,
            max_segments=MAX_SEGMENTS):
    """Split a chronological run of photos into 'chapters' of the event.

    Uses the shape of the shooting itself rather than a fixed clock interval:
    a wedding and a two-hour birthday party have very different natural gaps.
    """
    if not photos:
        return 0

    # Photos with no established capture time cannot take part in a timeline;
    # they become one final chapter of their own.
    dated = [p for p in photos if p.taken_src != ingest.UNKNOWN_TIME and p.taken]
    undated = [p for p in photos if p not in dated]

    if not dated:
        for p in photos:
            p.segment = 0
        return 1

    ordered = sorted(dated, key=lambda p: (p.taken, p.seq))
    if len(ordered) < 3:
        for p in ordered:
            p.segment = 0
        for p in undated:
            p.segment = 1 if undated else 0
        return 2 if undated else 1

    gaps = []
    for a, b in zip(ordered, ordered[1:]):
        gaps.append(max(0.0, (b.taken - a.taken).total_seconds()))
    positive = [g for g in gaps if g > 0]
    median_gap = float(np.median(positive)) if positive else 0.0
    threshold = max(min_gap, median_gap * gap_multiplier)

    # If that would shatter the event, raise the bar until it does not.
    while True:
        breaks = [i for i, g in enumerate(gaps) if g >= threshold]
        if len(breaks) + 1 <= max_segments or threshold > 86400:
            break
        threshold *= 1.6

    seg = 0
    ordered[0].segment = 0
    for i, p in enumerate(ordered[1:]):
        if gaps[i] >= threshold:
            seg += 1
        p.segment = seg

    if undated:
        seg += 1
        for p in undated:
            p.segment = seg
    return seg + 1


def _scene_cap(candidates, max_per_scene=MAX_PER_SCENE, bits=SCENE_BITS):
    """Keep at most N photos that look like the same scene, best first."""
    kept, scenes = [], []
    for p in candidates:
        if p.dhash is None:
            kept.append(p)
            continue
        matched = False
        for scene in scenes:
            if dedupe.hamming(p.dhash, scene["hash"]) <= bits:
                if scene["count"] < max_per_scene:
                    scene["count"] += 1
                    kept.append(p)
                else:
                    p.reasons.append("already %d frames from this scene" % max_per_scene)
                matched = True
                break
        if not matched:
            scenes.append({"hash": p.dhash, "count": 1})
            kept.append(p)
    return kept


def _allocate(counts, budget):
    """Share the budget across segments, weighted by sqrt of their size.

    Square root rather than linear: a segment with 200 frames deserves more
    pages than one with 20, but not ten times more - the album should still
    feel like it covers the whole day.
    """
    n = len(counts)
    if n == 0:
        return []
    if budget <= n:
        # Not enough budget for one each - favour the busiest segments.
        order = sorted(range(n), key=lambda i: -counts[i])
        alloc = [0] * n
        for i in order[:budget]:
            alloc[i] = 1
        return alloc

    weights = [math.sqrt(max(c, 1)) for c in counts]
    total_w = sum(weights)
    alloc = [1] * n                      # every chapter appears at least once
    remaining = budget - n
    exact = [w / total_w * remaining for w in weights]
    base = [int(math.floor(e)) for e in exact]

    for i in range(n):
        alloc[i] += base[i]

    # hand out the rounding remainder to the largest fractional parts
    left = remaining - sum(base)
    frac = sorted(range(n), key=lambda i: -(exact[i] - base[i]))
    for i in frac[:max(0, left)]:
        alloc[i] += 1

    # never allocate a segment more photos than it actually has
    for i in range(n):
        alloc[i] = min(alloc[i], counts[i])
    return alloc


def _too_soft(p, sharpness_floor, face_sharpness_floor):
    """True when the frame is too soft to print, whatever else it has going."""
    s = p.scores or {}
    if s.get("blurred_at_print"):
        # Measured on the real pixels: smeared even at half size, so no slot
        # in the album is small enough to hide it.
        return "blurred at print size"
    face_sharp = s.get("face_sharpness", -1.0)
    if face_sharp >= 0.0:
        # People in shot: their faces are what has to be sharp.
        if face_sharp < face_sharpness_floor:
            return "faces are not sharp enough to print"
    if s.get("sharpness", 1.0) < sharpness_floor:
        return "not sharp enough to print"
    return None


def select(photos, target_pages=24, photos_per_page=PHOTOS_PER_PAGE,
           max_per_scene=MAX_PER_SCENE, hero_fraction=HERO_FRACTION,
           min_score=0.0, technical_floor=TECHNICAL_FLOOR,
           sharpness_floor=SHARPNESS_FLOOR,
           face_sharpness_floor=FACE_SHARPNESS_FLOOR,
           scenic_share=SCENIC_SHARE, scenic_max=SCENIC_MAX,
           hero_separation_seconds=HERO_SEPARATION_SECONDS):
    """Pick the album's photos. Returns the selected list, chronological.

    Sub-standard frames are never used as padding: if the folder cannot fill
    the target page count with photos worth printing, the album comes out
    shorter instead of worse.
    """
    # Softness is decided before anything else, so a blurred frame can never be
    # pulled back in to fill a quiet stretch of the day.
    for p in _eligible(photos):
        why = _too_soft(p, sharpness_floor, face_sharpness_floor)
        if why:
            p.status = "rejected"
            p.reasons.append(why)

    pool = _eligible(photos)
    if min_score > 0:
        for p in pool:
            if p.score < min_score:
                p.status = "rejected"
                p.reasons.append("below minimum score %.0f" % min_score)
        pool = _eligible(photos)

    if not pool:
        return []

    # The bar every photo must clear, however thin its part of the day is.
    best = max(p.score for p in pool)
    floor = max(ABSOLUTE_FLOOR, best * RELATIVE_FLOOR)
    worthy = [p for p in pool
              if p.score >= floor and p.technical >= technical_floor]
    if not worthy:
        # Nothing clears the bar. Rather than produce an empty album, fall back
        # to the best few and let the report explain why they are marginal.
        worthy = sorted(pool, key=lambda p: -p.score)[:max(1, target_pages)]
        for p in worthy:
            p.reasons.append("used despite being below the quality bar - "
                             "no better frames in this folder")

    n_segments = segment(worthy)
    budget = max(1, int(round(target_pages * photos_per_page)))

    by_segment = {}
    for p in worthy:
        by_segment.setdefault(p.segment, []).append(p)

    # --- pass one: coverage ------------------------------------------------
    # Every chapter of the day contributes its single best photo, so the
    # arrivals and the goodbyes are both in the book even if the middle of the
    # afternoon produced far better pictures.
    selected = []
    scenes = {}
    for seg_id in sorted(by_segment):
        members = sorted(by_segment[seg_id], key=lambda p: (-p.score, p.seq))
        best = members[0]
        selected.append(best)
        scenes.setdefault(seg_id, []).append({"hash": best.dhash, "count": 1})
        best.reasons.append("best photo of this part of the day")

    # --- pass one and a quarter: one from every scene ----------------------
    # If the camera came back to something repeatedly, that thing mattered.
    # Taking the best frame of each distinct scene means nothing photographed
    # on the trip is missing from the book altogether - which is the point of
    # an album of memories, as opposed to a list of the highest scores.
    picked = set(id(p) for p in selected)
    scene_best = {}
    for p in sorted(worthy, key=lambda p: (-p.score, p.seq)):
        if id(p) in picked or p.dhash is None:
            continue
        key = None
        for existing in scene_best:
            if dedupe.hamming(p.dhash, existing) <= SCENE_BITS:
                key = existing
                break
        if key is None:
            scene_best[p.dhash] = p
    # The scenery cap applies here too. Places tend to be photographed from
    # several angles, so "one per scene" alone would quietly hand most of the
    # album to landscapes and leave the people out.
    cap = int(round(budget * max(0.0, min(1.0, scenic_max))))
    n_scenic = sum(1 for p in selected if not (p.scores or {}).get("faces", 0))
    for p in sorted(scene_best.values(), key=lambda p: (-p.score, p.seq)):
        if len(selected) >= budget:
            break
        if not (p.scores or {}).get("faces", 0):
            if n_scenic >= cap:
                continue
            n_scenic += 1
        p.reasons.append("the best frame of this scene")
        selected.append(p)
        picked.add(id(p))

    # --- pass one and a half: a place for the scenery ----------------------
    # Reserved before the open contest, because on face-weighted scores a
    # landmark almost never outranks a close-up of somebody laughing - and an
    # album with no establishing shots reads as a pile of selfies.
    picked = set(id(p) for p in selected)
    want_scenic = int(round(budget * max(0.0, scenic_share)))
    if want_scenic > 0:
        scenic = [p for p in worthy
                  if id(p) not in picked and not (p.scores or {}).get("faces", 0)]
        scenic.sort(key=lambda p: (-p.score, p.seq))
        added = 0
        for p in scenic:
            if added >= want_scenic or len(selected) >= budget:
                break
            seg_scenes = scenes.setdefault(p.segment, [])
            match = next((sc for sc in seg_scenes
                          if sc["hash"] is not None and p.dhash is not None
                          and dedupe.hamming(p.dhash, sc["hash"]) <= SCENE_BITS), None)
            if match is not None:
                if match["count"] >= max_per_scene:
                    continue
                match["count"] += 1
            else:
                seg_scenes.append({"hash": p.dhash, "count": 1})
            p.reasons.append("kept as an establishing shot of the place")
            selected.append(p)
            added += 1

    # --- pass two: quality -------------------------------------------------
    # Everything else is decided on merit across the whole event. Sharing the
    # budget out per segment instead starves the hour that actually produced
    # the good photographs - on a real folder that left 51 better frames on
    # the bench while weaker ones were printed to make up a segment's quota.
    picked = set(id(p) for p in selected)
    bench = sorted((p for p in worthy if id(p) not in picked),
                   key=lambda p: (-p.score, p.seq))

    scenic_cap = int(round(budget * max(0.0, min(1.0, scenic_max))))
    scenic_count = sum(1 for p in selected if not (p.scores or {}).get("faces", 0))

    for p in bench:
        if len(selected) >= budget:
            break
        # Keep the two kinds of photograph in proportion.
        is_scenic = not (p.scores or {}).get("faces", 0)
        if is_scenic and scenic_count >= scenic_cap:
            p.reasons.append("the album already has its share of scenery")
            continue
        # The scene cap still applies, so one photogenic spot cannot take over.
        seg_scenes = scenes.setdefault(p.segment, [])
        matched = None
        if p.dhash is not None:
            for sc in seg_scenes:
                if sc["hash"] is not None and \
                        dedupe.hamming(p.dhash, sc["hash"]) <= SCENE_BITS:
                    matched = sc
                    break
        if matched is not None:
            if matched["count"] >= max_per_scene:
                p.reasons.append("already %d frames from this scene" % max_per_scene)
                continue
            matched["count"] += 1
        else:
            seg_scenes.append({"hash": p.dhash, "count": 1})
        selected.append(p)
        if is_scenic:
            scenic_count += 1

    for p in pool:
        if p.status != "pending":
            continue
        if p.technical < technical_floor:
            p.reasons.append("too soft or badly exposed to print (quality %.0f%%)"
                             % (p.technical * 100))
        elif p.score < floor:
            p.reasons.append("below the album's quality bar (%.0f of 100)" % floor)

    chosen = set(id(p) for p in selected)
    for p in pool:
        if id(p) in chosen:
            p.status = "selected"
        else:
            p.status = "not-selected"
            if not p.reasons:
                p.reasons.append("ranked below the cut for this album")

    # Heroes get a page of their own in the layout stage. Two near-identical
    # frames must never both get one: side by side as full-page spreads, the
    # resemblance is glaring in a way it never is at thumbnail size.
    ranked = sorted(selected, key=lambda p: -p.score)
    n_heroes = max(1, int(round(len(ranked) * hero_fraction)))
    chosen_heroes = []
    for p in ranked:
        if len(chosen_heroes) >= n_heroes:
            break
        twin = any(
            p.dhash is not None and h.dhash is not None
            and dedupe.hamming(p.dhash, h.dhash) <= SCENE_BITS
            for h in chosen_heroes)
        # Timing catches what the visual hash misses: two selfies seconds apart
        # can be far apart as hashes because the camera swung, yet they are
        # plainly the same moment and belong together on one page.
        near_in_time = any(
            p.taken and h.taken
            and abs((p.taken - h.taken).total_seconds()) <= hero_separation_seconds
            for h in chosen_heroes)
        if twin or near_in_time:
            p.reasons.append("printed smaller - another frame from this moment "
                             "already has a full page")
            continue
        if (p.scores or {}).get("sharp_scale", 1.0) < 0.5:
            # A full page is exactly the size at which its softness shows.
            p.reasons.append("printed smaller - too soft to fill a page")
            continue
        p.hero = True
        chosen_heroes.append(p)

    selected.sort(key=lambda p: (p.taken, p.seq))
    return selected


def summary(photos, selected, n_segments):
    """Counts for the CLI and the report."""
    def n(status):
        return sum(1 for p in photos if p.status == status)
    return {
        "total": len(photos),
        "selected": len(selected),
        "heroes": sum(1 for p in selected if p.hero),
        "duplicates": n("duplicate"),
        "rejected": n("rejected"),
        "not_selected": n("not-selected"),
        "unreadable": n("unreadable"),
        "segments": n_segments,
    }
