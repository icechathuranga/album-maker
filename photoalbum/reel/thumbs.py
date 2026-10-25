"""Cover images to go with a reel.

  <video>.youtube.jpg   1280 x 720, the YouTube thumbnail
  <video>.cover.jpg     1080 x 1920, the cover frame for TikTok, Reels, Shorts

Both use the album's poster cover - full-height strips of the event's
photos, edge to edge, with the title in spaced capitals across them - built
from the photos the reel itself uses, so the thumbnail shows what is in the
video. Text is drawn into the image (a thumbnail has no separate text
layer) at a size that stays readable when YouTube shows it small.
"""
from ..deps import Image
from .. import cover
from . import titles

SIZES = {"youtube": (1280, 720), "cover": (1080, 1920)}
# Strips per image. A vertical frame is narrow, so it takes three: four
# 270-pixel strips crop each photo to a sliver of colour.
STRIPS = {"youtube": 4, "cover": 3}
JPEG_QUALITY = 90          # a YouTube thumbnail must stay under 2 MB


def _video_frames(recipe, sources, cache_dir, limit=30):
    """Still frames from the middle of the reel's video shots, as photos.

    The highlights of a trip are often only on video - the jeep in the mud,
    the elephant - so the thumbnail chooses from these as well. Each frame
    is scored with the album's own analysis; frames are cached on disk.
    """
    import hashlib
    import os
    from concurrent.futures import ThreadPoolExecutor
    from .. import ingest, quality, dedupe
    from . import ffm

    folder = os.path.join(cache_dir, "frames")
    os.makedirs(folder, exist_ok=True)
    wanted = []
    for seg in recipe["segments"]:
        clip = sources.get(seg["src"])
        if seg["kind"] != "clip" or clip is None or not seg.get("speed", 1.0) == 1.0:
            continue
        t = (seg["in"] + seg["out"]) / 2.0
        key = hashlib.sha1(("%s|%d|%.2f" % (clip.rel, int(clip.mtime), t)).encode()).hexdigest()[:16]
        wanted.append((clip, t, os.path.join(folder, key + ".jpg")))
    wanted = wanted[:limit]

    def grab(item):
        """Extract one frame (ffmpeg only - safe to run in parallel)."""
        clip, t, path = item
        if not os.path.exists(path):
            frame = next(ffm.decode_frames(clip.path, t, 0.2, 30,
                                           ffm.fit_size(clip.width, clip.height, 1920),
                                           hdr=clip.hdr), None)
            if frame is None:
                return None
            Image.fromarray(frame).save(path, "JPEG", quality=94)
        return item

    with ThreadPoolExecutor(max_workers=6) as pool:
        got = [x for x in pool.map(grab, wanted) if x is not None]
    # Scored one at a time: the face detector is shared and not thread-safe.
    frames = []
    for clip, t, path in got:
        p = ingest.Photo(path, folder)
        ingest.probe(p)
        p.taken = clip.taken
        quality.analyse(p, print_sharpness="off")
        p.dhash = dedupe.dhash(p)
        frames.append(p)
    # Only frames as sharp as a photo would need to be, and with detail all
    # over: a frame with a large featureless patch (a shirt filling one side,
    # a seat back) gets cropped down to that patch in a narrow strip.
    return [p for p in frames if p.status != "rejected"
            and (p.scores or {}).get("raw_laplacian_var", 0) >= 120
            and _flat_share(p.path) <= MAX_FLAT]


MAX_FLAT = 0.10           # share of the frame that may be featureless


def _flat_share(path):
    """Share of an 8x8 grid of tiles with almost no fine detail."""
    from ..deps import np, cv2
    g = np.asarray(Image.open(path).convert("L").resize((640, 360)), dtype=np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    H, W = g.shape
    n = 8
    v = [lap[i * H // n:(i + 1) * H // n, j * W // n:(j + 1) * W // n].var()
         for i in range(n) for j in range(n)]
    return float(np.mean(np.array(v) < 15.0))


def make(recipe, sources, out_base, title, subtitle, look="natural", cache_dir=None):
    """Write both images next to the video. Returns their paths."""
    used = []
    for seg in recipe["segments"]:
        src = sources.get(seg["src"])
        if seg["kind"] == "photo" and src is not None and src not in used:
            used.append(src)
    frames = _video_frames(recipe, sources, cache_dir) if cache_dir else []
    # The highlights of the video lead: a frame from it counts as one of the
    # best shots (the cover picker favours those). Close-ups of people are
    # left out - in a narrow strip a face filling the frame becomes a torso.
    for f in frames:
        f.hero = True
    used += frames
    # A thumbnail sells the place and the action - the jeep in the mud, the
    # river, the shrine - so shots with nobody in them are used when there
    # are enough; otherwise at least no close-ups (a face filling the frame
    # becomes a torso in a narrow strip).
    scenes = [p for p in used if not (p.scores or {}).get("faces", 0)]
    wide = [p for p in used if (p.scores or {}).get("face_largest", 0.0) <= 0.05]
    used = scenes if len(scenes) >= 8 else (wide if len(wide) >= 8 else used)
    # The strips read left to right in the order things happened.
    used.sort(key=lambda p: (p.taken is None, p.taken or 0, p.name))
    # The opener (when it is a photo) leads; the rest in story order.
    hero = used[0] if used else None
    if hero is None:
        return []
    out = []
    for name, (W, H) in SIZES.items():
        # cover.compose works in millimetres at a dpi; at 25.4 dpi they are pixels.
        # Four wide strips: at thumbnail size, more and narrower strips
        # turn a landscape scene into an unreadable sliver.
        img, text, _style = cover.compose("strips", hero, used, W, H, 25.4, look=look,
                                          strips=STRIPS[name])
        y = 1.0 - text.get("y", 0.6) if text else 0.4
        # A wider, deeper shade behind the whole title block, so the date
        # line stays readable over a white building or sky.
        img = cover._title_band(img, y + 0.04, spread=0.20, strength=0.45, edge=0.0)
        layer = titles.render(title, subtitle, W, H, y=y)
        img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
        path = "%s.%s.jpg" % (out_base, name)
        img.save(path, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)
        out.append(path)
    return out
