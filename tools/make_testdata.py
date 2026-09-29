#!/usr/bin/env python3
"""Generate a synthetic event folder so the pipeline can be tested end to end.

Creates inbox/_selftest/ with images that exercise every branch: blur, bad
exposure, near-duplicate bursts, mixed orientations, a PNG, a rotated-by-EXIF
frame, a low-resolution frame, and timestamps spread across one day.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from photoalbum.deps import Image, ImageDraw, ImageFilter, HAS_PIEXIF, piexif  # noqa: E402

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "inbox", "_selftest")

PALETTES = [
    [(232, 96, 64), (250, 190, 90), (60, 70, 110)],
    [(40, 110, 160), (150, 220, 230), (250, 250, 240)],
    [(120, 60, 140), (230, 120, 180), (255, 220, 150)],
    [(30, 120, 80), (180, 220, 120), (250, 240, 200)],
    [(190, 40, 70), (255, 170, 140), (70, 30, 60)],
]


def scene(w, h, seed):
    """A deterministic, detail-rich synthetic photo (gradient + shapes + texture)."""
    rnd = random.Random(seed)
    pal = PALETTES[seed % len(PALETTES)]
    img = Image.new("RGB", (w, h), pal[0])
    d = ImageDraw.Draw(img)

    # vertical gradient background
    for y in range(h):
        t = y / max(1, h - 1)
        c = tuple(int(pal[0][i] * (1 - t) + pal[2][i] * t) for i in range(3))
        d.line([(0, y), (w, y)], fill=c)

    # shapes give the Laplacian something real to measure
    for _ in range(18):
        x0 = rnd.randrange(0, w)
        y0 = rnd.randrange(0, h)
        s = rnd.randrange(int(min(w, h) * 0.05), int(min(w, h) * 0.30))
        col = tuple(min(255, max(0, pal[rnd.randrange(3)][i] + rnd.randrange(-40, 40)))
                    for i in range(3))
        if rnd.random() < 0.5:
            d.ellipse([x0, y0, x0 + s, y0 + s], fill=col)
        else:
            d.rectangle([x0, y0, x0 + s, y0 + int(s * 0.7)], fill=col)

    # high-frequency texture so sharp/blurred copies separate cleanly
    for _ in range(int(w * h / 900)):
        x = rnd.randrange(0, w)
        y = rnd.randrange(0, h)
        d.point((x, y), fill=(rnd.randrange(256),) * 3)
    for _ in range(40):
        x0, y0 = rnd.randrange(0, w), rnd.randrange(0, h)
        d.line([x0, y0, x0 + rnd.randrange(-90, 90), y0 + rnd.randrange(-90, 90)],
               fill=(255, 255, 255), width=1)
    return img


def exif_bytes(dt_str, orientation=1):
    if not HAS_PIEXIF:
        return None
    zeroth = {piexif.ImageIFD.Orientation: orientation,
              piexif.ImageIFD.Make: b"SelfTest",
              piexif.ImageIFD.Model: b"Synthetic"}
    exif = {piexif.ExifIFD.DateTimeOriginal: dt_str.encode()}
    return piexif.dump({"0th": zeroth, "Exif": exif, "GPS": {}, "1st": {}, "thumbnail": None})


def save(img, name, dt_str, orientation=1, quality=92):
    path = os.path.join(BASE, name)
    kw = {}
    if name.lower().endswith((".jpg", ".jpeg")):
        kw = {"quality": quality, "subsampling": 0}
        eb = exif_bytes(dt_str, orientation)
        if eb:
            kw["exif"] = eb
    img.save(path, **kw)
    return path


def main():
    os.makedirs(BASE, exist_ok=True)
    for f in os.listdir(BASE):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            os.remove(os.path.join(BASE, f))

    day = "2026:03:14"
    n = 0

    def stamp(h, m, s):
        return "%s %02d:%02d:%02d" % (day, h, m, s)

    # --- 8 good landscape frames, spread across the day -------------------
    for i in range(8):
        img = scene(2400, 1600, 100 + i)
        save(img, "good_land_%02d.jpg" % i, stamp(9 + i, 5 * i % 60, 0))
        n += 1

    # --- 4 good portrait frames -------------------------------------------
    for i in range(4):
        img = scene(1600, 2400, 200 + i)
        save(img, "good_port_%02d.jpg" % i, stamp(13, 10 + i * 7, 0))
        n += 1

    # --- 2 square frames ---------------------------------------------------
    for i in range(2):
        img = scene(1800, 1800, 300 + i)
        save(img, "good_square_%02d.jpg" % i, stamp(15, 20 + i, 0))
        n += 1

    # --- 4 blurred rejects -------------------------------------------------
    for i in range(4):
        img = scene(2400, 1600, 400 + i).filter(ImageFilter.GaussianBlur(6))
        save(img, "blurry_%02d.jpg" % i, stamp(11, 30 + i, 0))
        n += 1

    # --- 2 overexposed, 2 underexposed ------------------------------------
    for i in range(2):
        img = scene(2400, 1600, 500 + i).point(lambda p: min(255, int(p * 2.6)))
        save(img, "overexposed_%02d.jpg" % i, stamp(12, 40 + i, 0))
        n += 1
    for i in range(2):
        img = scene(2400, 1600, 600 + i).point(lambda p: int(p * 0.16))
        save(img, "underexposed_%02d.jpg" % i, stamp(19, 50 + i, 0))
        n += 1

    # --- a burst: 3 near-duplicates 1s apart, one sharper than the others --
    burst = scene(2400, 1600, 700)
    save(burst, "burst_a.jpg", stamp(16, 0, 0))
    save(burst.transform(burst.size, Image.AFFINE, (1, 0, 2, 0, 1, 2)),
         "burst_b.jpg", stamp(16, 0, 1))
    save(burst.transform(burst.size, Image.AFFINE, (1, 0, -3, 0, 1, 1))
         .filter(ImageFilter.GaussianBlur(1.6)), "burst_c.jpg", stamp(16, 0, 2))
    n += 3

    # --- rotated frame: portrait pixels tagged Orientation=6 ---------------
    rot = scene(2400, 1600, 800)
    save(rot, "needs_rotation.jpg", stamp(17, 0, 0), orientation=6)
    n += 1

    # --- low resolution: must be demoted to a small slot -------------------
    save(scene(640, 426, 900), "lowres_small.jpg", stamp(17, 30, 0))
    n += 1

    # --- a PNG (no EXIF at all) -------------------------------------------
    save(scene(2000, 1333, 950), "no_exif_shot.png", stamp(18, 0, 0))
    n += 1

    # --- heavily compressed / noisy ---------------------------------------
    save(scene(2400, 1600, 960), "lowquality_jpeg.jpg", stamp(18, 15, 0), quality=12)
    n += 1

    # --- a clump: 8 distinct good frames inside one minute at 14:00 --------
    # A pure top-N selection would fill the album from here and drop the rest
    # of the day; the segment allocation has to hold this to a couple of slots.
    for i in range(8):
        img = scene(2400, 1600, 1000 + i)
        save(img, "clump_%02d.jpg" % i, stamp(14, 0, i * 7))
        n += 1

    print("wrote %d images to %s" % (n, BASE))


if __name__ == "__main__":
    main()
