"""The album's looks, baked into 3D colour tables ffmpeg can apply to video.

A look is mostly a per-pixel mapping - tone curve, vibrance, saturation,
warmth - and any per-pixel mapping is exactly a 3D lookup table. So each
look is run once over a lattice of every colour and written as a .cube file
for ffmpeg's lut3d filter: video and album then share one colour science.

Two parts of a look are not per-pixel and are handled differently:
  clarity     a wide local-contrast pass; on video it shimmers from frame
              to frame, so it is left out.
  mono clip   black and white points measured per photo; on video a fixed
              stretch stands in, so frames do not pump.
"""
import os

from ..deps import np, Image, ImageEnhance
from .. import enhance

SIZE = 33
MONO_STRETCH = (10, 246)     # fixed black and white points for mono


def _lattice():
    r = np.linspace(0, 255, SIZE)
    b, g, rr = np.meshgrid(r, r, r, indexing="ij")    # red varies fastest
    rgb = np.stack([rr, g, b], axis=-1).reshape(SIZE * SIZE, SIZE, 3)
    return Image.fromarray(np.clip(np.round(rgb), 0, 255).astype(np.uint8))


def build(look, out_dir):
    """Write <look>.cube into out_dir (once) and return its path."""
    path = os.path.join(out_dir, "%s.cube" % look)
    if os.path.exists(path):
        return path
    spec = enhance.LOOKS.get(look, {})
    img = _lattice()
    if spec.get("clip"):
        lo, hi = MONO_STRETCH
        a = (np.asarray(img, dtype=np.float32) - lo) * (255.0 / (hi - lo))
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if spec.get("scurve", 1.0) > 1.0 or spec.get("lift"):
        img, _ = enhance._scurve(img, spec.get("scurve", 1.0), spec.get("lift", 0.0),
                                 spec.get("rolloff", 0.0))
    if spec.get("vibrance"):
        img, _ = enhance._vibrance(img, spec["vibrance"])
    sat = spec.get("saturation", 1.0)
    if abs(sat - 1.0) > 0.01:
        img = ImageEnhance.Color(img).enhance(sat)
    if spec.get("warmth"):
        img, _ = enhance._warmth(img, spec["warmth"])

    a = np.asarray(img, dtype=np.float32).reshape(-1, 3) / 255.0
    os.makedirs(out_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write("TITLE \"%s\"\nLUT_3D_SIZE %d\n" % (look, SIZE))
        for r, g, b in a:
            fh.write("%.5f %.5f %.5f\n" % (r, g, b))
    os.replace(tmp, path)
    return path
