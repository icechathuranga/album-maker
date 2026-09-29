"""Settings, resolved from defaults -> album.json -> per-event.json -> CLI flags.

JSON rather than TOML: this box runs Python 3.10, which has no tomllib.
"""
import json
import os

DEFAULTS = {
    # --- album shape ---
    "target_pages": 24,
    "page_tolerance": 0.10,        # land within +/-10% of the page count asked for
    "page_size": "a4",              # a4 a5 a3 letter legal square8 square10 square12 6x4 7x5
    "orientation": "portrait",      # auto | portrait | landscape
    "margin_mm": 16.0,
    "gutter_mm": 2.5,
    "max_photos_per_page": 6,
    "hero_full_page": True,
    "multi_photo_bias": 1.25,        # higher = more photos per page
    "cover_page": True,
    "cover_photo": True,            # use the best photo as the cover image
    # classic | strips | collage | split | duotone | frame
    "cover_style": "strips",
    "smart_crop": True,             # content-aware cropping instead of centre
    "page_numbers": False,
    "bleed_mm": 3.0,               # trim allowance for full-bleed pages
    "bleed_heroes": True,          # let the best photos run off the paper edge
    "bleed_singles": True,         # any lone photo may run to the paper edge
    "bleed_multi_every": 2,        # every Nth multi-photo page runs to the edge
                                   # (1 = all of them, 0 = none)

    # How much of a photo may be cropped to make it fill its cell before the
    # cell instead shrinks to the photo. Higher = fuller pages, less white
    # space, slightly more of each photo trimmed away.
    "fill_crop_budget": 0.52,

    # --- print quality ---
    "target_dpi": 300,
    "min_dpi": 170,
    "jpeg_quality": 92,

    # --- selection ---
    "photos_per_page": 2.4,        # higher = more photos in the album
    "max_per_scene": 2,
    "hero_fraction": 0.16,
    "hero_separation_seconds": 240,  # min gap between two full-page photos
    "scenic_share": 0.28,          # album share reserved for places and things
    "scenic_max": 0.42,            # ...and the most it may take
    "technical_floor": 0.50,
    "sharpness_floor": 0.55,       # reject anything softer than this
    "face_sharpness_floor": 0.50,  # ...measured on the faces when there are any
    # Blur that only shows at full size: strict | normal | off. Soft photos
    # are printed smaller, and ones smeared even at half size are left out.
    "print_sharpness": "normal",
    "min_score": 0.0,
    "detect_faces": True,
    "auto_rotate": True,           # turn sideways photos upright when EXIF is gone

    # --- duplicate handling ---
    "burst_seconds": 12,
    "burst_bits": 12,
    "exact_bits": 4,

    # --- captions ---
    # off | date | place | auto
    #   place = only ever prints a location, and nothing when there is no GPS
    #   auto  = location when known, otherwise the day and part of day
    "captions": "place",

    # --- review output ---
    "slideshow": True,              # write slideshow.html beside the PDF
    "slideshow_dpi": 96,

    # --- look ---
    # auto | natural | vivid | warm | film | mono | soft | crisp | golden
    #   auto = chosen per chapter from its light and subject, one per page
    "look": "auto",

    # --- processing ---
    "enhance": True,
    "workers": 0,                   # 0 = auto (cpu count, capped)

    # --- scoring weights (tune after reading a real report) ---
    "content_mix": 0.62,
    "technical_weights": None,      # null = built-in defaults
    "content_weights": None,
}

CONFIG_NAME = "album.json"

# Spoken-shorthand aliases for page sizes, so "8x8" and "A4" both work.
_SIZE_ALIASES = {
    "8x8": "square8", "10x10": "square10", "12x12": "square12",
    "square": "square10", "sq8": "square8", "sq10": "square10",
    "sq12": "square12", "4x6": "6x4", "5x7": "7x5", "us": "letter",
}

_LOOKS = ("natural", "vivid", "warm", "film", "mono", "soft", "crisp", "golden")
# "auto" in a spec already means orientation, so the automatic look has its own word.
_LOOK_ALIASES = {"autolook": "auto"}
_ORIENTATIONS = ("portrait", "landscape", "auto")
_COVER_STYLES = ("classic", "strips", "collage", "split", "duotone", "frame")

# Density words, for "A4 landscape 35 pages tight".
_DENSITY = {
    "tight":  {"margin_mm": 10.0, "gutter_mm": 2.0, "fill_crop_budget": 0.58},
    "airy":   {"margin_mm": 22.0, "gutter_mm": 4.0, "fill_crop_budget": 0.34},
}


def parse_spec(text, page_sizes=None):
    """Read a plain-language album spec such as "A4 landscape 35pages".

    Order does not matter and the words can be joined or separate, because
    nobody wants to remember flag names to print a photo album. Anything not
    recognised is returned so the caller can complain about it rather than
    silently ignoring it.

    Returns (overrides, unknown_tokens).
    """
    import re

    sizes = set(page_sizes or ())
    out, unknown = {}, []
    if not text:
        return out, unknown

    # "35pages" -> "35 pages"; strip commas and stray punctuation
    cleaned = re.sub(r"(\d)\s*(pages?|pp?)\b", r"\1 pages", text.lower())
    cleaned = cleaned.replace(",", " ")

    for token in cleaned.split():
        token = token.strip()
        if not token or token in ("pages", "page", "p", "pp", "in", "on", "with"):
            continue

        alias = _SIZE_ALIASES.get(token, token)
        if alias in sizes:
            out["page_size"] = alias
        elif token in _ORIENTATIONS:
            out["orientation"] = token
        elif token in _LOOKS:
            out["look"] = token
        elif token in _LOOK_ALIASES:
            out["look"] = _LOOK_ALIASES[token]
        elif token in _COVER_STYLES:
            out["cover_style"] = token
        elif token in _DENSITY:
            out.update(_DENSITY[token])
        elif re.fullmatch(r"\d{1,3}", token):
            out["target_pages"] = int(token)
        elif re.fullmatch(r"\d{2,3}dpi", token):
            out["target_dpi"] = int(token[:-3])
        else:
            unknown.append(token)

    return out, unknown


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}
    except ValueError as exc:
        raise SystemExit("%s is not valid JSON: %s" % (path, exc))


def load(project_root, event_folder=None, overrides=None):
    """Merge every layer of configuration into one dict.

    An event folder may carry its own album.json - handy when one trip wants
    square pages and everything else stays A4.
    """
    cfg = dict(DEFAULTS)
    cfg.update(_load_json(os.path.join(project_root, CONFIG_NAME)))
    if event_folder:
        cfg.update(_load_json(os.path.join(event_folder, CONFIG_NAME)))
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def write_default(path):
    """Write a commented-by-example config the user can edit."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(DEFAULTS, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def read_list_file(folder, name):
    """Read keep.txt / skip.txt: one filename per line, # for comments."""
    return set(read_list_ordered(folder, name))


def read_list_ordered(folder, name):
    """The same, in the order written - cover.txt places photos by it."""
    path = os.path.join(folder, name)
    out = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                name_ = os.path.basename(line).lower()
                if line and name_ not in out:
                    out.append(name_)
    except FileNotFoundError:
        pass
    return out
