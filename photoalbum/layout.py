"""Turn a selected photo list into printable pages.

The look this aims at is the modern printed wedding/travel album rather than a
contact sheet: photographs that run off the edge of the paper, pairs split on
the golden ratio instead of down the middle, single pictures given room to
breathe, and a page rhythm that varies as the book goes on.

Three ideas do most of the work:

  adaptive slots   For a single photo the slot is built to *the photo's own
                   shape*, so a tall frame gets a tall slot and nothing is
                   cropped away to satisfy the grid.

  full bleed       Heroes run past all four paper edges. Nothing signals
                   "printed album" more than an image with no white border,
                   and nothing signals "office document" more than everything
                   sitting inside the same rectangle.

  weighted grids   Multi-photo pages still use grids, but the columns and rows
                   are unequal - roughly 1.6 : 1 - so pages look composed
                   rather than tabulated.

A slot whose shape is too far from its photo is refused outright, which is what
stops the letterboxing and the half-cropped faces the plain grid produced.
"""
import math


def _hamming(a, b):
    """Bit distance between two perceptual hashes."""
    return bin(a ^ b).count("1")

# ---------------------------------------------------------------------------
# Paper
# ---------------------------------------------------------------------------
PAGE_SIZES_MM = {
    "a4": (210.0, 297.0),
    "a5": (148.0, 210.0),
    "a3": (297.0, 420.0),
    "letter": (215.9, 279.4),
    "legal": (215.9, 355.6),
    "square8": (203.2, 203.2),      # 8 x 8 in
    "square10": (254.0, 254.0),     # 10 x 10 in
    "square12": (304.8, 304.8),     # 12 x 12 in
    "6x4": (152.4, 101.6),
    "7x5": (177.8, 127.0),
}

TARGET_DPI = 300.0      # what a print shop wants
MIN_DPI = 170.0         # below this a photo is visibly soft on paper

# Beyond this, a slot is the wrong shape for the photo and must not be used.
MAX_CROP_LOSS = 0.52

# Full bleed is stricter. A landscape photo run to the edges of a portrait page
# loses nearly half its width, which on a group shot means losing a person - so
# a lone photo whose shape is far from the paper's gets an adaptive slot that
# keeps the whole frame instead.
BLEED_MAX_CROP_LOSS = 0.30

# How much of a photo may be cropped away to help it fill its cell. Past this
# the SLOT shrinks to the photo's shape instead of the photo being cut - which
# is what lets tall 9:16 phone frames share a page with 4:3 ones at all.
#
# Set generously on purpose. In a printed album the photographs in a cluster
# butt up against each other and read as one block; ragged gaps where a slot
# shrank to fit its picture are the thing that looks amateur. Content-aware
# cropping protects the faces, so trimming the picture is the better trade.
FILL_CROP_BUDGET = 0.52

PHI = 1.618             # the split used for asymmetric pages

# ---------------------------------------------------------------------------
# Templates
#
#   cells     (col, row, colspan, rowspan) on a cols x rows grid
#   col_w     relative column widths (absent = equal)
#   row_h     relative row heights   (absent = equal)
#   style     "bleed" runs off the paper; otherwise a normal margined page
# ---------------------------------------------------------------------------
TEMPLATES = {
    # --- one photo -------------------------------------------------------
    "bleed_full": {"cols": 1, "rows": 1, "cells": [(0, 0, 1, 1)], "style": "bleed"},
    "full":       {"cols": 1, "rows": 1, "cells": [(0, 0, 1, 1)]},

    # --- two photos ------------------------------------------------------
    "pair_wide_l": {"cols": 2, "rows": 1, "cells": [(0, 0, 1, 1), (1, 0, 1, 1)],
                    "col_w": [PHI, 1.0]},
    "pair_wide_r": {"cols": 2, "rows": 1, "cells": [(0, 0, 1, 1), (1, 0, 1, 1)],
                    "col_w": [1.0, PHI]},
    "pair_tall_t": {"cols": 1, "rows": 2, "cells": [(0, 0, 1, 1), (0, 1, 1, 1)],
                    "row_h": [PHI, 1.0]},
    "pair_tall_b": {"cols": 1, "rows": 2, "cells": [(0, 0, 1, 1), (0, 1, 1, 1)],
                    "row_h": [1.0, PHI]},
    "two_h":       {"cols": 2, "rows": 1, "cells": [(0, 0, 1, 1), (1, 0, 1, 1)]},
    "two_v":       {"cols": 1, "rows": 2, "cells": [(0, 0, 1, 1), (0, 1, 1, 1)]},

    # --- three photos ----------------------------------------------------
    "big_left":   {"cols": 2, "rows": 2,
                   "cells": [(0, 0, 1, 2), (1, 0, 1, 1), (1, 1, 1, 1)],
                   "col_w": [PHI, 1.0]},
    "big_right":  {"cols": 2, "rows": 2,
                   "cells": [(1, 0, 1, 2), (0, 0, 1, 1), (0, 1, 1, 1)],
                   "col_w": [1.0, PHI]},
    "big_top":    {"cols": 2, "rows": 2,
                   "cells": [(0, 0, 2, 1), (0, 1, 1, 1), (1, 1, 1, 1)],
                   "row_h": [PHI, 1.0]},
    "big_bottom": {"cols": 2, "rows": 2,
                   "cells": [(0, 1, 2, 1), (0, 0, 1, 1), (1, 0, 1, 1)],
                   "row_h": [1.0, PHI]},
    "three_row":  {"cols": 3, "rows": 1,
                   "cells": [(0, 0, 1, 1), (1, 0, 1, 1), (2, 0, 1, 1)]},
    "three_col":  {"cols": 1, "rows": 3,
                   "cells": [(0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 1, 1)]},

    # --- overlap: one photo filling the page, a second laid over it -------
    # The collage move from printed wedding albums. The inset needs a white
    # border to separate it from whatever is behind, or it reads as a glitch.
    # The inset's corner is not fixed here - _place_inset moves it to whichever
    # corner covers the least of the photograph underneath.
    "inset_br":  {"style": "overlap", "rects": [
        (0.0, 0.0, 1.0, 1.0, {"bleed": True}),
        (0.570, 0.585, 0.360, 0.340, {"frame": 2.2, "shadow": False})]},


    # --- four photos -----------------------------------------------------
    "four_left":  {"cols": 2, "rows": 3,
                   "cells": [(0, 0, 1, 3), (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 1, 1)],
                   "col_w": [PHI, 1.0]},
    "four_grid":  {"cols": 2, "rows": 2,
                   "cells": [(0, 0, 1, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1)]},
    "four_right": {"cols": 2, "rows": 3,
                   "cells": [(1, 0, 1, 3), (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 1, 1)],
                   "col_w": [1.0, PHI]},
    "four_band":  {"cols": 3, "rows": 2,
                   "cells": [(0, 0, 3, 1), (0, 1, 1, 1), (1, 1, 1, 1), (2, 1, 1, 1)],
                   "row_h": [PHI, 1.0]},

    # --- five and six: the dense mosaic pages of a printed album ----------
    "six_grid":    {"cols": 3, "rows": 2,
                    "cells": [(0, 0, 1, 1), (1, 0, 1, 1), (2, 0, 1, 1),
                              (0, 1, 1, 1), (1, 1, 1, 1), (2, 1, 1, 1)]},
    "six_feature": {"cols": 3, "rows": 3,
                    "cells": [(0, 0, 2, 2), (2, 0, 1, 1), (2, 1, 1, 1),
                              (0, 2, 1, 1), (1, 2, 1, 1), (2, 2, 1, 1)],
                    "col_w": [PHI, PHI, 1.0], "row_h": [PHI, PHI, 1.0]},
}

# Candidate templates per photo count, best first. Asymmetric options lead so
# an even split is only reached for when nothing more interesting fits.
BY_COUNT = {
    2: ["pair_wide_l", "pair_wide_r", "pair_tall_t", "pair_tall_b", "two_h",
        "two_v", "inset_br"],
    3: ["big_left", "big_right", "big_top", "big_bottom", "three_row", "three_col"],
    4: ["four_left", "four_right", "four_band", "four_grid"],
    6: ["six_feature", "six_grid"],
}


class Slot:
    """One rectangle on a page, in millimetres from the top-left corner.

    A slot may sit outside the paper edge - that is the bleed, trimmed off
    after printing.
    """

    __slots__ = ("x", "y", "w", "h", "photo", "effective_dpi", "crop_loss",
                 "bleed", "frame", "shadow", "fixed")

    def __init__(self, x, y, w, h, bleed=False, frame=0.0, shadow=False,
                 fixed=False):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.photo = None
        self.effective_dpi = 0.0
        self.crop_loss = 0.0
        self.bleed = bleed
        self.frame = frame        # white border in mm, for overlaid insets
        self.shadow = shadow
        self.fixed = fixed        # never shrink this slot to the photo shape

    @property
    def aspect(self):
        return self.w / self.h if self.h else 1.0

    @property
    def area(self):
        return self.w * self.h


class Page:
    __slots__ = ("index", "template", "slots", "kind", "look")

    def __init__(self, index, template, slots, kind="photos"):
        self.index = index
        self.template = template
        self.slots = slots
        self.kind = kind
        self.look = None           # set by enhance.auto_looks; None = album look

    @property
    def photos(self):
        return [s.photo for s in self.slots if s.photo is not None]

    @property
    def is_bleed(self):
        return any(s.bleed for s in self.slots)


def page_size(name, orientation="auto", photos=None):
    """Resolve a page-size name and orientation into (width_mm, height_mm)."""
    w, h = PAGE_SIZES_MM[name.lower()]
    if orientation == "portrait":
        return (min(w, h), max(w, h))
    if orientation == "landscape":
        return (max(w, h), min(w, h))
    if orientation == "auto" and photos:
        # Portrait pages are the more useful shape even for landscape photos:
        # two wide frames stack naturally down a tall page, whereas a landscape
        # page split in half gives two narrow slots that no wide photo fits.
        # Only an overwhelmingly landscape set is worth turning the paper.
        land = sum(1 for p in photos if p.shape == "landscape")
        total = max(1, len(photos))
        if land / float(total) >= 0.80:
            return (max(w, h), min(w, h))
        return (min(w, h), max(w, h))
    return (w, h)


def _spans(total, weights, count, gutter):
    """Split `total` into `count` spans with the given relative weights."""
    if not weights or len(weights) != count:
        weights = [1.0] * count
    usable = total - gutter * (count - 1)
    wsum = float(sum(weights)) or 1.0
    sizes = [usable * (w / wsum) for w in weights]
    offsets, run = [], 0.0
    for s in sizes:
        offsets.append(run)
        run += s + gutter
    return offsets, sizes


def build_slots(name, page_w, page_h, margin, gutter, bleed_mm=3.0,
                bleed_grid=False):
    """Lay a template out in millimetres.

    `bleed_grid` runs a multi-photo grid off all four paper edges, leaving only
    the thin gutters between pictures. That is the dominant look in printed
    wedding albums - panels butted together, no white border round the outside.
    """
    t = TEMPLATES[name]

    if t.get("style") == "overlap":
        # Explicit rectangles rather than a grid: the whole point is that the
        # pieces are not aligned to each other.
        out = []
        for (fx, fy, fw, fh, opt) in t["rects"]:
            if opt.get("bleed"):
                out.append(Slot(-bleed_mm, -bleed_mm,
                                page_w + 2 * bleed_mm, page_h + 2 * bleed_mm,
                                bleed=True, fixed=True))
            else:
                out.append(Slot(fx * page_w, fy * page_h,
                                fw * page_w, fh * page_h,
                                frame=opt.get("frame", 0.0),
                                shadow=opt.get("shadow", False)))
        return out

    cols, rows, cells = t["cols"], t["rows"], t["cells"]

    if t.get("style") == "bleed":
        # Run past every paper edge so the trim never leaves a white sliver.
        origin_x, origin_y = -bleed_mm, -bleed_mm
        inner_w = page_w + 2 * bleed_mm
        inner_h = page_h + 2 * bleed_mm
        gut = 0.0
    elif bleed_grid:
        origin_x, origin_y = -bleed_mm, -bleed_mm
        inner_w = page_w + 2 * bleed_mm
        inner_h = page_h + 2 * bleed_mm
        gut = gutter
    else:
        origin_x, origin_y = margin, margin
        inner_w = page_w - 2 * margin
        inner_h = page_h - 2 * margin
        gut = gutter

    xoff, xsize = _spans(inner_w, t.get("col_w"), cols, gut)
    yoff, ysize = _spans(inner_h, t.get("row_h"), rows, gut)

    edge = (t.get("style") == "bleed") or bleed_grid
    slots = []
    for (c, r, cs, rs) in cells:
        x = origin_x + xoff[c]
        y = origin_y + yoff[r]
        w = sum(xsize[c:c + cs]) + gut * (cs - 1)
        h = sum(ysize[r:r + rs]) + gut * (rs - 1)
        # A bleeding grid must fill its cell exactly, so those slots are fixed:
        # shrinking one to its photo's shape would open a white gap at the trim.
        slots.append(Slot(x, y, w, h, bleed=edge, fixed=bleed_grid))
    return slots


def band_slot(photo, page_w, page_h, bleed_mm=3.0):
    """A photo run to two opposite paper edges at its own proportions.

    The case this exists for: a wide photo on a tall page. Cropping it to full
    bleed would cut people off the ends, and centring it inside the margins
    leaves it floating in white like a slide. Letting it span the full width
    instead - bleeding left and right, white only above and below - keeps every
    face, wastes no width, and is a normal, deliberate album page.
    """
    a = max(0.05, photo.aspect)
    page_aspect = page_w / float(page_h)

    if a > page_aspect:          # wider than the page: bleed left and right
        w = page_w + 2 * bleed_mm
        h = w / a
        x = -bleed_mm
        y = (page_h - h) / 2.0
    else:                        # taller than the page: bleed top and bottom
        h = page_h + 2 * bleed_mm
        w = h * a
        y = -bleed_mm
        x = (page_w - w) / 2.0
    return [Slot(x, y, w, h, bleed=True, fixed=True)]


def adaptive_slot(photo, page_w, page_h, margin, fill=1.0, align="center"):
    """A slot shaped to the photo, sized to fill `fill` of the available area.

    This is what lets a single photograph keep its own proportions on the page
    instead of being cropped to fit a fixed rectangle - and the space left
    around it is the white space that makes an album page look considered.
    """
    inner_w = page_w - 2 * margin
    inner_h = page_h - 2 * margin
    a = max(0.05, photo.aspect)

    # Largest rectangle of the photo's aspect that fits inside the margins.
    if inner_w / inner_h > a:
        h = inner_h
        w = h * a
    else:
        w = inner_w
        h = w / a

    scale = math.sqrt(max(0.05, min(1.0, fill)))
    w, h = w * scale, h * scale

    # Always centred. An off-centre photo needs something to balance it; alone
    # on the page it just reads as a mistake, which is exactly how it looked.
    x = (page_w - w) / 2.0
    y = (page_h - h) / 2.0

    return [Slot(x, y, w, h)]


def sharp_pixels(photo):
    """Width and height in pixels that are actually sharp.

    A frame that is smeared at full size but crisp at half has, for printing,
    half the pixels its file claims. Using these everywhere a print size is
    decided is what sends soft photos to small slots instead of full pages.
    """
    k = (photo.scores or {}).get("sharp_scale", 1.0)
    return photo.width * k, photo.height * k


def effective_dpi(photo, slot):
    """Printed resolution once the photo is scaled to cover the slot."""
    w_in = slot.w / 25.4
    h_in = slot.h / 25.4
    if w_in <= 0 or h_in <= 0:
        return 0.0
    w_px, h_px = sharp_pixels(photo)
    return min(w_px / w_in, h_px / h_in)


def crop_loss(photo, slot):
    """Fraction of the photo discarded when cropped to the slot's aspect."""
    pa, sa = photo.aspect, slot.aspect
    if pa <= 0 or sa <= 0:
        return 1.0
    return 1.0 - (min(pa, sa) / max(pa, sa))


def _assign_cost(photo, slot, min_dpi, max_loss=MAX_CROP_LOSS):
    """Lower is better. Returns inf when the slot is the wrong shape."""
    loss = crop_loss(photo, slot)
    if loss > max_loss:
        # Refusing outright is what stops both letterboxing and the crops that
        # cut people in half: the chooser must find a slot that actually fits.
        return float("inf")

    dpi = effective_dpi(photo, slot)
    if dpi < min_dpi:
        # Refuse rather than merely discourage. The chooser then falls back to
        # a denser template with smaller cells, or to a single page, where the
        # slot is shrunk to whatever the photo can actually fill sharply.
        return float("inf")

    cost = loss * 2.4
    if dpi < TARGET_DPI:
        cost += (TARGET_DPI - dpi) / TARGET_DPI * 0.35
    # Mild preference for putting the strongest photo in the biggest slot.
    cost -= (photo.score / 100.0) * (slot.area ** 0.5) * 0.0022
    return cost


def fit_slot_to_photo(slot, photo, budget=None):
    """Shrink a slot inside its cell until it nearly matches the photo's shape.

    Grid cells have whatever proportions the template gives them, which is
    rarely what the photograph is. Rather than cropping the picture to the
    cell - the thing that produced letterboxing, sliced faces and pages of one
    lonely photo - the rectangle itself gives way, keeping its centre and
    losing the excess. A small crop budget is still spent first, so pages stay
    reasonably full rather than dissolving into white space.
    """
    budget = FILL_CROP_BUDGET if budget is None else budget
    pa = max(0.05, photo.aspect)
    lo, hi = pa * (1.0 - budget), pa / (1.0 - budget)
    target = min(max(slot.aspect, lo), hi)

    cx, cy = slot.x + slot.w / 2.0, slot.y + slot.h / 2.0
    if slot.w / slot.h > target:        # cell too wide - narrow it
        h = slot.h
        w = h * target
    else:                                # cell too tall - shorten it
        w = slot.w
        h = w / target
    slot.x, slot.y = cx - w / 2.0, cy - h / 2.0
    slot.w, slot.h = w, h
    return slot


def _best_assignment(photos, slots, min_dpi):
    """Greedy slot filling by cost. Returns (slots, cost) or (None, inf)."""
    pairs = []
    for pi, p in enumerate(photos):
        for si, s in enumerate(slots):
            c = _assign_cost(p, s, min_dpi)
            if c != float("inf"):
                pairs.append((c, pi, si))
    pairs.sort(key=lambda t: t[0])

    chosen = []
    used_p, used_s, total = set(), set(), 0.0
    for cost, pi, si in pairs:
        if pi in used_p or si in used_s:
            continue
        used_p.add(pi)
        used_s.add(si)
        chosen.append((pi, si))
        total += cost

    if len(chosen) != len(photos):
        return None, float("inf")     # somebody had nowhere acceptable to go

    for pi, si in chosen:
        slot, photo = slots[si], photos[pi]
        if not slot.bleed and not slot.fixed:
            fit_slot_to_photo(slot, photo)
        slot.photo = photo
        slot.effective_dpi = effective_dpi(photo, slot)
        slot.crop_loss = crop_loss(photo, slot)
        photo.used_slot = slot
    return slots, total


# Where an overlaid inset may sit, as (x, y) of its top-left corner in page
# fractions. All four corners are tried and the emptiest one wins.
INSET_POSITIONS = ((0.070, 0.585), (0.570, 0.585),
                   (0.070, 0.075), (0.570, 0.075))


def _faces_in_page_space(photo, page_aspect):
    """Approximate where this photo's faces land once it fills the page.

    A cover-crop is assumed to be roughly centred. The real crop is content
    aware and therefore keeps faces even more securely inside the frame, so
    this errs towards over-reporting where people are - which is the safe
    direction when deciding what not to cover up.
    """
    boxes = getattr(photo, "face_boxes", None) or []
    if not boxes:
        return []
    pa = max(0.05, photo.aspect)
    if pa > page_aspect:                     # sides cropped away
        vis_w, vis_h = page_aspect / pa, 1.0
    else:                                    # top and bottom cropped away
        vis_w, vis_h = 1.0, pa / page_aspect
    ox, oy = (1.0 - vis_w) / 2.0, (1.0 - vis_h) / 2.0

    out = []
    for (fx, fy, fw, fh, _s) in boxes:
        x0 = (fx - ox) / vis_w
        y0 = (fy - oy) / vis_h
        out.append((x0, y0, fw / vis_w, fh / vis_h))
    return out


def _place_inset(slots, page_w, page_h):
    """Move an overlaid inset to wherever it hides the least of the photo below.

    Dropping a second picture on top of the first is only worth doing if it
    lands on background - sky, wall, foliage. Covering somebody's face with it
    ruins both photographs at once.
    """
    bg = next((s for s in slots if s.bleed and s.photo is not None), None)
    inset = next((s for s in slots if s.frame and s.photo is not None), None)
    if bg is None or inset is None:
        return slots

    page_aspect = page_w / float(page_h)
    faces = _faces_in_page_space(bg.photo, page_aspect)
    fw, fh = inset.w / page_w, inset.h / page_h

    best, best_cost = None, None
    for (px, py) in INSET_POSITIONS:
        cost = 0.0
        for (x0, y0, w0, h0) in faces:
            ox = max(0.0, min(px + fw, x0 + w0) - max(px, x0))
            oy = max(0.0, min(py + fh, y0 + h0) - max(py, y0))
            overlap = ox * oy
            if overlap > 0:
                # Weighted by face size: obscuring the main subject is far
                # worse than clipping a stranger in the background.
                cost += overlap * (1.0 + 12.0 * (w0 * h0))
        # All else equal, prefer the lower corners - that is where the eye
        # expects an inset to sit in a printed album.
        cost += 0.004 if py < 0.4 else 0.0
        if best_cost is None or cost < best_cost:
            best, best_cost = (px, py), cost

    inset.x, inset.y = best[0] * page_w, best[1] * page_h
    return slots


def _page_cost(photos, template, page_w, page_h, margin, gutter, min_dpi, bleed_mm,
               bleed_grid=False):
    slots = build_slots(template, page_w, page_h, margin, gutter, bleed_mm,
                        bleed_grid=bleed_grid)
    if len(slots) != len(photos):
        return None, float("inf")
    slots, cost = _best_assignment(list(photos), slots, min_dpi)
    if slots is not None and TEMPLATES[template].get("style") == "overlap":
        _place_inset(slots, page_w, page_h)
    return slots, cost


def max_printable(photo, min_dpi):
    """Largest width and height in mm this photo can fill at min_dpi."""
    if min_dpi <= 0:
        return float("inf"), float("inf")
    w_px, h_px = sharp_pixels(photo)
    return (w_px / min_dpi * 25.4, h_px / min_dpi * 25.4)


def _single_page(photo, page_w, page_h, margin, gutter, min_dpi, bleed_mm,
                 index, allow_bleed, align):
    """One photo alone: full bleed when it suits the paper, otherwise adaptive.

    Bleeding is tried for every lone photo, not just heroes. A single picture
    marooned in the middle of a white page with wide bands above and below is
    the most obviously amateur page an album can have; running it to the paper
    edge, or at least out to the margins, fixes it.
    """
    # A photo with too few pixels cannot fill a page honestly, whatever its
    # shape. Phone-messaging apps hand back 0.9 MP copies that look fine on a
    # screen and print at 85 DPI across A4 - so those are printed smaller
    # instead, which is the whole reason resolution is measured at all.
    max_w, max_h = max_printable(photo, min_dpi)
    fits_page = (max_w >= page_w * 0.99 and max_h >= page_h * 0.99)

    if allow_bleed and fits_page \
            and crop_loss(photo, Slot(0, 0, page_w, page_h)) <= BLEED_MAX_CROP_LOSS:
        slots, _ = _page_cost([photo], "bleed_full", page_w, page_h,
                              margin, gutter, min_dpi, bleed_mm)
        if slots is not None:
            return Page(index, "bleed_full", slots)

    # Full bleed was refused because the shapes are too far apart. Rather than
    # float the photo inside the margins, run it to the two edges it can fill
    # at its own proportions - no crop, no side margins.
    if allow_bleed and fits_page:
        slots = band_slot(photo, page_w, page_h, bleed_mm)
        slot = slots[0]
        slot.photo = photo
        slot.effective_dpi = effective_dpi(photo, slot)
        slot.crop_loss = crop_loss(photo, slot)
        photo.used_slot = slot
        return Page(index, "band", slots)

    slots = adaptive_slot(photo, page_w, page_h, margin, fill=1.0, align=align)
    slot = slots[0]

    # Shrink until it prints at an honest resolution.
    if slot.w > max_w or slot.h > max_h:
        shrink = min(max_w / slot.w if slot.w else 1.0,
                     max_h / slot.h if slot.h else 1.0, 1.0)
        new_w, new_h = slot.w * shrink, slot.h * shrink
        slot.x += (slot.w - new_w) / 2.0
        slot.y += (slot.h - new_h) / 2.0
        slot.w, slot.h = new_w, new_h
        why = ("printed smaller so its softness does not show"
               if (photo.scores or {}).get("sharp_scale", 1.0) < 1.0
               else "printed smaller so it stays sharp (%.1f MP)" % photo.megapixels)
        # Layout runs once per page-count attempt; say it once.
        if why not in photo.reasons:
            photo.reasons.append(why)
    slot.photo = photo
    slot.effective_dpi = effective_dpi(photo, slot)
    slot.crop_loss = crop_loss(photo, slot)
    photo.used_slot = slot
    return Page(index, "single_adaptive", slots)


def compose(selected, page_w, page_h, margin=16.0, gutter=2.5,
            min_dpi=MIN_DPI, max_per_page=4, hero_full_page=True,
            multi_photo_bias=0.9, bleed_mm=3.0, bleed_heroes=True,
            bleed_singles=True, bleed_multi_every=2,
            same_scene_seconds=300, same_scene_bits=20):
    """Group the selected photos into pages, chronological order preserved."""
    pages = []
    queue = list(selected)
    index = 0
    solo = 0
    multi = 0

    while queue:
        head = queue[0]

        # Is the very next photo another look at the same thing?
        twin_next = False
        if len(queue) > 1:
            a, b = queue[0], queue[1]
            twin_next = (
                (a.taken and b.taken and
                 abs((b.taken - a.taken).total_seconds()) <= same_scene_seconds)
                or (a.dhash is not None and b.dhash is not None
                    and _hamming(a.dhash, b.dhash) <= same_scene_bits))

        if hero_full_page and head.hero and not twin_next:
            pages.append(_single_page(
                head, page_w, page_h, margin, gutter, min_dpi, bleed_mm, index,
                allow_bleed=bleed_heroes,
                align="center"))
            solo += 1
            queue.pop(0)
            index += 1
            continue

        run = 0
        while run < min(max_per_page, len(queue)) and not queue[run].hero:
            run += 1
        run = max(1, run)

        # A photo that would otherwise sit alone on a page, next to another
        # frame of the same thing, shares the page with it instead. Two
        # consecutive full-page views of one temple read as an accident; the
        # pair on one page reads as a deliberate look at the same subject.
        # This applies even when one of them is a hero - the stronger photo
        # still takes the larger slot, because slots are assigned by score.
        if run == 1 and twin_next:
            run = 2

        # Alternate bleeding mosaics with bordered ones so the book has a
        # rhythm instead of every page looking identical.
        bleed_grid = (bleed_multi_every > 0 and
                      (multi % bleed_multi_every) == 0)

        best = (None, None, float("inf"))
        for count in range(run, 1, -1):
            group = queue[:count]
            for template in BY_COUNT.get(count, []):
                slots, cost = _page_cost(group, template, page_w, page_h,
                                         margin, gutter, min_dpi, bleed_mm,
                                         bleed_grid=bleed_grid)
                if slots is None:
                    continue
                # A page holding more photos earns a bonus, otherwise every
                # page collapses to a single picture. Raise multi_photo_bias in
                # album.json for a denser book, lower it for one per page.
                cost -= multi_photo_bias * (count - 1)
                if cost < best[2]:
                    best = (template, slots, cost)

        template, slots, _ = best
        if template is None:
            pages.append(_single_page(
                head, page_w, page_h, margin, gutter, min_dpi, bleed_mm, index,
                allow_bleed=bleed_singles, align="center"))
            solo += 1
            del queue[:1]
            index += 1
            continue

        used = len([s for s in slots if s.photo is not None])
        pages.append(Page(index, template, slots))
        del queue[:used]
        index += 1
        multi += 1

    return pages


def stats(pages):
    """Print-readiness figures for the report."""
    dpis = [s.effective_dpi for p in pages for s in p.slots if s.photo]
    losses = [s.crop_loss for p in pages for s in p.slots if s.photo]
    if not dpis:
        return {"pages": len(pages), "photos": 0}
    return {
        "pages": len(pages),
        "photos": len(dpis),
        "min_dpi": round(min(dpis)),
        "median_dpi": round(sorted(dpis)[len(dpis) // 2]),
        "below_min_dpi": sum(1 for d in dpis if d < MIN_DPI),
        "max_crop_loss_pct": round(max(losses) * 100, 1),
        "bleed_pages": sum(1 for p in pages if p.is_bleed),
        "multi_photo_pages": sum(1 for p in pages if len(p.photos) > 1),
    }
