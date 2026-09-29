"""Collapse burst sequences and near-identical frames down to the best one.

Two photos join the same group when either:
  * they look almost identical (hamming <= EXACT_BITS) - a re-save or a copy, or
  * they look similar (hamming <= BURST_BITS) AND were shot within the burst
    window - the "everyone say cheese" run of five near-identical frames.

The time condition matters: two photos of the same wall taken hours apart are
different moments and both belong in the album.
"""
from .deps import np, Image
from . import ingest

EXACT_BITS = 4      # hamming distance for "the same picture twice"
BURST_BITS = 12     # hamming distance for "same moment, slightly different"
BURST_SECONDS = 12  # how far apart burst frames may be

# How much the similarity test relaxes for frames taken moments apart. Two
# photographs three seconds apart that look broadly alike are the same moment,
# even if someone moved enough to shift the hash - a fixed threshold let a pair
# four seconds apart through and they landed on facing full-page spreads.
TIGHT_SECONDS = 5.0
TIGHT_BONUS_BITS = 7


def dhash(photo, size=8):
    """64-bit difference hash: 1 bit per horizontal gradient in a 9x8 thumbnail."""
    try:
        with Image.open(photo.path) as img:
            from .deps import ImageOps
            img = ImageOps.exif_transpose(img).convert("L").resize(
                (size + 1, size), Image.BILINEAR)
    except Exception:
        return None
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hamming(a, b):
    return bin(a ^ b).count("1")


class _Union:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def group(photos, burst_seconds=BURST_SECONDS, burst_bits=BURST_BITS,
          exact_bits=EXACT_BITS):
    """Assign photo.group and mark all but the best of each group as duplicates.

    Photos must already be in chronological order. Comparison only looks
    forward until the time window closes, so this stays near-linear rather
    than comparing every pair.
    """
    live = [p for p in photos if p.status not in ("unreadable",)]
    for p in live:
        if p.dhash is None:
            p.dhash = dhash(p)

    usable = [p for p in live if p.dhash is not None]
    uf = _Union(len(usable))

    for i, a in enumerate(usable):
        for j in range(i + 1, len(usable)):
            b = usable[j]
            dt = None
            if a.taken and b.taken:
                dt = abs((b.taken - a.taken).total_seconds())
            dist = hamming(a.dhash, b.dhash)

            if dist <= exact_bits:
                uf.union(i, j)          # identical regardless of timing
            elif dt is not None and dt <= burst_seconds:
                # The closer together in time, the more visual difference is
                # still the same moment rather than a new one.
                allowed = burst_bits
                if dt <= TIGHT_SECONDS:
                    allowed += TIGHT_BONUS_BITS
                if dist <= allowed:
                    uf.union(i, j)

            # chronological order means once we are past the window, and the
            # frames are no longer identical, nothing further can match on time
            if dt is not None and dt > burst_seconds and dist > exact_bits:
                if j - i > 40:
                    break

    groups = {}
    for idx, p in enumerate(usable):
        groups.setdefault(uf.find(idx), []).append(p)

    n_groups = 0
    for gid, members in sorted(groups.items(), key=lambda kv: kv[1][0].seq):
        for p in members:
            p.group = n_groups
        if len(members) > 1:
            members.sort(key=lambda p: (-p.score, p.seq))
            best = members[0]
            # Eye state is already folded into the score, so the winner is
            # usually the open-eyed frame - worth saying out loud in the report.
            blinks = sum(1 for m in members[1:]
                         if 0 <= m.scores.get("eye_ratio", -1) < 0.5)
            if blinks and best.scores.get("eye_ratio", -1) >= 0.5:
                best.reasons.append(
                    "best of %d similar frames - the others had eyes closed"
                    % len(members))
            else:
                best.reasons.append("best of %d similar frames" % len(members))
            for other in members[1:]:
                other.status = "duplicate"
                other.reasons.append("near-duplicate of %s" % best.name)
        n_groups += 1

    return n_groups
