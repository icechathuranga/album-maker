# Event photo album builder

Turns a folder of event photos into a printable PDF album.

```
inbox/Thailand-Bangkok-2026-March/   →   outbox/Thailand-Bangkok-2026-March.pdf
```

No AI at run time. The same folder always produces the same album, and every
decision it makes is written down where you can see and override it.

## Use it

```bash
./make-album "Thailand-Bangkok-2026-March"              # one event, defaults
./make-album "Trip" A4 landscape 35pages                # say it in plain words
./make-album "Trip" square10 30pages film tight
./make-album --all                                      # every folder in inbox/
./make-album --list                                     # what is waiting in inbox/
./make-album "Trip" --dry-run                           # review only, no PDF
./make-album "Trip" collage                             # pick a cover design
./make-album "Trip" --covers --dry-run                  # preview every cover design
```

Words after the event name are read as a spec, in any order: a paper size
(`a4 a3 a5 letter legal square8 square10 square12 8x8 10x10 12x12 6x4 7x5`),
an orientation (`portrait landscape auto`), a page count (`35pages`, `35`), a
look (`natural vivid warm film mono soft crisp golden`, or `autolook` to
choose one per chapter), a density (`tight` or `airy`), a cover
design (`classic strips collage split duotone frame`) or a resolution
(`200dpi`). Anything it does not recognise it tells you about
instead of ignoring. `--spec "A4 landscape 35pages"` does the same thing.

Drop a folder into `inbox/`, run it, collect three things from `outbox/`:

| Output | What it is |
|---|---|
| `<Event>.pdf` | the printable album |
| `<Event>/slideshow.html` | full-screen slideshow — arrows to move, **G** all pages, **F** full screen |
| `<Event>/report.html` | every photo, its scores, and why it was used or dropped |
| `<Event>/covers.pdf` | with `--covers`: one page per cover design, labelled, to choose from |

## Cover designs

| Style | What it looks like |
|---|---|
| `strips` (default) | five (portrait) or six (landscape) photos cut into tall vertical bars, title on a light band below |
| `collage` | a wall of 10-12 photos; the band across the middle is blurred into frosted glass and the title sits on it |
| `split` | one large photo on top, a band in a colour taken from that photo, and a row of three or four below |
| `duotone` | one photo printed in two inks drawn from its own colours, with a large title |
| `frame` | one photo matted on a light page with a keyline, like a gallery print |
| `classic` | one photo full bleed, darkened at the foot for the title (the old cover) |

Multi-photo covers are chosen to cover the whole event - each pick is marked
down for sharing a chapter or a few minutes with one already chosen, and for
tipping the set too far towards people or places. Strips prefer photos whose
faces fit inside a narrow bar, so a wide group selfie is not sliced through.
If a folder is too thin for the style asked for, the cover falls back to
`classic` and the run says so.

Run with `--covers` (fast with `--dry-run`) to get `covers.pdf` with every
design built from your photos, then set the one you like with a spec word,
`--cover-style`, or `"cover_style"` in `album.json`.

## Overriding its choices

Put either file in the **event folder**, one filename per line:

```
inbox/Thailand-Bangkok-2026-March/keep.txt    → force these in
inbox/Thailand-Bangkok-2026-March/skip.txt    → force these out
inbox/Thailand-Bangkok-2026-March/cover.txt   → use this one as the cover
```

The first photo named is the featured one: the whole cover in `classic`,
`duotone` and `frame`, the large photo in `split`, the top-left tile in
`collage`. For `strips`, the photos named fill the bars left to right in the
order written, and the tool fills any bars left over. For the multi-photo
designs you can list several: they are used first, and the tool fills the
rest.

`cover.txt` exists because choosing a cover means judging whether a picture
*represents* the trip, and nothing measurable separates a temple from a
close-up of a shopping bag - both are sharp, colourful, detailed and contain
no people. Three signals were tried against real folders (text density,
sky/openness, and how often the subject was photographed) and none of them
separated the two. So name the photo you want.

`report.html` prints the exact line to paste for each photo. Use these rather
than fighting the settings — they always win.

## How it decides

1. **Read** every photo, correct rotation, and work out when it was taken:
   EXIF first, then the timestamp most cameras write into the filename
   (`IMG20260301105643`), which survives copying when EXIF does not, then the
   file date, then interpolation from its neighbours. Pages run in that order.
2. **Score** on two independent axes. *Technical* (sharpness, exposure,
   contrast, noise, JPEG blocking) is a gate. *Content* (faces, how big they
   are, eyes open, posed groups, single-subject portraits, subject isolation,
   colour, tonal interest) is what actually ranks.
   **Image size never affects ranking** — it only decides how large a photo
   can print.

   It has no way to recognise a photographed receipt, ticket or SIM packet.
   A text detector was tried and measured against real photos, where the
   signals turned out not to separate at all, so it was removed rather than
   left in to quietly demote good pictures. Put those filenames in
   `skip.txt` — that is what it is for.
3. **Reject** anything soft. Where there are people, sharpness is judged on
   their faces, not on the sharpest corner of the frame — and on *all* of
   them, so the group selfie where only the person holding the phone is in
   focus is ranked down rather than printed.

   Sharpness is also checked **at the size it will print**. Shrinking a
   16 MP phone photo to screen size hides motion blur completely, so the tool
   also measures how fast fine detail falls away as the frame is enlarged. A
   photo that is only sharp at half or quarter size is printed that small
   (never as a full page); one that is smeared even at half size is left out
   as "blurred at print size". `report.html` says how wide each soft photo
   can print. On the photos the old version printed, 14 of 40 full-page
   heroes were soft at full size.
4. **Collapse bursts** — near-identical frames within a few seconds become one,
   and the frame where eyes are open wins.
5. **Curate**: split the event into chapters by natural gaps in shooting, take
   the best of each so the whole day is covered, reserve a share for scenery,
   then fill the rest on merit across the whole event.
6. **Lay out** on templates matched to the photos' shapes — full bleed for the
   strongest, tight mosaics for the rest. Slots are shaped so nothing is
   letterboxed and no face is cut in half.
7. **Crop** with a content-aware search that keeps whole faces — including
   people it would be tempting to trim off the edge — and composes to the rule
   of thirds. A photo is never enlarged past the size its pixels support: a
   0.9 MP messaging-app copy prints small and sharp rather than filling a page
   at 85 DPI.
8. **Correct** colour and tone. The colour cast is measured only on what
   ought to be grey (walls, shirts, tablecloths), 60% of it is removed, and
   each chapter of the day gets one correction, so photos side by side match.
   A frame with almost no grey in it (a sunset, a stage, underwater) is left
   alone, and a correction that would turn faces green or blue is backed
   off. Crushed shadows and clipped highlights are eased back.
9. **Grade** with a look, resize to the exact pixels each slot needs at
   300 DPI, and write the PDF. With `look: auto` (the default) each chapter
   gets a look suited to its light and subject: `film` or `warm` for dim and
   grainy, `warm` or `golden` for lamplight, `vivid` or `soft` for people,
   `vivid` or `crisp` for daylight scenery. The pick among those is seeded
   from the event name, so it varies from album to album but the same folder
   always gives the same result. Every photo on a page shares one look, and
   consecutive chapters don't repeat the same look. `mono` is only used when
   asked for.

Pages alternate between mosaics that bleed off all four paper edges — the
dominant look in printed wedding albums, and the one the reference spreads in
`refs/` use — and bordered clusters, so the book has a rhythm.

Where a photo carries GPS, the page names the place ("Bangkok, Thailand"),
resolved against a 34,000-entry city list bundled in `models/`. Nothing is
looked up online.

## Settings

`./make-album --init-config` writes `album.json` next to the tool. Drop an
`album.json` inside an event folder to override just that album.

The ones worth knowing:

| Setting | Default | Effect |
|---|---|---|
| `target_pages` | 24 | aim for this many pages; a thin folder gives fewer rather than padding |
| `page_size` | `a4` | `a4 a5 a3 letter legal square8 square10 square12 6x4 7x5` |
| `orientation` | `portrait` | portrait stacks wide photos best; `auto` turns the paper only for a heavily landscape set |
| `look` | `auto` | `auto` (per chapter), or one of `natural vivid warm film mono soft crisp golden` for the whole album |
| `cover_style` | `strips` | `classic strips collage split duotone frame` |
| `sharpness_floor` | 0.55 | raise to be stricter about blur |
| `print_sharpness` | `normal` | `strict`, `normal` or `off` - how soft a photo may be at full size before it is printed smaller |
| `multi_photo_bias` | 0.9 | higher = more photos per page |
| `fill_crop_budget` | 0.52 | higher = fuller pages, less white space, more of each photo trimmed |
| `margin_mm` / `gutter_mm` | 16 / 2.5 | wide page margin, tight gaps between photos — the printed-album look |
| `bleed_multi_every` | 2 | every Nth multi-photo page runs edge to edge (1 = all, 0 = none) |
| `captions` | `place` | `off`, `date`, `place` (nothing without GPS), or `auto` |
| `scenic_share` | 0.28 | album share reserved for places, objects and views (`--scenic`) |
| `max_per_scene` | 2 | how many frames one location may contribute |
| `hero_separation_seconds` | 240 | minimum gap between two full-page photos |
| `page_tolerance` | 0.10 | how close to the requested page count it must land |
| `photos_per_page` | 2.4 | starting density; the fitting loop adjusts it |

## Choosing the page count

Page count is an *output* of the layout, not an input: how many pages a set of
photos fills depends on their shapes and which templates fit them. So the tool
composes the album, measures it, adjusts how many photos to select, and repeats
until it lands within `page_tolerance` of what you asked for. If the folder
runs out of photos worth printing it says so rather than padding.

For an album of memories the number that matters is how many *distinct moments*
survive - photos taken seconds apart of the same thing count once. `report.html`
and `album.json` record which moments were covered. On a 556-photo folder, 24
pages held 119 of 267 moments; 75 pages held 182. Raising the page count is a
far more effective lever than adjusting any scoring weight.

## Requirements

Python 3.10+. The libraries live in `vendor/` and the face model in
`models/` — nothing is downloaded at run time. Ghostscript (`gs`) is optional
and only used for the slideshow.

`vendor/` is not kept in git. After cloning, install the libraries into it
once (and re-run the same command to upgrade them):

```bash
pip3 install --target vendor --upgrade \
  "opencv-python-headless==4.11.0.86" pillow pillow-heif numpy reportlab piexif
```

**Keep OpenCV pinned below 5.** OpenCV 5 removed `CascadeClassifier` and ships
no Haar cascades, which silently disables face and eye detection. The banner
line printed on every run tells you which face detector is actually live.
