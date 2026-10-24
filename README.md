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
choose one per chapter), a genre (`wedding travel zine`), a page style
(`paper blended floating bordered`), a density (`tight` or `airy`), a cover
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

A photo named in `cover.txt` goes on the cover even when the album itself
leaves it out (for example, as a near-duplicate of a frame it prints). It is
not forced into the pages; also list it in `keep.txt` if you want it there.

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

Pages follow what current wedding and travel album designers do
(MILK, Artifact Uprising, Graphistudio and others):

- **Curated pages.** One to three photos a page, one clearly leading. "Too
  many photos per page" is the most common amateur mistake, so the 4- and
  6-photo grids are kept for detail pages only.
- **Paper, not white.** Each page is a flat paper tone taken from its own
  photos: a faint warm cream beside sunsets, a cool grey beside the sea, and
  near-black on night pages. Photos have crisp edges. There are no drop
  shadows or blurred photo backgrounds, which designers now call dated.
- **One gap and one margin** across the whole book (`page_gap_mm`,
  `page_margin_mm`).
- **Photo on photo.** A small photo laid over a full-page one, on the
  quietest part of the picture, with a hairline paper edge. Sometimes two,
  stacked, when they show different things.
- **Pacing.** A busy page is always followed by a calm one. Photos with
  nobody in them are gathered into small **detail pages** between the
  people pages, and small photos that cannot fill a page share one in a
  row instead of floating alone.
- **Chapter openers.** Where shooting stops for more than 90 minutes or
  moves more than 5 km, the chapter opens with a full-page establishing shot
  and the town's name across it. At most one opener per 12 pages, and never
  the same town twice in a row.
- **A route map** after the cover when the photos carry GPS and the trip
  has three or more stops. It is drawn from the photos alone, with no map
  tiles and no internet.
- **Black and white, mixed in.** About 10% of photos (`mono_share`), a
  whole page at a time so they are never beside colour ones. The pages
  chosen are those where colour helps least, and people pages. The black
  and white is high-contrast, with rich blacks.

Other page styles: `blended` (edge to edge, photos cross-fading into each
other), `floating` (soft shadows over a blurred wash) and `bordered` (white
margins and gaps, the old look).

Genre presets set several of these at once: `wedding` (more black and
white, no map), `travel` (map and openers) and `zine` (film look, a quarter
black and white, tighter spacing). Anything else in the same command wins
over the preset, e.g. `./make-album "Trip" travel vivid`.

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
| `page_style` | `paper` | `paper` (tinted paper, curated pages), `blend`, `float` or `bordered` |
| `page_gap_mm` / `page_margin_mm` | 5 / 14 | the one gap and one margin used on paper pages |
| `mono_share` | 0.10 | share of photos printed black and white, a page at a time (0 = none) |
| `chapter_openers` | true | opening pages at real breaks in the day |
| `route_map` | true | a map of the trip after the cover when photos have GPS |
| `margin_mm` / `gutter_mm` | 16 / 2.5 | page margin and gaps for `bordered` pages |
| `bleed_multi_every` | 2 | `bordered` only: every Nth multi-photo page runs edge to edge (1 = all, 0 = none) |
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

## Short videos: make-reel

The same event folder can also become a vertical video for TikTok, Reels or
Shorts (1080×1920, 30 fps, H.264/AAC, −14 LUFS), told in the order it
happened. With music it runs the length of the whole song, from its first
beat (or as long as the footage can fill); without music, 3 minutes. A
length you give (`90s`, `2m`, `--duration 178`) always wins and is met to
the frame: cut on whole beats, the last shot trimmed to fit.

```bash
./make-reel "Trip"                        # one new edit (a new seed every run)
./make-reel "Trip" --variants 4           # four different previews + index.html
./make-reel "Trip" --seed 48213 --final   # re-make the one you liked, full quality
./make-reel "Trip" 90s                    # length: 90s, 2m, 1m30s, or --duration 120
./make-reel "Trip" places soft            # steer it in words
./make-reel --list                        # videos, photos and music per folder
```

Put phone videos (`.mp4 .mov .m4v .3gp .webm .mkv`) beside the photos in
`inbox/<Event>/`, and your own music (`.mp3 .m4a .wav .flac`) in `music/`
next to the tool or in the event folder; nothing is downloaded, and
`music/` is kept out of git. List clips you do not want in the event's
`skip.txt`, as for the album. The analysis of the footage is cached in
`cache/reels/<Event>/` (safe to delete), so clearing `outbox/` never costs a
re-analysis.

### How it edits

The rules follow how a professional editor cuts a travel or event film:

- **The story in order.** Every clip moment and photo appears in capture
  order. The film opens on an **establishing shot** - early in the first
  part of the day, nobody close, calm and sharp - held 4-6 s with the title,
  placed away from any faces. It **closes** on the widest shot of the last
  part, with a fade. A small "DAY 2 · TOWN · 10 OCTOBER" caption and a dip
  mark each new day.
- **Every part of the day gets its share.** Screen time is divided between
  the chapters of the event, no chapter above 30%, so one long activity
  cannot swallow the mornings. A chapter of photos only becomes a short
  photo montage. When there is too much material, the middle is thinned by
  score; the end of the trip is never cut off.
- **Long recordings used to the full.** A long clip gives its first good
  wide shot, its moment of most action and its loudest moment (a cheer
  rather than talk), then more of its best moments - three, plus one per
  45 seconds, up to twelve - at least 20 s apart and visually distinct.
  Each chapter fills its time with video first and photos after.
- **Shot choice.** Blurred photos are left out: the album's softness
  rules, plus a check for motion blur (too little fine detail anywhere in
  the frame) and for photos too soft to fill a phone screen. Static talking
  heads are marked down, and so is soft 1080p landscape footage. Shake is measured at the full frame rate: very
  shaky moments are left out, mildly shaky ones stabilised (ffmpeg
  vidstab). Never three close-ups or four people-less shots in a row.
- **Rhythm.** About 2.2-3 s a shot for a 3-minute film (about 1-1.3 s for a
  30-second one), faster in busy chapters and slower at the start and end,
  with a held "breath" shot about every 35 s. Shots start just before the
  action. Long empty stretches can run at double speed, and one fast moment
  from 50-60 fps footage at half speed.
- **Music.** Tempo and beats are found offline; every cut lands on a beat,
  the film is a whole number of 16-beat phrases, and the music fades over
  its last two seconds on a phrase boundary.
- **Sound mixed like an editor.** Each clip is levelled on its own and the
  low rumble removed; clips cross-fade over 0.3 s; the next shot's voice or
  cheer can start early (a J-cut); under a photo the previous clip's sound
  carries on quietly (an L-cut). Music dips gently under kept natural sound.
  Loudness is set in two passes, so it does not pump.
- **Framing for vertical.** The 9:16 window keeps the main person in frame
  for the whole shot and follows them smoothly; scenery from landscape
  clips drifts slowly across. Photos drift 4-8%, never a fast zoom.
- **Colour per chapter**, all from one family of looks, so cuts inside a
  scene match and night is not graded like day; brightness and white
  balance matched within each chapter.
- **Transitions:** hard cuts. A dip only where the day changes, a dissolve
  only between two photos (`soft`), nothing at all with `clean`.

### Cover images

Every video comes with two cover images beside it:

| File | Size | For |
|---|---|---|
| `<video>.youtube.jpg` | 1280×720 | the YouTube thumbnail |
| `<video>.cover.jpg` | 1080×1920 | the cover frame on TikTok, Reels, Shorts |

They use the album's poster cover - full-height strips edge to edge with the
title in spaced capitals - built from the reel's own photos and from sharp
frames of its video shots, in the order they happened. Scenes and action
(the place, the jeep, the shrine) are preferred over people's close-ups,
and frames with large featureless patches are skipped. `index.html` shows
both under each video, ready to download.

### Variants

Every run picks a **seed**; the same seed always gives the same edit, saved
as `<Event>-s<seed>.recipe.json` beside the video in `outbox/<Event>/reels/`.
What a seed varies: the **emphasis** - `people` (people moments first, more
photos), `places` (scenery and wide shots first), `balanced` (every part of
the day equally) or `footage` (time by how much was filmed), drawn from
the seed alone so `--seed N` always re-makes the same edit - plus pacing,
colour family, transitions, which moments are used, one chapter in black
and white, and where in the music the film starts. `--variants N` gives each
version a different emphasis, and `index.html` shows them side by side with
the command to re-make each one.

### Limits

- 1080p landscape footage cropped to vertical is upscaled about 1.8× and
  looks a little soft; 4K or portrait footage stays sharp.
- Cheering and speech are found from loudness and voice frequencies, which
  is crude.
- The tool cannot tell what a clip is about: a pre-trip office call scores
  like any other face. `skip.txt` is the answer.
- The same seed always gives the same recipe; an `.mp4` can differ in a few
  bytes between machines.
- Requires ffmpeg (`sudo apt install ffmpeg`).

`tools/make_testreel.py` builds a synthetic test folder of phone-like clips
(rotated, HDR, variable frame rate, a cut inside a clip, a shaky clip) and a
120 BPM track, in `inbox/_selftest-reel/`.

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
