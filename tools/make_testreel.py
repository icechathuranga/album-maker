#!/usr/bin/env python3
"""Build a synthetic event folder of phone-like videos to test make-reel.

Creates inbox/_selftest-reel/ from photos already in inbox/ (people make the
face code do real work) plus ffmpeg's own generators:

  VID..._pan*.mp4      landscape pans across a photo, with "crowd" audio
  VID..._rotated.mp4   stored sideways with a rotate=90 tag, like a phone
  VID..._vfr.mp4       variable frame rate
  VID..._hlg.mp4       10-bit HEVC flagged as HLG HDR
  VID..._cut.mp4       two scenes joined (a cut inside one clip)
  VID..._shaky.mp4     the same pan with hand-shake
  IMG...jpg            a few stills
  music-120bpm.m4a     a 40-second kick/hat track at 120 BPM

Every file carries a phone-style timestamp in its name, spread over a day.
Run:  python3 tools/make_testreel.py [source-event]
"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "inbox", "_selftest-reel")
FF = ["ffmpeg", "-v", "error", "-y"]


def run(args):
    subprocess.run(FF + args, check=True)


def people_photos(event, n):
    """Photos the album already judged good, with faces in them."""
    path = os.path.join(ROOT, "outbox", event, "album.json")
    with open(path) as fh:
        data = json.load(fh)
    good = [p for p in data["photos"] if p["status"] == "selected"
            and p["metrics"].get("faces", 0) >= 1 and p["width"] > p["height"]]
    good.sort(key=lambda p: -p["metrics"].get("faces_score", 0))
    return [os.path.join(ROOT, "inbox", event, p["rel"]) for p in good[:n]]


def crowd_audio(seconds):
    """Pink-ish noise bed with three cheer bursts and a 220 Hz hum."""
    return ("aevalsrc='0.05*sin(2*PI*220*t)+0.04*(random(0)-0.5)"
            "+0.5*(random(1)-0.5)*(between(mod(t,%g),1.2,1.9))':s=48000:d=%g"
            % (seconds / 2.0, seconds))


def pan(src, out, seconds=6, shake=False):
    x = "(iw-1920)*t/%d" % seconds
    y = "(ih-1080)/2"
    if shake:
        x += "+18*sin(t*31)+11*sin(t*53)"
        y += "+14*sin(t*41)+9*sin(t*67)"
    run(["-loop", "1", "-i", src, "-f", "lavfi", "-i", crowd_audio(seconds),
         "-vf", "scale=2600:-2,crop=1920:1080:'%s':'%s',fps=30" % (x, y),
         "-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-c:a", "aac", "-shortest", out])


def main():
    event = sys.argv[1] if len(sys.argv) > 1 else "Nemi-Birthday-2026-March"
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    srcs = people_photos(event, 6)
    if len(srcs) < 4:
        sys.exit("need an event with an album.json and photos of people")

    stamp = ["20260315_%02d%02d00" % (10 + i // 2, (i % 2) * 30) for i in range(12)]
    pan(srcs[0], os.path.join(OUT, "VID%s_pan1.mp4" % stamp[0]))
    pan(srcs[1], os.path.join(OUT, "VID%s_pan2.mp4" % stamp[1]))
    pan(srcs[2], os.path.join(OUT, "VID%s_shaky.mp4" % stamp[2]), shake=True)

    # Portrait footage stored sideways with a rotation flag, as phones do.
    tmp = os.path.join(OUT, "_portrait.mp4")
    run(["-loop", "1", "-i", srcs[3], "-f", "lavfi", "-i", crowd_audio(5),
         "-vf", "scale=-2:1920,crop=1080:1920,transpose=1,fps=30", "-t", "5",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", tmp])
    run(["-i", tmp, "-c", "copy", "-metadata:s:v:0", "rotate=90",
         os.path.join(OUT, "VID%s_rotated.mp4" % stamp[3])])
    os.remove(tmp)

    # Variable frame rate: drop an irregular pattern of frames.
    run(["-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30:duration=5",
         "-f", "lavfi", "-i", "sine=frequency=330:duration=5",
         "-vf", "select='not(mod(n\\,3))+not(mod(n\\,7))'", "-vsync", "vfr",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         os.path.join(OUT, "VID%s_vfr.mp4" % stamp[4])])

    # 10-bit HEVC flagged as HLG, like an iPhone or Samsung HDR video.
    run(["-loop", "1", "-i", srcs[4 % len(srcs)], "-f", "lavfi", "-i", crowd_audio(4),
         "-vf", "scale=2600:-2,crop=1920:1080,fps=30", "-t", "4",
         "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", "log-level=error",
         "-color_primaries", "bt2020", "-color_trc", "arib-std-b67",
         "-colorspace", "bt2020nc", "-tag:v", "hvc1", "-c:a", "aac", "-shortest",
         os.path.join(OUT, "VID%s_hlg.mp4" % stamp[5])])

    # Two scenes in one file.
    a, b = os.path.join(OUT, "_a.mp4"), os.path.join(OUT, "_b.mp4")
    pan(srcs[1], a, 3)
    pan(srcs[5 % len(srcs)], b, 3)
    lst = os.path.join(OUT, "_list.txt")
    with open(lst, "w") as fh:
        fh.write("file '%s'\nfile '%s'\n" % (a, b))
    run(["-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
         os.path.join(OUT, "VID%s_cut.mp4" % stamp[6])])
    for f in (a, b, lst):
        os.remove(f)

    # Stills between the clips.
    for i, src in enumerate(srcs[:4]):
        shutil.copy(src, os.path.join(OUT, "IMG%s.jpg" % stamp[7 + i]))

    # 120 BPM: a kick on every beat, a hat on every off-beat.
    run(["-f", "lavfi", "-i",
         "aevalsrc='0.9*sin(2*PI*55*t)*exp(-30*mod(t,0.5))"
         "+0.25*(random(0)-0.5)*exp(-90*mod(t+0.25,0.5))':s=44100:d=40",
         "-c:a", "aac", "-b:a", "160k", os.path.join(OUT, "music-120bpm.m4a")])
    print("wrote %s (%d files)" % (OUT, len(os.listdir(OUT))))


if __name__ == "__main__":
    main()
