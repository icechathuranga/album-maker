"""index.html beside the reels: every version side by side, to choose from.

Each card plays the video, shows a strip of its frames, what kind of edit it
is, and the exact command that re-makes it at full quality.
"""
import glob
import html
import json
import os

from . import editor

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reels · %(event)s</title><style>
:root{--bg:#f6f3ee;--ink:#2b2622;--soft:#7a716a;--card:#fffdf9;--line:#e4ddd3}
@media (prefers-color-scheme:dark){:root{--bg:#171513;--ink:#ece6df;--soft:#a39a91;--card:#211e1b;--line:#38322d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,sans-serif;padding:28px 16px}
h1{font:400 26px/1.2 Georgia,serif;letter-spacing:.18em;text-transform:uppercase;margin:0 0 4px}
p.lead{color:var(--soft);margin:0 0 24px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
video{width:100%%;aspect-ratio:9/16;background:#000;border-radius:6px;display:block}
img{width:100%%;margin-top:8px;border-radius:4px;display:block}
.seed{font:600 13px ui-monospace,monospace;margin:10px 0 2px}
.sum{color:var(--soft);font-size:13px}
code{display:block;margin-top:8px;padding:6px 8px;background:var(--bg);border-radius:4px;
font-size:12px;overflow-x:auto;white-space:nowrap}
</style></head><body><h1>%(event)s</h1>
<p class="lead">%(count)d versions. Pick one and re-make it at full quality with its command.</p>
<div class="grid">%(cards)s</div></body></html>
"""

CARD = """<div class="card"><video src="%(video)s" controls preload="metadata" playsinline></video>
<img src="%(strip)s" alt="frames from seed %(seed)s" loading="lazy">
<div class="seed">seed %(seed)s%(quality)s</div><div class="sum">%(summary)s</div>
<code>./make-reel "%(event)s" --seed %(seed)s --final</code></div>"""


def write(reels_dir, event):
    cards = []
    recipes = sorted(glob.glob(os.path.join(reels_dir, "*.recipe.json")),
                     key=os.path.getmtime, reverse=True)
    for path in recipes:
        with open(path) as fh:
            r = json.load(fh)
        base = path[:-len(".recipe.json")]
        video = base + ".mp4"
        if not os.path.exists(video):
            continue
        cards.append(CARD % {
            "video": html.escape(os.path.basename(video)),
            "strip": html.escape(os.path.basename(base + ".strip.jpg")),
            "seed": r["seed"], "event": html.escape(event),
            "quality": " · preview" if base.endswith("-preview") else "",
            "summary": html.escape(editor.summary(r))})
    out = os.path.join(reels_dir, "index.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(PAGE % {"event": html.escape(event), "count": len(cards),
                         "cards": "".join(cards)})
    return out
