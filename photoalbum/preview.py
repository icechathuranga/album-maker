"""A full-screen slideshow of the finished album, written next to the PDF.

The PDF is the deliverable, but flicking through pages in a viewer is a poor
way to judge a layout. This renders every page to an image and wraps them in a
single self-contained HTML file: arrow keys or swipe to move, F for full
screen, G for a grid of every spread at once.

It runs on Ghostscript, which is already needed to check print output, and
degrades to "no slideshow" rather than failing the build if it is absent.
"""
import glob
import html
import os
import shutil
import subprocess


def _ghostscript():
    for name in ("gs", "gswin64c", "gswin32c"):
        path = shutil.which(name)
        if path:
            return path
    return None


def rasterise(pdf_path, out_dir, dpi=96, max_pages=200):
    """Render PDF pages to JPEGs. Returns the list of relative paths."""
    gs = _ghostscript()
    if not gs:
        return []

    pages_dir = os.path.join(out_dir, "pages")
    if os.path.isdir(pages_dir):
        shutil.rmtree(pages_dir, ignore_errors=True)
    os.makedirs(pages_dir, exist_ok=True)

    cmd = [
        gs, "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER",
        "-sDEVICE=jpeg", "-dJPEGQ=82",
        "-r%d" % dpi,
        "-dLastPage=%d" % max_pages,
        "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4",
        "-sOutputFile=%s" % os.path.join(pages_dir, "page%03d.jpg"),
        pdf_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=600)
    except Exception:
        return []

    files = sorted(glob.glob(os.path.join(pages_dir, "page*.jpg")))
    return [os.path.relpath(f, out_dir) for f in files]


CSS = """
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0b0b0d;--fg:#f2f2f4;--dim:#7d8088;--accent:#e8b45a}
html,body{height:100%}
body{background:var(--bg);color:var(--fg);overflow:hidden;
     font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,sans-serif}

#stage{position:fixed;inset:0;display:grid;place-items:center;padding:3.2vmin 3.2vmin 8vmin}
.slide{position:absolute;inset:3.2vmin 3.2vmin 8vmin;display:grid;place-items:center;
       opacity:0;transform:scale(1.03);transition:opacity .55s ease,transform .55s ease;
       pointer-events:none}
.slide.on{opacity:1;transform:scale(1);pointer-events:auto}
.slide img{max-width:100%;max-height:100%;object-fit:contain;
           box-shadow:0 2.4vmin 6vmin rgba(0,0,0,.62);border-radius:2px;background:#fff}

/* soft vignette so the page edges read against the dark ground */
#stage::after{content:"";position:fixed;inset:0;pointer-events:none;
   background:radial-gradient(120% 90% at 50% 45%,transparent 55%,rgba(0,0,0,.55))}

#bar{position:fixed;left:0;right:0;bottom:0;height:8vmin;min-height:52px;
     display:flex;align-items:center;gap:18px;padding:0 22px;z-index:5;
     background:linear-gradient(to top,rgba(0,0,0,.85),transparent)}
#bar button{background:none;border:1px solid rgba(255,255,255,.22);color:var(--fg);
     width:34px;height:34px;border-radius:50%;cursor:pointer;font-size:15px;line-height:1;
     display:grid;place-items:center;transition:background .2s,border-color .2s}
#bar button:hover{background:rgba(255,255,255,.12);border-color:rgba(255,255,255,.45)}
#count{color:var(--dim);font-variant-numeric:tabular-nums;letter-spacing:.04em;font-size:12.5px}
#title{color:var(--fg);font-weight:600;letter-spacing:-.01em;margin-right:auto}
#title span{color:var(--dim);font-weight:400;margin-left:10px;font-size:12.5px}
#track{position:fixed;left:0;top:0;height:2px;background:var(--accent);width:0;
       transition:width .45s ease;z-index:6}
#hint{position:fixed;right:22px;bottom:calc(8vmin + 14px);color:var(--dim);font-size:11.5px;
      opacity:.85;z-index:5}

/* grid overview */
#grid{position:fixed;inset:0;background:var(--bg);overflow-y:auto;padding:26px;
      display:none;grid-template-columns:repeat(auto-fill,minmax(168px,1fr));gap:16px;z-index:10}
#grid.on{display:grid}
#grid figure{cursor:pointer;transition:transform .18s}
#grid figure:hover{transform:translateY(-3px)}
#grid img{width:100%;display:block;border-radius:2px;background:#fff;
          box-shadow:0 6px 20px rgba(0,0,0,.5)}
#grid figcaption{color:var(--dim);font-size:11px;text-align:center;margin-top:6px}
@media (max-width:640px){#hint{display:none}}
"""

JS = """
const slides=[...document.querySelectorAll('.slide')];
const track=document.getElementById('track');
const count=document.getElementById('count');
const grid=document.getElementById('grid');
let i=0,timer=null;

function show(n){
  i=(n+slides.length)%slides.length;
  slides.forEach((s,k)=>s.classList.toggle('on',k===i));
  count.textContent=(i+1)+' / '+slides.length;
  track.style.width=((i+1)/slides.length*100)+'%';
}
function next(){show(i+1)}
function prev(){show(i-1)}
function togglePlay(){
  const b=document.getElementById('play');
  if(timer){clearInterval(timer);timer=null;b.textContent='▶';}
  else{timer=setInterval(next,3800);b.textContent='❚❚';next();}
}
function toggleGrid(){grid.classList.toggle('on')}
function full(){
  if(!document.fullscreenElement)document.documentElement.requestFullscreen?.();
  else document.exitFullscreen?.();
}
document.addEventListener('keydown',e=>{
  if(e.key==='ArrowRight'||e.key===' '){e.preventDefault();next()}
  else if(e.key==='ArrowLeft')prev();
  else if(e.key==='f'||e.key==='F')full();
  else if(e.key==='g'||e.key==='G')toggleGrid();
  else if(e.key==='p'||e.key==='P')togglePlay();
  else if(e.key==='Escape')grid.classList.remove('on');
});
let x0=null;
document.addEventListener('touchstart',e=>x0=e.touches[0].clientX,{passive:true});
document.addEventListener('touchend',e=>{
  if(x0===null)return;
  const dx=e.changedTouches[0].clientX-x0;
  if(Math.abs(dx)>45){dx<0?next():prev()}
  x0=null;
},{passive:true});
document.getElementById('stage').addEventListener('click',e=>{
  if(e.target.closest('#bar'))return;
  (e.clientX>window.innerWidth/2)?next():prev();
});
grid.addEventListener('click',e=>{
  const f=e.target.closest('figure');
  if(f){show(+f.dataset.i);grid.classList.remove('on')}
});
show(0);
"""


def write_slideshow(out_dir, title, subtitle, page_files, pdf_rel):
    """Write slideshow.html into out_dir. Returns its path, or None."""
    if not page_files:
        return None

    slides = "\n".join(
        '<div class="slide"><img src="%s" alt="Page %d" %s></div>'
        % (html.escape(f), n + 1, 'loading="lazy"' if n > 2 else "")
        for n, f in enumerate(page_files))

    thumbs = "\n".join(
        '<figure data-i="%d"><img src="%s" alt="" loading="lazy">'
        '<figcaption>%d</figcaption></figure>' % (n, html.escape(f), n + 1)
        for n, f in enumerate(page_files))

    doc = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>%s</title><style>%s</style></head><body>
<div id="track"></div>
<div id="stage">%s</div>
<div id="grid">%s</div>
<div id="bar">
  <div id="title">%s<span>%s</span></div>
  <button id="prev" title="Previous (left arrow)">&#8249;</button>
  <button id="play" title="Play (P)">&#9654;</button>
  <button id="next" title="Next (right arrow)">&#8250;</button>
  <button id="gridb" title="All pages (G)">&#9638;</button>
  <button id="fs" title="Full screen (F)">&#9974;</button>
  <a href="%s" download style="color:var(--dim);text-decoration:none;font-size:12.5px">PDF</a>
  <div id="count"></div>
</div>
<div id="hint">click sides or &#8592; &#8594; &middot; P play &middot; G all pages &middot; F full screen</div>
<script>%s
document.getElementById('prev').onclick=prev;
document.getElementById('next').onclick=next;
document.getElementById('play').onclick=togglePlay;
document.getElementById('gridb').onclick=toggleGrid;
document.getElementById('fs').onclick=full;
</script></body></html>""" % (
        html.escape(title), CSS, slides, thumbs,
        html.escape(title), html.escape(subtitle),
        html.escape(pdf_rel), JS)

    path = os.path.join(out_dir, "slideshow.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path


def build(pdf_path, out_dir, title, subtitle, dpi=96):
    """Rasterise the album and write the slideshow. Returns its path or None."""
    files = rasterise(pdf_path, out_dir, dpi=dpi)
    if not files:
        return None
    return write_slideshow(out_dir, title, subtitle, files,
                           os.path.relpath(pdf_path, out_dir))
