#!/usr/bin/env python3
"""BALL-position clicker: gold labels that break the teacher ceiling (BallNet program).

One click per frame ON the ball; if it's not visible press n; if you can't tell press u.
Auto-advances by --step frames per answer (~5 min per camera-window). Labels are BOTH
training gold (positives + TRUE negatives — far cleaner than teacher-absent frames) and
the first true ball-position eval.

GT JSON per clip: data/gt_ball/<name>.json  {frame: [x, y] | "none" | "unclear"}
(x,y in ORIGINAL pixel coordinates)

  python scripts/annotate_ball.py --manifest configs/ball_windows.json    # :8005
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, JSONResponse

UI = """<!doctype html><html><head><meta charset=utf-8><title>Ball clicks</title><style>
body{margin:0;background:#111;color:#ddd;font:15px system-ui;display:flex;flex-direction:column;align-items:center}
#bar{padding:8px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}.pill{background:#262626;border-radius:10px;padding:3px 10px}
#im{max-width:98vw;max-height:78vh;cursor:crosshair;border:1px solid #333}
kbd{background:#333;border-radius:4px;padding:1px 5px}#help{padding:6px;color:#999;font-size:13px}
button{font-size:14px;padding:4px 10px;border-radius:8px;border:1px solid #444;background:#222;color:#ddd;cursor:pointer}</style>
</head><body>
<div id=bar><button onclick="clipTo(cur-1)">&#8678; prev</button><span class=pill id=clip>clip</span>
<button onclick="clipTo(cur+1)">next &#8680;</button><span class=pill id=pos>0</span>
<span class=pill id=done>0 labeled</span><span class=pill id=cue>&nbsp;</span></div>
<img id=im><div id=help><b>CLICK the ball</b> (auto-advance) &nbsp; <kbd>n</kbd>no ball visible
 <kbd>u</kbd>can't tell &nbsp; <kbd>&larr;/&rarr;</kbd>&plusmn;1 <kbd>j/k</kbd>&plusmn;10
 <kbd>z</kbd>undo <kbd>g</kbd>next unlabeled <kbd>[</kbd>/<kbd>]</kbd>clips</div>
<script>
let i=0,N=1,cur=0,clips=[],step=2,gt={};
const im=document.getElementById('im');
function show(){im.src='/frame/'+i+'?c='+cur;
 document.getElementById('pos').textContent=i+'/'+(N-1);
 document.getElementById('done').textContent=Object.keys(gt).length+' labeled';
 const v=gt[i];document.getElementById('cue').textContent=v?(Array.isArray(v)?'ball @ '+v.map(Math.round):v):'--';}
function applyState(d){N=d.n_frames;gt=d.gt;cur=d.cur;clips=d.clips;step=d.step;
 document.getElementById('clip').textContent=(cur+1)+'/'+clips.length+'  '+clips[cur];}
async function clipTo(k){if(k<0||k>=clips.length)return;
 const d=await(await fetch('/api/clip/'+k)).json();applyState(d);i=0;show();}
async function mark(val){const r=await fetch('/api/mark',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({frame:i,val})});gt=await r.json();i=Math.min(N-1,i+step);show();}
im.onclick=e=>{const r=im.getBoundingClientRect();
 const x=(e.clientX-r.left)/r.width*im.naturalWidth, y=(e.clientY-r.top)/r.height*im.naturalHeight;
 mark([Math.round(x*10)/10, Math.round(y*10)/10]);};
document.addEventListener('keydown',e=>{
 if(e.key==='ArrowRight'&&i<N-1)i++;else if(e.key==='ArrowLeft'&&i>0)i--;
 else if(e.key==='k')i=Math.min(N-1,i+10);else if(e.key==='j')i=Math.max(0,i-10);
 else if(e.key==='n')return mark('none');else if(e.key==='u')return mark('unclear');
 else if(e.key==='z')return fetch('/api/undo?c='+cur,{method:'POST'}).then(r=>r.json()).then(d=>{gt=d;show();});
 else if(e.key==='g'){for(let f=0;f<N;f+=step){if(!(f in gt)){i=f;break;}}}
 else if(e.key===']')return clipTo(cur+1);else if(e.key==='[')return clipTo(cur-1);
 else return; show();});
fetch('/api/state').then(r=>r.json()).then(d=>{applyState(d);
 for(let f=0;f<N;f+=step){if(!(f in d.gt)){i=f;break;}} show();});
</script></body></html>"""

app = FastAPI(title="ball clicker")
STATE: dict = {"clips": [], "cur": 0, "ops": {}}


def _clip() -> dict:
    c = STATE["clips"][STATE["cur"]]
    if c.get("jpegs") is None:
        import cv2
        print(f"[encode] {c['name']} ...", flush=True)
        cap = cv2.VideoCapture(c["video"])
        jpegs = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            jpegs.append(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
        cap.release()
        c["jpegs"] = jpegs
        print(f"[encode] {c['name']}: {len(jpegs)} frames", flush=True)
    return c


def _gt_path(c: dict) -> Path:
    return Path(f"data/gt_ball/{c['name']}.json")


def _gt(c: dict) -> dict:
    p = _gt_path(c)
    return json.loads(p.read_text()) if p.exists() else {}


def _save(c: dict, gt: dict) -> None:
    p = _gt_path(c)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(gt))
    os.replace(tmp, p)


def _state() -> dict:
    c = _clip()
    return {"n_frames": len(c["jpegs"]), "gt": _gt(c), "cur": STATE["cur"],
            "clips": [x["name"] for x in STATE["clips"]], "step": c.get("step", 2)}


@app.get("/", response_class=HTMLResponse)
def index():
    return UI


@app.get("/frame/{i}")
def frame(i: int):
    jpegs = _clip()["jpegs"]
    return Response(jpegs[max(0, min(i, len(jpegs) - 1))], media_type="image/jpeg")


@app.get("/api/state")
def state():
    return _state()


@app.get("/api/clip/{k}")
def switch(k: int):
    STATE["cur"] = max(0, min(k, len(STATE["clips"]) - 1))
    return _state()


@app.post("/api/mark")
async def mark(body: dict):
    c = _clip()
    gt = _gt(c)
    f, val = str(int(body.get("frame", 0))), body.get("val")
    ok_xy = isinstance(val, list) and len(val) == 2
    if not (ok_xy or val in ("none", "unclear")):
        return JSONResponse({"error": "bad mark"}, status_code=422)
    gt[f] = val
    STATE["ops"].setdefault(STATE["cur"], []).append(f)
    _save(c, gt)
    return gt


@app.post("/api/undo")
async def undo(c: int = 0):
    cl = _clip()
    gt = _gt(cl)
    ops = STATE["ops"].get(STATE["cur"], [])
    if ops:
        gt.pop(ops.pop(), None)
        _save(cl, gt)
    return gt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSON list of {name, video, step}")
    ap.add_argument("--port", type=int, default=8005)
    a = ap.parse_args()
    clips = json.loads(Path(a.manifest).read_text())
    for c in clips:
        c.setdefault("jpegs", None)
        c.setdefault("step", 2)
    STATE["clips"] = clips
    _clip()
    print(f"ball clicker: {len(clips)} clip(s) -> data/gt_ball/<name>.json")
    uvicorn.run(app, host="127.0.0.1", port=a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
