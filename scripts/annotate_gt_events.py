#!/usr/bin/env python3
"""Ground-truth EVENT labeller (audit P1): step through clips and record who has possession
+ pass/turnover moments. ~10 min of operator time per clip turns every possession/pass claim
into measured precision/recall.

PLAYLIST: clips come from a manifest (configs/gt_windows.json) and the operator switches
clips IN the UI ([ / ] keys or buttons) — no server relaunch. Each entry can show several
cameras SIDE BY SIDE (cover both baskets so possession is never out of view); per-camera
frame offsets put everything on the first (reference) camera's timeline.

manifest entry: {"name": "c2a354fe_333_14", "videos": "clips/NL.mp4:-3,clips/NR.mp4:-1",
                 "team_a": "BLACK", "team_b": "WHITE"}

Keys:  <-/->  1 frame   j/k  10   a/b  possession from here   l  loose
       p  PASS (at the throw)     t  TURNOVER (whenever a<->b changes)
       z  undo last mark          [ ]  previous / next clip
GT JSON per clip: data/gt_events/<name>.json  {"breakpoints":[{frame,holder}],
                                               "moments":[{frame,type}]}

  python scripts/annotate_gt_events.py --manifest configs/gt_windows.json   # :8004
  python scripts/annotate_gt_events.py --video clip.mp4                     # single-clip
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, JSONResponse

UI = """<!doctype html><html><head><meta charset=utf-8><title>GT events</title><style>
body{margin:0;background:#111;color:#ddd;font:15px system-ui;display:flex;flex-direction:column;align-items:center}
#bar{padding:8px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}.pill{background:#262626;border-radius:10px;padding:3px 10px}
img{max-width:97vw;max-height:70vh;border:1px solid #333}#tl{width:97vw;height:26px;border:1px solid #333;margin:6px 0}
kbd{background:#333;border-radius:4px;padding:1px 5px}#help{padding:6px;color:#999;font-size:13px}
b.A{color:#4cf}b.B{color:#6f6}b.L{color:#aaa}
button{font-size:14px;padding:4px 10px;border-radius:8px;border:1px solid #444;background:#222;color:#ddd;cursor:pointer}</style>
</head><body>
<div id=bar><button onclick="clipTo(cur-1)">&#8678; prev clip</button>
<span class=pill id=clip>clip</span>
<button onclick="clipTo(cur+1)">next clip &#8680;</button>
<span class=pill id=pos>0</span><span class=pill>holder: <b id=holder class=L>?</b></span>
<span class=pill id=marks>0 marks</span></div>
<img id=im><canvas id=tl></canvas>
<div id=help><kbd>&larr;/&rarr;</kbd>1 <kbd>j/k</kbd>10 &nbsp; possession from here:
 <kbd>a</kbd><b class=A id=ta>A</b> <kbd>b</kbd><b class=B id=tb>B</b> <kbd>l</kbd>loose &nbsp;
 moment: <kbd>p</kbd>pass(at the throw) <kbd>t</kbd>turnover(when a&harr;b) &nbsp; <kbd>z</kbd>undo
 &nbsp; <kbd>[</kbd>/<kbd>]</kbd>prev/next clip</div>
<script>
let i=0,N=1,cur=0,clips=[],gt={breakpoints:[],moments:[]};
const im=document.getElementById('im'),tl=document.getElementById('tl'),ctx=tl.getContext('2d');
const COL={A:'#4cf',B:'#6f6',loose:'#555'};
function holderAt(f){let h='?';for(const b of gt.breakpoints)if(b.frame<=f)h=b.holder;return h;}
function draw(){tl.width=tl.clientWidth;tl.height=26;ctx.fillStyle='#222';ctx.fillRect(0,0,tl.width,26);
 for(let k=0;k<gt.breakpoints.length;k++){const b=gt.breakpoints[k];
  const x0=b.frame/N*tl.width,x1=(k+1<gt.breakpoints.length?gt.breakpoints[k+1].frame:N)/N*tl.width;
  ctx.fillStyle=COL[b.holder]||'#555';ctx.fillRect(x0,4,x1-x0,18);}
 for(const m of gt.moments){ctx.fillStyle=m.type==='pass'?'#fff':'#f55';ctx.fillRect(m.frame/N*tl.width-1,0,3,26);}
 ctx.fillStyle='#fd6';ctx.fillRect(i/N*tl.width-1,0,2,26);
 document.getElementById('pos').textContent=i+'/'+(N-1);
 const h=holderAt(i);const el=document.getElementById('holder');el.textContent=h;el.className=h==='A'?'A':h==='B'?'B':'L';
 document.getElementById('marks').textContent=gt.breakpoints.length+' holder marks, '+gt.moments.length+' moments';}
function show(){im.src='/frame/'+i+'?c='+cur;draw();}
function applyState(d){N=d.n_frames;gt=d.gt;cur=d.cur;clips=d.clips;
 document.getElementById('clip').textContent=(cur+1)+'/'+clips.length+'  '+clips[cur];
 document.getElementById('ta').textContent=d.team_a;document.getElementById('tb').textContent=d.team_b;}
async function clipTo(k){if(k<0||k>=clips.length)return;
 document.getElementById('clip').textContent='loading...';
 const d=await(await fetch('/api/clip/'+k)).json();applyState(d);i=0;show();}
async function mark(kind,val){const r=await fetch('/api/mark',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({kind,frame:i,val})});gt=await r.json();draw();}
document.addEventListener('keydown',e=>{
 if(e.key==='ArrowRight'&&i<N-1)i++;else if(e.key==='ArrowLeft'&&i>0)i--;
 else if(e.key==='k')i=Math.min(N-1,i+10);else if(e.key==='j')i=Math.max(0,i-10);
 else if(e.key==='a')return mark('breakpoint',{holder:'A'});else if(e.key==='b')return mark('breakpoint',{holder:'B'});
 else if(e.key==='l')return mark('breakpoint',{holder:'loose'});
 else if(e.key==='p')return mark('moment',{type:'pass'});else if(e.key==='t')return mark('moment',{type:'turnover'});
 else if(e.key==='z')return mark('undo',null);
 else if(e.key===']')return clipTo(cur+1);else if(e.key==='[')return clipTo(cur-1);
 else return; show();});
tl.onclick=e=>{i=Math.floor(e.offsetX/tl.width*N);show();};
fetch('/api/state').then(r=>r.json()).then(d=>{applyState(d);show();});
</script></body></html>"""

app = FastAPI(title="gt-event labeller")
STATE: dict = {"clips": [], "cur": 0, "ops": {}}


def _encode(videos_spec: str) -> tuple[list[bytes], float]:
    """Pre-encode side-by-side composite JPEGs (cv2.VideoCapture is not thread-safe;
    serving raw decoders crashed live under fast stepping)."""
    import cv2
    import numpy as np
    sources = []
    for spec in videos_spec.split(","):
        path, _, off = spec.partition(":")
        sources.append((path, int(off or 0)))
    caps = [cv2.VideoCapture(p) for p, _ in sources]
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
    # prime: each camera shows its frame max(0, off) at reference frame 0; negative
    # offsets HOLD their first frame until the reference catches up (cam fc = r + off)
    last = []
    for cap, (_, off) in zip(caps, sources):
        for _ in range(max(0, off)):
            cap.read()
        ok, img = cap.read()
        if not ok:
            for c in caps:
                c.release()
            return [], fps
        last.append(img)

    def compose():
        row = [cv2.resize(im, (960, 540)) for im in last] if len(last) > 1 else last
        return cv2.imencode(".jpg", np.hstack(row) if len(row) > 1 else row[0],
                            [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()

    jpegs = [compose()]
    r = 1
    while r < 4000:
        ref_alive = True
        for k, (cap, (_, off)) in enumerate(zip(caps, sources)):
            if r + off >= 1:                       # this camera advances at this ref step
                ok, img = cap.read()
                if ok:
                    last[k] = img                  # else: hold last frame (cam ended early)
                elif k == 0:
                    ref_alive = False              # reference camera defines clip length
        if not ref_alive:
            break
        jpegs.append(compose())
        r += 1
    for cap in caps:
        cap.release()
    return jpegs, fps


def _clip() -> dict:
    c = STATE["clips"][STATE["cur"]]
    if c.get("jpegs") is None:
        print(f"[encode] {c['name']} ...", flush=True)
        c["jpegs"], c["fps"] = _encode(c["videos"])
        print(f"[encode] {c['name']}: {len(c['jpegs'])} frames", flush=True)
    return c


def _gt_path(c: dict) -> Path:
    return Path(f"data/gt_events/{c['name']}.json")


def _gt(c: dict) -> dict:
    p = _gt_path(c)
    if p.exists():
        return json.loads(p.read_text())
    return {"video": c["videos"], "n_frames": len(c["jpegs"] or []), "fps": c.get("fps", 30.0),
            "breakpoints": [], "moments": []}


def _save(c: dict, gt: dict) -> None:
    p = _gt_path(c)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(gt, indent=1))
    os.replace(tmp, p)


def _state() -> dict:
    c = _clip()
    return {"n_frames": len(c["jpegs"]), "gt": _gt(c), "cur": STATE["cur"],
            "clips": [x["name"] for x in STATE["clips"]],
            "team_a": c.get("team_a", "A"), "team_b": c.get("team_b", "B")}


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
    kind, f, val = body.get("kind"), int(body.get("frame", 0)), body.get("val") or {}
    ops = STATE["ops"].setdefault(STATE["cur"], [])
    if kind == "breakpoint" and val.get("holder") in ("A", "B", "loose"):
        curh = "loose"
        for b in sorted(gt["breakpoints"], key=lambda b: b["frame"]):
            if b["frame"] <= f:
                curh = b["holder"]
        if curh == val["holder"]:
            return gt                              # no-op: holder unchanged (anti key-spam)
        gt["breakpoints"] = [b for b in gt["breakpoints"] if b["frame"] != f] \
            + [{"frame": f, "holder": val["holder"]}]
        gt["breakpoints"].sort(key=lambda b: b["frame"])
        ops.append(("breakpoint", f))
    elif kind == "moment" and val.get("type") in ("pass", "turnover"):
        if any(abs(m["frame"] - f) <= 5 and m["type"] == val["type"] for m in gt["moments"]):
            return gt                              # duplicate press within 5 frames: ignore
        gt["moments"].append({"frame": f, "type": val["type"]})
        gt["moments"].sort(key=lambda m: m["frame"])
        ops.append(("moment", f, val["type"]))
    elif kind == "undo" and ops:
        op = ops.pop()
        if op[0] == "breakpoint":
            gt["breakpoints"] = [b for b in gt["breakpoints"] if b["frame"] != op[1]]
        else:
            for k in range(len(gt["moments"]) - 1, -1, -1):
                if gt["moments"][k]["frame"] == op[1] and gt["moments"][k]["type"] == op[2]:
                    del gt["moments"][k]
                    break
    else:
        return JSONResponse({"error": "bad mark"}, status_code=422)
    _save(c, gt)
    return gt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None, help="JSON list of {name, videos, team_a, team_b}")
    ap.add_argument("--video", default=None, help="single-clip mode: 'path[:off],path[:off]'")
    ap.add_argument("--team-a", default="A")
    ap.add_argument("--team-b", default="B")
    ap.add_argument("--port", type=int, default=8004)
    a = ap.parse_args()
    if a.manifest:
        clips = json.loads(Path(a.manifest).read_text())
    elif a.video:
        clips = [{"name": Path(a.video.split(",")[0].split(":")[0]).stem,
                  "videos": a.video, "team_a": a.team_a, "team_b": a.team_b}]
    else:
        raise SystemExit("need --manifest or --video")
    for c in clips:
        c.setdefault("jpegs", None)
    STATE["clips"] = clips
    _clip()                                        # encode the first clip up-front
    print(f"GT labeller: {len(clips)} clip(s); marks -> data/gt_events/<name>.json")
    uvicorn.run(app, host="127.0.0.1", port=a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
