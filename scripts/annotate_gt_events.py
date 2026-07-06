#!/usr/bin/env python3
"""Ground-truth EVENT labeller (audit P1): step through one camera's clip and record who has
possession + the pass/turnover moments. ~20 min of operator time per 12s window turns every
future ball/possession/event claim into a MEASURED precision/recall instead of an eyeball.

Keys:  <-/->  step 1 frame   j/k  step 10   a/b  possession Team A(cyan)/B(green) from here
       l  ball loose from here   p  PASS at this frame   t  TURNOVER at this frame
       z  undo last mark         (marks save atomically on every key)

GT JSON: {"video", "n_frames", "fps",
          "breakpoints": [{"frame", "holder": "A"|"B"|"loose"}],   # holder from that frame on
          "moments":     [{"frame", "type": "pass"|"turnover"}]}

  python scripts/annotate_gt_events.py --video data/clips/e6fba750_FL_47_12.mp4 \
      --out data/gt_events/e6_47_12.json          # http://127.0.0.1:8004
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
#bar{padding:8px;display:flex;gap:12px;align-items:center}.pill{background:#262626;border-radius:10px;padding:3px 10px}
img{max-width:96vw;max-height:74vh;border:1px solid #333}#tl{width:96vw;height:26px;border:1px solid #333;margin:6px 0}
kbd{background:#333;border-radius:4px;padding:1px 5px}#help{padding:6px;color:#999;font-size:13px}
b.A{color:#4cf}b.B{color:#6f6}b.L{color:#aaa}</style></head><body>
<div id=bar><span class=pill id=pos>0</span><span class=pill>holder: <b id=holder class=L>?</b></span>
<span class=pill id=marks>0 breakpoints, 0 moments</span></div>
<img id=im><canvas id=tl></canvas>
<div id=help><kbd>&larr;/&rarr;</kbd>1 <kbd>j/k</kbd>10 &nbsp; possession from here: <kbd>a</kbd><b class=A>A</b>
 <kbd>b</kbd><b class=B>B</b> <kbd>l</kbd>loose &nbsp; moment: <kbd>p</kbd>pass <kbd>t</kbd>turnover &nbsp; <kbd>z</kbd>undo</div>
<script>
let i=0,N=1,gt={breakpoints:[],moments:[]};
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
 document.getElementById('marks').textContent=gt.breakpoints.length+' breakpoints, '+gt.moments.length+' moments';}
function show(){im.src='/frame/'+i;draw();}
async function mark(kind,val){const r=await fetch('/api/mark',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({kind,frame:i,val})});gt=await r.json();draw();}
document.addEventListener('keydown',e=>{
 if(e.key==='ArrowRight'&&i<N-1)i++;else if(e.key==='ArrowLeft'&&i>0)i--;
 else if(e.key==='k')i=Math.min(N-1,i+10);else if(e.key==='j')i=Math.max(0,i-10);
 else if(e.key==='a')return mark('breakpoint',{holder:'A'});else if(e.key==='b')return mark('breakpoint',{holder:'B'});
 else if(e.key==='l')return mark('breakpoint',{holder:'loose'});
 else if(e.key==='p')return mark('moment',{type:'pass'});else if(e.key==='t')return mark('moment',{type:'turnover'});
 else if(e.key==='z')return mark('undo',null);else return; show();});
tl.onclick=e=>{i=Math.floor(e.offsetX/tl.width*N);show();};
fetch('/api/state').then(r=>r.json()).then(d=>{N=d.n_frames;gt=d.gt;show();});
</script></body></html>"""

app = FastAPI(title="gt-event labeller")
STATE = {"video": "", "out": Path(), "n": 0, "fps": 30.0, "cap": None, "ops": []}


def _gt() -> dict:
    if STATE["out"].exists():
        return json.loads(STATE["out"].read_text())
    return {"video": STATE["video"], "n_frames": STATE["n"], "fps": STATE["fps"],
            "breakpoints": [], "moments": []}


def _save(gt: dict) -> None:
    tmp = STATE["out"].with_suffix(".tmp")
    tmp.write_text(json.dumps(gt, indent=1))
    os.replace(tmp, STATE["out"])


@app.get("/", response_class=HTMLResponse)
def index():
    return UI


@app.get("/frame/{i}")
def frame(i: int):
    # frames are pre-encoded at startup: cv2.VideoCapture is NOT thread-safe and fast
    # stepping fires concurrent requests (libavcodec async_lock crash, seen live)
    jpegs = STATE["jpegs"]
    return Response(jpegs[max(0, min(i, len(jpegs) - 1))], media_type="image/jpeg")


@app.get("/api/state")
def state():
    return {"n_frames": STATE["n"], "gt": _gt()}


@app.post("/api/mark")
async def mark(body: dict):
    gt = _gt()
    kind, f, val = body.get("kind"), int(body.get("frame", 0)), body.get("val") or {}
    if kind == "breakpoint" and val.get("holder") in ("A", "B", "loose"):
        # NO-OP if the holder is already this value at frame f — pressing the key every
        # frame (a natural operator instinct) must not spam hundreds of marks.
        cur = "loose"
        for b in sorted(gt["breakpoints"], key=lambda b: b["frame"]):
            if b["frame"] <= f:
                cur = b["holder"]
        if cur == val["holder"]:
            return gt
        gt["breakpoints"] = [b for b in gt["breakpoints"] if b["frame"] != f] \
            + [{"frame": f, "holder": val["holder"]}]
        gt["breakpoints"].sort(key=lambda b: b["frame"])
        STATE["ops"].append(("breakpoint", f))
    elif kind == "moment" and val.get("type") in ("pass", "turnover"):
        if any(abs(m["frame"] - f) <= 5 and m["type"] == val["type"] for m in gt["moments"]):
            return gt                              # duplicate press within 5 frames: ignore
        gt["moments"].append({"frame": f, "type": val["type"]})
        gt["moments"].sort(key=lambda m: m["frame"])
        STATE["ops"].append(("moment", f, val["type"]))
    elif kind == "undo" and STATE["ops"]:
        op = STATE["ops"].pop()
        if op[0] == "breakpoint":
            gt["breakpoints"] = [b for b in gt["breakpoints"] if b["frame"] != op[1]]
        else:
            for k in range(len(gt["moments"]) - 1, -1, -1):
                if gt["moments"][k]["frame"] == op[1] and gt["moments"][k]["type"] == op[2]:
                    del gt["moments"][k]
                    break
    else:
        return JSONResponse({"error": "bad mark"}, status_code=422)
    _save(gt)
    return gt


def main() -> int:
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None, help="default: data/gt_events/<clipstem>.json")
    ap.add_argument("--port", type=int, default=8004)
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    jpegs = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        jpegs.append(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes())
    cap.release()
    print(f"pre-encoded {len(jpegs)} frames ({sum(map(len, jpegs)) // 1_000_000} MB in memory)")
    STATE.update(video=a.video, jpegs=jpegs, n=len(jpegs), fps=fps,
                 out=Path(a.out or f"data/gt_events/{Path(a.video).stem}.json"))
    STATE["out"].parent.mkdir(parents=True, exist_ok=True)
    print(f"GT labeller: {a.video} ({STATE['n']} frames) -> {STATE['out']}")
    uvicorn.run(app, host="127.0.0.1", port=a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
