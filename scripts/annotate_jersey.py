#!/usr/bin/env python3
"""Jersey-NUMBER annotation tool (docs/05): draw a tight box around the number AND type it.

The box does double duty: it trains a number-LOCALIZER (the stock e6 one doesn't generalize)
and gives the recogniser a clean number crop. Draw the box on the visible number (front number
sits low-right of the team name; back number is centred). Mark 'none'/'unclear' when no number
is legible (common on this fisheye footage). Labels -> <pool>/labels.json:
  {crop: {"number": "11", "box": [x1,y1,x2,y2]}}  (box normalised 0..1 to the crop; null if none)

  python scripts/annotate_jersey.py --pool data/jersey_pool        # http://127.0.0.1:8003
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

UI = """<!doctype html><html><head><meta charset=utf-8><title>Jersey #</title><style>
body{margin:0;background:#111;color:#ddd;font:15px system-ui;height:100vh;display:flex;flex-direction:column;align-items:center}
#bar{padding:8px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}.pill{background:#262626;border-radius:10px;padding:3px 10px}
#wrap{flex:1;display:flex;align-items:center}canvas{border:1px solid #333;background:#000;cursor:crosshair}
input{font-size:22px;width:90px;text-align:center;padding:6px;border-radius:8px;border:1px solid #444;background:#1a1a1a;color:#fff}
kbd{background:#333;border-radius:4px;padding:1px 5px}#help{padding:6px;color:#999;font-size:12px}
button{font-size:14px;padding:6px 10px;border-radius:8px;border:1px solid #444;background:#222;color:#ddd;cursor:pointer}
b#need{color:#fd6}</style></head><body>
<div id=bar><span class=pill id=pos>-/-</span><span class=pill id=done>0 labeled</span>
<input id=num placeholder="#" autocomplete=off>
<span class=pill>box: <b id=need>draw one</b></span>
<button onclick="save(num.value)">save (Enter)</button>
<button onclick="save('none')">none (n)</button><button onclick="save('unclear')">unclear (u)</button></div>
<div id=wrap><canvas id=cv></canvas></div>
<div id=help><kbd>drag</kbd>box around the number <kbd>0-9</kbd>type it <kbd>Enter</kbd>save+next
 <kbd>n</kbd>none <kbd>u</kbd>unclear <kbd>z</kbd>undo box <kbd>c</kbd>clear box <kbd>&larr;/&rarr;</kbd>prev/next <kbd>g</kbd>next-unlabeled</div>
<script>
const cv=document.getElementById('cv'),ctx=cv.getContext('2d'),num=document.getElementById('num');
let i=0,total=0,img=new Image(),box=null,drag=null,sc=1,ox=0,oy=0,iw=0,ih=0,boxHist=[];
function draw(){ctx.clearRect(0,0,cv.width,cv.height);ctx.drawImage(img,ox,oy,iw*sc,ih*sc);
 const b=drag||box; if(b){ctx.lineWidth=2;ctx.strokeStyle=drag?'#3af':'#3f6';ctx.setLineDash(drag?[5,3]:[]);
  ctx.strokeRect(ox+b.x1*iw*sc,oy+b.y1*ih*sc,(b.x2-b.x1)*iw*sc,(b.y2-b.y1)*ih*sc);ctx.setLineDash([]);}
 document.getElementById('need').textContent=box?'set ✓':'draw one';document.getElementById('need').style.color=box?'#6f6':'#fd6';}
function toImg(e){const r=cv.getBoundingClientRect();return [((e.clientX-r.left)-ox)/(iw*sc),((e.clientY-r.top)-oy)/(ih*sc)];}
cv.onmousedown=e=>{const[x,y]=toImg(e);drag={x1:x,y1:y,x2:x,y2:y};};
cv.onmousemove=e=>{if(!drag)return;const[x,y]=toImg(e);drag.x2=x;drag.y2=y;draw();};
cv.onmouseup=e=>{if(!drag)return;let{x1,y1,x2,y2}=drag;drag=null;
 if(Math.abs(x2-x1)>0.02&&Math.abs(y2-y1)>0.02){boxHist.push(box);box={x1:Math.min(x1,x2),y1:Math.min(y1,y2),x2:Math.max(x1,x2),y2:Math.max(y1,y2)};}draw();};
async function load(j){const d=await(await fetch('/api/item/'+j)).json();i=d.idx;total=d.total;
 document.getElementById('pos').textContent=(i+1)+'/'+total;document.getElementById('done').textContent=d.done+' labeled';
 const b=d.label&&d.label.box; box=b?{x1:b[0],y1:b[1],x2:b[2],y2:b[3]}:null; boxHist=[];
 num.value=(d.label&&d.label.number&&!['none','unclear'].includes(d.label.number))?d.label.number:'';
 img=new Image();img.onload=()=>{const MH=620;sc=Math.min(MH/img.height,2.5);iw=img.width;ih=img.height;
  cv.width=iw*sc;cv.height=ih*sc;ox=0;oy=0;draw();num.focus();};img.src='/crop/'+i+'?t='+Date.now();}
async function save(v){v=String(v).trim();
 const payload={number:v,box:(box&&!['none','unclear'].includes(v))?[box.x1,box.y1,box.x2,box.y2]:null};
 const r=await fetch('/api/save/'+i,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
 if(!r.ok){const e=await r.json().catch(()=>({}));alert(e.error||'save rejected');return;}
 if(i<total-1)load(i+1);else load(i);}
document.addEventListener('keydown',e=>{const a=document.activeElement===num;
 if(e.key==='Enter'){save(num.value);e.preventDefault();}
 else if(e.key==='n'&&!a)save('none');else if(e.key==='u'&&!a)save('unclear');
 else if(e.key==='c'&&!a){boxHist.push(box);box=null;draw();}
 else if(e.key==='z'&&!a){if(boxHist.length){box=boxHist.pop();draw();}}
 else if(e.key==='ArrowRight'){if(i<total-1)load(i+1);}else if(e.key==='ArrowLeft'){if(i>0)load(i-1);}
 else if(e.key==='g'&&!a)fetch('/api/next_unlabeled?after='+i).then(r=>r.json()).then(d=>{if(d.idx>=0)load(d.idx);});});
fetch('/api/next_unlabeled?after=-1').then(r=>r.json()).then(d=>load(d.idx>=0?d.idx:0));  // resume where we left off
</script></body></html>"""

app = FastAPI(title="jersey-number annotator")
POOL = Path("data/jersey_pool")


def _items():
    return json.loads((POOL / "items.json").read_text())


def _labels():
    p = POOL / "labels.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    # migrate old string labels (number-only, no box) -> dict form
    return {k: (v if isinstance(v, dict) else {"number": v, "box": None}) for k, v in raw.items()}


@app.get("/", response_class=HTMLResponse)
def index():
    return UI


@app.get("/crop/{i}")
def crop(i: int):
    return FileResponse(str(POOL / "crops" / _items()[i]["crop"]))


@app.get("/api/item/{i}")
def item(i: int):
    items, labels = _items(), _labels()
    return {"idx": i, "total": len(items), "done": len(labels), "label": labels.get(items[i]["crop"])}


def _write_labels(labels: dict) -> None:
    """Atomic replace + rolling backup — labels.json is the SOLE (gitignored) copy of
    operator hours; a crash mid-write must never truncate it (2026-07-02 audit)."""
    p, tmp = POOL / "labels.json", POOL / "labels.json.tmp"
    tmp.write_text(json.dumps(labels))
    os.replace(tmp, p)
    if len(labels) % 20 == 0:
        shutil.copy(p, POOL / "labels.bak.json")


@app.post("/api/save/{i}")
async def save(i: int, body: dict):
    num, box = str(body.get("number", "")).strip(), body.get("box")
    if num in ("none", "unclear"):
        box = None
    elif num.isdigit() and len(num) <= 2:
        if not (isinstance(box, list) and len(box) == 4):
            return JSONResponse({"error": "numeric label needs a number box — draw one first"},
                                status_code=422)
    else:
        return JSONResponse({"error": f"invalid label {num!r} — digits 0-99, or none/unclear"},
                            status_code=422)
    items, labels = _items(), _labels()
    labels[items[i]["crop"]] = {"number": num, "box": box}
    _write_labels(labels)
    return JSONResponse({"done": len(labels)})


@app.get("/api/next_unlabeled")
def next_unlabeled(after: int = -1):
    items, labels = _items(), _labels()
    for j in range(after + 1, len(items)):
        if items[j]["crop"] not in labels:
            return {"idx": j}
    return {"idx": -1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/jersey_pool")
    ap.add_argument("--port", type=int, default=8003)
    a = ap.parse_args()
    global POOL
    POOL = Path(a.pool)
    uvicorn.run(app, host="127.0.0.1", port=a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
