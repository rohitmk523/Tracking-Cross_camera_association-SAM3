#!/usr/bin/env python3
"""Jersey-NUMBER annotation tool (docs/05): label each large player crop with its number.

Crops come from scripts/extract_jersey_crops.py. The operator types the number (or marks
'none'/'unclear' for occluded/back-turned/blurred crops -- expected to be many on this
fisheye footage). Labels -> <pool>/labels.json, which feeds the number-recogniser training.

  python scripts/annotate_jersey.py --pool data/jersey_pool        # http://127.0.0.1:8003
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

UI = """<!doctype html><html><head><meta charset=utf-8><title>Jersey #</title><style>
body{margin:0;background:#111;color:#ddd;font:15px system-ui;height:100vh;display:flex;flex-direction:column;align-items:center}
#bar{padding:8px;display:flex;gap:10px;align-items:center}.pill{background:#262626;border-radius:10px;padding:3px 10px}
#wrap{flex:1;display:flex;align-items:center}img{max-height:70vh;border:1px solid #333;background:#000}
input{font-size:22px;width:90px;text-align:center;padding:6px;border-radius:8px;border:1px solid #444;background:#1a1a1a;color:#fff}
kbd{background:#333;border-radius:4px;padding:1px 5px}#help{padding:6px;color:#999;font-size:12px}
button{font-size:14px;padding:6px 10px;border-radius:8px;border:1px solid #444;background:#222;color:#ddd;cursor:pointer}
</style></head><body>
<div id=bar><span class=pill id=pos>-/-</span><span class=pill id=done>0 labeled</span>
<input id=num placeholder="#" autocomplete=off>
<button onclick="save(document.getElementById('num').value)">save</button>
<button onclick="save('none')">none (n)</button><button onclick="save('unclear')">unclear (u)</button></div>
<div id=wrap><img id=img></div>
<div id=help><kbd>0-9</kbd>type number <kbd>Enter</kbd>save+next <kbd>n</kbd>none <kbd>u</kbd>unclear <kbd>&larr;/&rarr;</kbd>prev/next <kbd>g</kbd>next-unlabeled</div>
<script>
let i=0,total=0;
async function load(j){const r=await fetch('/api/item/'+j);const d=await r.json();i=d.idx;total=d.total;
 document.getElementById('pos').textContent=(i+1)+'/'+total;document.getElementById('done').textContent=d.done+' labeled';
 document.getElementById('img').src='/crop/'+i+'?t='+Date.now();
 document.getElementById('num').value=(d.label&&d.label!=='none'&&d.label!=='unclear')?d.label:'';document.getElementById('num').focus();}
async function save(v){await fetch('/api/save/'+i,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label:String(v).trim()})});
 if(i<total-1)load(i+1);else load(i);}
document.addEventListener('keydown',e=>{const a=document.activeElement===document.getElementById('num');
 if(e.key==='Enter'){save(document.getElementById('num').value);e.preventDefault();}
 else if(e.key==='n'&&!a)save('none');else if(e.key==='u'&&!a)save('unclear');
 else if(e.key==='ArrowRight'){if(i<total-1)load(i+1);}else if(e.key==='ArrowLeft'){if(i>0)load(i-1);}
 else if(e.key==='g'&&!a)fetch('/api/next_unlabeled?after='+i).then(r=>r.json()).then(d=>{if(d.idx>=0)load(d.idx);});});
load(0);
</script></body></html>"""

app = FastAPI(title="jersey-number annotator")
POOL = Path("data/jersey_pool")


def _items():
    return json.loads((POOL / "items.json").read_text())


def _labels():
    p = POOL / "labels.json"
    return json.loads(p.read_text()) if p.exists() else {}


@app.get("/", response_class=HTMLResponse)
def index():
    return UI


@app.get("/crop/{i}")
def crop(i: int):
    return FileResponse(str(POOL / "crops" / _items()[i]["crop"]))


@app.get("/api/item/{i}")
def item(i: int):
    items, labels = _items(), _labels()
    return {"idx": i, "total": len(items), "done": len(labels),
            "label": labels.get(items[i]["crop"]), "crop": items[i]["crop"]}


@app.post("/api/save/{i}")
async def save(i: int, body: dict):
    items, labels = _items(), _labels()
    labels[items[i]["crop"]] = body.get("label", "")
    (POOL / "labels.json").write_text(json.dumps(labels))
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
