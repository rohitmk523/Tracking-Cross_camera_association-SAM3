#!/usr/bin/env python3
"""Per-player ground-truth annotation (operator design, :8006).

2x2 SYNCED camera frames for a whole window. Pick a player; every detection box is
GREEN; click the box that is your player on each angle -> turns YELLOW (click again
to deselect; one box per angle per player). Not visible in an angle = select nothing
there. 'a' approves the frame for the current player and advances (your selections
AUTO-CARRY to the next frame by overlap, so mostly you just press 'a' and correct).
Arrow keys move freely. Everything saves continuously to data/gt_players/.

Boxes come from the CACHED DETECTIONS (version-independent), so this ground truth
scores every past and future tracking version by IoU matching.

  python scripts/annotate_player.py --game e6fba750 --tag 44_60 --start 44 --port 8006
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ANGLES = ("FL", "FR", "NL", "NR")
TILE_W, TILE_H = 800, 450          # 2x2 -> 1600x900 canvas
GREEN, YELLOW = (80, 200, 80), (0, 220, 255)
MIN_SCORE = 0.35


def load_dets(game, tag):
    """cached detections per angle -> {ang: {frame: [box,...]}} (person classes)."""
    out = {}
    for ang in ANGLES:
        p = REPO / f"runs/dets_cache/{game}_{ang}_{tag}_small_1280_t0.25.dets.npz"
        z = np.load(p)
        m = {}
        for b, s, c, f in zip(z["boxes"], z["scores"], z["classes"], z["frame_idx"]):
            if int(c) in (0, 1) and float(s) >= MIN_SCORE:
                m.setdefault(int(f), []).append([float(v) for v in b])
        out[ang] = m
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua


class App:
    def __init__(self, a):
        self.game, self.tag = a.game, a.tag
        self.dets = load_dets(a.game, a.tag)
        self.caps = {ang: cv2.VideoCapture(str(REPO / f"data/clips/{a.game}_{ang}_{a.tag}.mp4"))
                     for ang in ANGLES}
        self.pos = {ang: -1 for ang in ANGLES}
        # audio-sync offsets onto the FL timeline (same rule as fusion)
        from uball_cc.fusion.audiosync import audio_offset_seconds
        self.off = {"FL": 0}
        ref = str(REPO / f"data/clips/{a.game}_FL_{a.tag}.mp4")
        for ang in ("FR", "NL", "NR"):
            off_s, _ = audio_offset_seconds(ref, str(REPO / f"data/clips/{a.game}_{ang}_{a.tag}.mp4"))
            self.off[ang] = int(round(off_s * 29.97))
        self.n = min(int(self.caps[ang].get(cv2.CAP_PROP_FRAME_COUNT)) - max(0, self.off[ang])
                     for ang in ANGLES)
        # first frame at which EVERY camera has synced coverage (negative offsets
        # mean the earliest ref frames precede those cameras' clips)
        self.first = max(0, max(-o for o in self.off.values()))
        self.gt_path = REPO / f"data/gt_players/{a.game}_{a.tag}.json"
        self.gt_path.parent.mkdir(parents=True, exist_ok=True)
        self.gt = json.loads(self.gt_path.read_text()) if self.gt_path.exists() else {}
        self.lock = threading.Lock()

    def _grab(self, ang, f):
        fc = f + self.off[ang]
        cap = self.caps[ang]
        if fc != self.pos[ang] + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fc))
        ok, img = cap.read()
        self.pos[ang] = fc
        return img if ok else np.zeros((1080, 1920, 3), np.uint8)

    def sel(self, player):
        return self.gt.setdefault(player, {"frames": {}, "approved": {}})

    def dets_at(self, ang, f):
        return self.dets[ang].get(f + self.off[ang], [])

    def render(self, f, player):
        s = self.sel(player)["frames"].get(str(f), {})
        canvas = np.zeros((900, 1600, 3), np.uint8)
        for k, ang in enumerate(ANGLES):
            img = self._grab(ang, f)
            ih, iw = img.shape[:2]
            chosen = s.get(ang)
            for bi, b in enumerate(self.dets_at(ang, f)):
                col = YELLOW if (chosen is not None and bi == chosen) else GREEN
                th = 6 if col == YELLOW else 2
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, th)
            tile = cv2.resize(img, (TILE_W, TILE_H))
            r, c = divmod(k, 2)
            canvas[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = tile
            cv2.putText(canvas, ang, (c * TILE_W + 10, r * TILE_H + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return buf.tobytes()

    def click(self, f, player, x, y):
        c, r = int(x // TILE_W), int(y // TILE_H)
        ang = ANGLES[r * 2 + c]
        img_x = (x - c * TILE_W) * 1920 / TILE_W
        img_y = (y - r * TILE_H) * 1080 / TILE_H
        hits = [(bi, b) for bi, b in enumerate(self.dets_at(ang, f))
                if b[0] <= img_x <= b[2] and b[1] <= img_y <= b[3]]
        if not hits:
            return
        bi = min(hits, key=lambda h: (h[1][2] - h[1][0]) * (h[1][3] - h[1][1]))[0]
        with self.lock:
            fr = self.sel(player)["frames"].setdefault(str(f), {})
            if fr.get(ang) == bi:
                del fr[ang]                          # toggle off
            else:
                fr[ang] = bi
            self.save()

    def approve(self, f, player):
        with self.lock:
            s = self.sel(player)
            s["approved"][str(f)] = True
            # AUTO-CARRY to f+1 by IoU
            cur = s["frames"].get(str(f), {})
            nxt = s["frames"].setdefault(str(f + 1), {})
            for ang, bi in cur.items():
                if ang in nxt:
                    continue
                boxes_now = self.dets_at(ang, f)
                boxes_next = self.dets_at(ang, f + 1)
                if bi >= len(boxes_now) or not boxes_next:
                    continue
                ious = [iou(boxes_now[bi], b2) for b2 in boxes_next]
                bj = int(np.argmax(ious))
                if ious[bj] >= 0.3:
                    nxt[ang] = bj
            self.save()

    def save(self):
        self.gt_path.write_text(json.dumps(self.gt))


PAGE = """<!doctype html><html><head><title>player GT</title><style>
body{background:#111;color:#eee;font-family:sans-serif;margin:10px}
#img{cursor:crosshair;border:1px solid #333;width:100%;max-width:1600px;height:auto;display:block}
.b{background:#333;color:#eee;border:1px solid #555;padding:6px 12px;margin:2px;cursor:pointer}
.active{background:#664}
</style></head><body>
<div>
 <span id="who"></span> | frame <span id="f">0</span>/<span id="n">?</span>
 | approved <span id="ap">0</span>
 | keys: <b>a</b>=approve+next &nbsp;<b>&larr;/&rarr;</b>=move &nbsp;<b>Shift+&larr;/&rarr;</b>=&plusmn;10
</div>
<div id="players"></div>
<img id="img"/>
<script>
let f=0, n=0, first=0, player=null;
async function state(){const r=await fetch('/state');const s=await r.json();n=s.n;
 first=s.first||0; if(f<first){f=first;}
 document.getElementById('n').textContent=n;
 const div=document.getElementById('players');div.innerHTML='';
 s.players.forEach(p=>{const b=document.createElement('button');b.className='b'+(p===s.player?' active':'');
  b.textContent=p+' ('+(s.approved[p]||0)+')';b.onclick=()=>{fetch('/player/'+encodeURIComponent(p)).then(()=>{player=p;state();load();});};
  div.appendChild(b);});
 player=s.player;document.getElementById('who').textContent=player;
 document.getElementById('ap').textContent=s.approved[player]||0;}
function load(){document.getElementById('img').src='/frame/'+f+'?t='+Date.now();
 document.getElementById('f').textContent=f;}
document.getElementById('img').onclick=async e=>{
 const r=e.target.getBoundingClientRect();
 const sx=1600/r.width, sy=900/r.height;
 await fetch('/click',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({f:f,x:(e.clientX-r.left)*sx,y:(e.clientY-r.top)*sy})});load();};
document.onkeydown=async e=>{
 if(e.key==='a'){await fetch('/approve/'+f,{method:'POST'});f=Math.min(f+1,n-1);load();state();}
 else if(e.key==='ArrowRight'){f=Math.min(f+(e.shiftKey?10:1),n-1);load();}
 else if(e.key==='ArrowLeft'){f=Math.max(f-(e.shiftKey?10:1),first);load();}};
state();load();
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--players", default=None,
                    help="comma list; default: roster numbers from plays snapshot + extra slots")
    ap.add_argument("--port", type=int, default=8006)
    a = ap.parse_args()

    app_state = App(a)
    if a.players:
        players = a.players.split(",")
    else:
        snap = REPO / "data/plays/windows_snapshot.json"
        players = []
        if snap.exists():
            d = json.loads(snap.read_text())
            players = [f"#{k}" for k in sorted(
                d.get(f"roster_{a.game}", {}), key=lambda x: int(x))]
        players += [f"extra_{i}" for i in range(1, 4)] + ["ref_1", "ref_2", "ref_3"]
    current = {"player": players[0]}

    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    import uvicorn

    api = FastAPI()

    @api.get("/")
    def index():
        return HTMLResponse(PAGE)

    @api.get("/state")
    def state():
        approved = {p: len(app_state.gt.get(p, {}).get("approved", {})) for p in players}
        return JSONResponse({"n": app_state.n, "first": app_state.first,
                             "players": players,
                             "player": current["player"], "approved": approved})

    @api.get("/player/{name}")
    def set_player(name: str):
        current["player"] = name
        return JSONResponse({"ok": True})

    @api.get("/frame/{f}")
    def frame(f: int):
        return Response(app_state.render(f, current["player"]), media_type="image/jpeg")

    @api.post("/click")
    async def click(req: Request):
        d = await req.json()
        app_state.click(int(d["f"]), current["player"], float(d["x"]), float(d["y"]))
        return JSONResponse({"ok": True})

    @api.post("/approve/{f}")
    def approve(f: int):
        app_state.approve(f, current["player"])
        return JSONResponse({"ok": True})

    print(f"player-GT tool: http://localhost:{a.port}  "
          f"({app_state.n} frames, players: {', '.join(players)})")
    uvicorn.run(api, host="0.0.0.0", port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
