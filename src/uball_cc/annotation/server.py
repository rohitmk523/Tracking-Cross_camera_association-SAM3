"""FastAPI annotator for player/referee/ball (3-class YOLO), flat pool, resumable.

  python scripts/annotate.py            # -> http://127.0.0.1:8000
  UBALL_ANNOT_POOL=data/annotate_pool python scripts/annotate.py

Classes: 0=player 1=referee 2=ball. Keys: 1/2/3 draw-class, drag=new box,
click=select, D=delete, U=undo, A=approve, S=save, N/P=next/prev, G=next pending.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image
from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[3]
POOL = Path(os.environ.get("UBALL_ANNOT_POOL", str(REPO / "data" / "annotate_pool")))
IMG_DIR = POOL / "images"
LBL_DIR = POOL / "labels"
STATE_FILE = POOL / "review_state.json"
UI_HTML = Path(__file__).resolve().parent / "ui.html"
CLASSES = ("player", "referee", "ball")


class Box(BaseModel):
    cls: int            # 0 player, 1 referee, 2 ball
    cx: float           # YOLO-normalized [0,1]
    cy: float
    w: float
    h: float


class SaveReq(BaseModel):
    boxes: list[Box]
    approve: bool


class Item:
    __slots__ = ("stem", "img", "label", "w", "h")

    def __init__(self, stem, img, label, w, h):
        self.stem, self.img, self.label, self.w, self.h = stem, img, label, w, h


def build_index() -> list[Item]:
    items = []
    for img in sorted(IMG_DIR.glob("*.jpg")):
        try:
            with Image.open(img) as im:
                w, h = im.size
        except Exception:                              # noqa: BLE001
            continue
        items.append(Item(img.stem, img, LBL_DIR / f"{img.stem}.txt", w, h))
    return items


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:                              # noqa: BLE001
            return {}
    return {}


def save_state(state: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(POOL), suffix=".json")
    os.close(fd)
    Path(tmp).write_text(json.dumps(state))
    Path(tmp).replace(STATE_FILE)


def read_boxes(label: Path) -> list[dict]:
    out = []
    if not label.exists():
        return out
    for ln in label.read_text().strip().splitlines():
        p = ln.split()
        if len(p) >= 5:
            out.append({"cls": int(p[0]), "cx": float(p[1]), "cy": float(p[2]),
                        "w": float(p[3]), "h": float(p[4])})
    return out


def write_boxes(label: Path, boxes: list[Box]) -> None:
    label.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for b in boxes:
        cx, cy, w, h = (max(0.0, min(1.0, v)) for v in (b.cx, b.cy, b.w, b.h))
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{int(b.cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label.write_text("\n".join(lines) + ("\n" if lines else ""))


app = FastAPI(title="Uball Player/Ref/Ball Annotator")
INDEX: list[Item] = []
STATE: dict = {}


@app.on_event("startup")
def _startup() -> None:
    global INDEX, STATE
    LBL_DIR.mkdir(parents=True, exist_ok=True)
    INDEX = build_index()
    STATE = load_state()
    done = sum(1 for v in STATE.values() if v.get("status") == "approved")
    print(f"pool={POOL}  indexed {len(INDEX)} images  approved={done}")


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return UI_HTML.read_text()


@app.get("/api/state")
def api_state() -> dict:
    done = sum(1 for v in STATE.values() if v.get("status") == "approved")
    return {"total": len(INDEX), "approved": done, "pending": len(INDEX) - done}


@app.get("/api/item/{idx}")
def api_item(idx: int) -> dict:
    if not (0 <= idx < len(INDEX)):
        raise HTTPException(404, "idx out of range")
    it = INDEX[idx]
    return {"idx": idx, "total": len(INDEX), "stem": it.stem, "w": it.w, "h": it.h,
            "classes": list(CLASSES), "boxes": read_boxes(it.label),
            "status": STATE.get(it.stem, {}).get("status", "pending"),
            "approved": sum(1 for v in STATE.values() if v.get("status") == "approved")}


@app.get("/img/{idx}")
def api_img(idx: int):
    if not (0 <= idx < len(INDEX)):
        raise HTTPException(404, "idx out of range")
    return FileResponse(INDEX[idx].img)


@app.post("/api/save/{idx}")
def api_save(idx: int, req: SaveReq) -> dict:
    if not (0 <= idx < len(INDEX)):
        raise HTTPException(404, "idx out of range")
    it = INDEX[idx]
    write_boxes(it.label, req.boxes)
    STATE[it.stem] = {"status": "approved" if req.approve else
                      STATE.get(it.stem, {}).get("status", "pending"), "ts": time.time()}
    save_state(STATE)
    nxt = -1
    for j in list(range(idx + 1, len(INDEX))) + list(range(0, idx + 1)):
        if STATE.get(INDEX[j].stem, {}).get("status") != "approved":
            nxt = j
            break
    done = sum(1 for v in STATE.values() if v.get("status") == "approved")
    return {"ok": True, "next_pending": nxt, "approved": done, "total": len(INDEX)}


@app.get("/api/next_pending")
def api_next_pending(after: int = -1) -> dict:
    n = len(INDEX)
    for k in range(1, n + 1):
        j = (after + k) % n
        if STATE.get(INDEX[j].stem, {}).get("status") != "approved":
            return {"idx": j}
    return {"idx": -1}


def run(host="127.0.0.1", port=8000) -> None:
    uvicorn.run(app, host=host, port=port, log_level="warning")
