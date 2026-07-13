#!/usr/bin/env python3
"""Highlights v2 — ALL FOUR angles in a 2x2 grid, the predicted player boxed
in every view (identity-track-driven), right panel: PIPELINE prediction on
top, GROUND TRUTH below.

  python scripts/render_highlights_grid.py --plays /tmp/highlight_plays_e6.json \
      --clips-dir runs/event_demo/clips_hl4 --out runs/event_demo/highlights5min_e6fba750.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from game_meta import GAME_OFFS, GAME_CHUNKS

W, H = 1600, 900
PANEL = 360
FPS = 29.97
ANGLES = ("FL", "FR", "NL", "NR")
GRID_W = W - PANEL
CELL_W, CELL_H = GRID_W // 2, H // 2


def wrap(txt, n=24):
    words, lines, cur = txt.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= n:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def block(img, y0, title, rows, color):
    cv2.putText(img, title, (GRID_W + 18, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.line(img, (GRID_W + 18, y0 + 14), (W - 18, y0 + 14), color, 1)
    y = y0 + 52
    for label, val in rows:
        cv2.putText(img, label, (GRID_W + 18, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (150, 150, 155), 1)
        y += 25
        for ln in wrap(str(val)):
            cv2.putText(img, ln, (GRID_W + 18, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (235, 235, 240), 2)
            y += 30
        y += 10
    return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="e6fba750")
    ap.add_argument("--plays", default="/tmp/highlight_plays_e6.json")
    ap.add_argument("--clips-dir", default="runs/event_demo/clips_hl4")
    ap.add_argument("--out", default="runs/event_demo/highlights5min_e6fba750.mp4")
    a = ap.parse_args()
    game = a.game
    OFFS = GAME_OFFS[game]
    sel = json.loads(Path(a.plays).read_text())
    roster = json.loads((REPO / f"data/rosters/{game}.json").read_text())
    by_num = defaultdict(list)
    for pr in roster["players"]:
        by_num[pr["num"]].append(pr)

    def stream_of(name):
        recs = [p for p in roster["players"] if p["name"] == name.rstrip("?")]
        if not recs:
            return None
        r = recs[0]
        dual = len(by_num[r["num"]]) > 1
        return f"n{r['num']}" + (("B" if r["team"] == 1 else "W") if dual else "")

    # tracks per angle over all chunks, keyed by GLOBAL angle frame
    tracks = defaultdict(lambda: defaultdict(dict))     # sid -> ang -> {gf: box}
    for tag in GAME_CHUNKS[game]:
        t0c = float(tag.split("_")[0])
        tdir = REPO / (f"runs/events_fg_{tag}" if game == "e6fba750"
                       else f"runs/events_fg_{game[:3]}_{tag}")
        for p in tdir.glob(f"{game}_{tag}__n*__*.json"):
            parts = p.stem.split("__")
            sid, ang = parts[1], parts[2]
            d = json.loads(p.read_text())["frames"]
            for fr, r in d.items():
                if r.get("present"):
                    tracks[sid][ang][round(t0c * FPS) + int(fr)] = r["box"]

    tmp = str(REPO / a.out) + ".raw.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for ci, c in enumerate(sel):
        t = c["t"]
        sid = stream_of(c["pred_player"] or "")
        caps = {}
        for ang in ANGLES:
            p = REPO / f"{a.clips_dir}/play_{t}_{ang}.mp4"
            caps[ang] = cv2.VideoCapture(str(p)) if p.exists() else None
        pred_rows = [("PLAYER", (c["pred_player"] or "?").rstrip("?"))]
        if c.get("pred_zone"):
            pred_rows.append(("POINTS", c["pred_zone"]))
        pred_rows.append(("CALL", c["pred_cls"].replace("_", " ")))
        gt_rows = [("EVENT", c["label"].replace("_", " ")),
                   ("PLAYER", c["gt_player"])]
        if c.get("gt_zone"):
            gt_rows.append(("POINTS", c["gt_zone"]))
        for fi in range(int(10 * FPS)):
            canvas = np.zeros((H, W, 3), np.uint8)
            canvas[:, GRID_W:] = (24, 24, 28)
            t_now = t - 5.5 + fi / FPS
            for gi, ang in enumerate(ANGLES):
                cap = caps[ang]
                if cap is None:
                    continue
                ok, frame = cap.read()
                if not ok:
                    continue
                fh, fw = frame.shape[:2]
                cell = cv2.resize(frame, (CELL_W, CELL_H))
                # box the predicted player in THIS angle (track-driven)
                gf = round(t_now * FPS) + OFFS[ang]
                box = tracks.get(sid, {}).get(ang, {}).get(gf) if sid else None
                if box is not None:
                    sx, sy = CELL_W / fw, CELL_H / fh
                    cv2.rectangle(cell, (int(box[0] * sx), int(box[1] * sy)),
                                  (int(box[2] * sx), int(box[3] * sy)),
                                  (90, 230, 255), 2)
                x0 = (gi % 2) * CELL_W
                y0 = (gi // 2) * CELL_H
                canvas[y0:y0 + CELL_H, x0:x0 + CELL_W] = cell
                cv2.putText(canvas, ang, (x0 + 10, y0 + 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 205), 2)
            yb = block(canvas, 46, "PIPELINE (ours)", pred_rows, (60, 180, 230))
            block(canvas, max(yb + 40, H // 2 + 20), "GROUND TRUTH", gt_rows,
                  (200, 200, 210))
            cv2.putText(canvas, f"play {ci+1}/{len(sel)}  t={int(t)//60:02d}:{int(t)%60:02d}",
                        (GRID_W + 18, H - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (150, 150, 155), 1)
            vw.write(canvas)
        for cap in caps.values():
            if cap is not None:
                cap.release()
        print(f"play {ci+1}/{len(sel)} rendered", flush=True)
    vw.release()
    outp = REPO / a.out
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", tmp,
                    "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p",
                    "-y", str(outp)], check=True)
    Path(tmp).unlink()
    print(f"-> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
