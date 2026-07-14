#!/usr/bin/env python3
"""Continuous demo segment, ALL FOUR angles in a 2x2 grid + live event feeds.

Left 2x2: FL/FR/NL/NR with the predicted player boxed in every view at each
CV event (track-driven). Right panel: PIPELINE feed (top) and GROUND TRUTH
feed (bottom), events appearing at their timestamps. Honest by construction.

  python scripts/render_demo_segment_grid.py --game e6fba750 --t0 360 \
      --out runs/event_demo/demo5min_e6fba750.mp4
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
ROWS = 4


def short(cls):
    return cls.replace("FREE_THROW", "FT").replace("_", " ")


def feed(img, y0, y1, title, items, t_now, color):
    cv2.putText(img, title, (GRID_W + 16, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
    cv2.line(img, (GRID_W + 16, y0 + 42), (W - 16, y0 + 42), color, 1)
    y = y0 + 74
    for e in items[-ROWS:][::-1]:
        hot = (t_now - e["t"]) <= 2.5
        col = (90, 230, 255) if hot else (225, 225, 230)
        cv2.putText(img, f"{int(e['t'])//60:02d}:{int(e['t'])%60:02d} {short(e['classification'])[:22]}",
                    (GRID_W + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2 if hot else 1)
        y += 22
        cv2.putText(img, f"  {(e.get('player_a') or '?')[:24]}", (GRID_W + 16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, col if hot else (150, 155, 160), 1)
        y += 28
        if y > y1 - 10:
            break


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--dur", type=float, default=300.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    game = a.game
    OFFS = GAME_OFFS[game]

    plays = json.loads((REPO / f"data/plays/{game}_full.json").read_text())["plays"]
    ev = json.loads((REPO / f"runs/tracking/ledger/events_v2_{game}.json").read_text())["events"]
    pos = REPO / f"runs/tracking/ledger/possession_events_{game}.json"
    if pos.exists():
        ev = ev + json.loads(pos.read_text())["events"]
    ev = [e for e in ev if e.get("tier", "high") == "high"]
    ev.sort(key=lambda e: e["t"])
    gt = [{"t": p["t"], "classification": p["cls"], "player_a": p["a"]} for p in plays]

    roster = json.loads((REPO / f"data/rosters/{game}.json").read_text())
    by_num = defaultdict(list)
    for pr in roster["players"]:
        by_num[pr["num"]].append(pr)

    def stream_of(name):
        recs = [p for p in roster["players"] if p["name"] == (name or "").rstrip("?")]
        if not recs:
            return None
        r = recs[0]
        dual = len(by_num[r["num"]]) > 1
        return f"n{r['num']}" + (("B" if r["team"] == 1 else "W") if dual else "")

    tracks = defaultdict(lambda: defaultdict(dict))
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

    caps = {}
    for ang in ANGLES:
        p = REPO / f"runs/event_demo/_seg4_{game}_{ang}.mp4"
        caps[ang] = cv2.VideoCapture(str(p)) if p.exists() else None

    tmp = str(REPO / a.out) + ".raw.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n_frames = int(a.dur * FPS)
    for fi in range(n_frames):
        t_now = a.t0 + fi / FPS
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:, GRID_W:] = (24, 24, 28)
        active = [e for e in ev if e["t"] - 0.8 <= t_now <= e["t"] + 3.0
                  and e.get("player_a")]
        for gi, ang in enumerate(ANGLES):
            cap = caps[ang]
            if cap is None:
                continue
            ok, frame = cap.read()
            if not ok:
                continue
            fh, fw = frame.shape[:2]
            cell = cv2.resize(frame, (CELL_W, CELL_H))
            gf = round(t_now * FPS) + OFFS[ang]
            for e in active:
                sid = stream_of(e["player_a"])
                box = tracks.get(sid, {}).get(ang, {}).get(gf) if sid else None
                if box is None:
                    continue
                sx, sy = CELL_W / fw, CELL_H / fh
                cv2.rectangle(cell, (int(box[0] * sx), int(box[1] * sy)),
                              (int(box[2] * sx), int(box[3] * sy)), (90, 230, 255), 2)
            x0, y0 = (gi % 2) * CELL_W, (gi // 2) * CELL_H
            canvas[y0:y0 + CELL_H, x0:x0 + CELL_W] = cell
            cv2.putText(canvas, ang, (x0 + 10, y0 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (205, 205, 210), 2)
        cv_feed = [e for e in ev if a.t0 - 1 <= e["t"] <= t_now]
        gt_feed = [e for e in gt if a.t0 - 1 <= e["t"] <= t_now]
        feed(canvas, 8, H // 2 - 8, "PIPELINE (ours)", cv_feed, t_now, (60, 180, 230))
        feed(canvas, H // 2 + 4, H - 40, "GROUND TRUTH", gt_feed, t_now, (200, 200, 210))
        clock = f"{int(t_now)//60:02d}:{int(t_now)%60:02d}"
        cv2.putText(canvas, clock, (GRID_W + 16, H - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (240, 240, 245), 2)
        vw.write(canvas)
        if fi % 3000 == 0:
            print(f"  {fi/FPS:.0f}s", flush=True)
    vw.release()
    for c in caps.values():
        if c is not None:
            c.release()
    outp = REPO / a.out
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", tmp,
                    "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p",
                    "-y", str(outp)], check=True)
    Path(tmp).unlink()
    print(f"-> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
