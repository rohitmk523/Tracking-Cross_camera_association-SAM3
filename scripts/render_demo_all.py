#!/usr/bin/env python3
"""WHOLE-PIPELINE demo: every identified player, all four cameras, one output.

Players are picked up the moment they first appear in any camera, boxed with a
per-player colour + number in every view that holds them, and placed on the
top-down court map via the 2D->3D projection. This is the production behaviour in
one video — not per-player runs stitched together.

  python scripts/render_demo_all.py --game e6fba750 --tag 44_60 \
      --tracks-dir runs/demo_all --out demo_allplayers
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
ANGLES = ("FL", "FR", "NL", "NR")
FPS = 29.97
W, H = 1920, 810

COLORS = [(80, 200, 255), (90, 255, 120), (255, 150, 80), (255, 90, 220),
          (120, 120, 255), (60, 230, 230), (200, 255, 100), (255, 200, 60),
          (180, 130, 255), (100, 255, 190)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tracks-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from uball_cc.fusion.homography import load_calib, project_pixels
    from uball_cc.fusion.court import draw_court

    key = f"{a.game}_{a.tag}"
    calib = {ang: load_calib(str(REPO / f"configs/calib/{ang}.json")) for ang in ANGLES}

    tracks = {}                                   # pl -> ang -> {cf: box}
    for p in sorted((REPO / a.tracks_dir).glob(f"{key}__n*__*.json")):
        parts = p.stem.split("__")
        pl, ang = "#" + parts[1][1:], parts[2]
        d = json.loads(p.read_text())
        tracks.setdefault(pl, {})[ang] = {int(f): r["box"] for f, r in d["frames"].items()
                                          if r.get("present") and r.get("box")}
    players = sorted(tracks, key=lambda s: (int(''.join(c for c in s if c.isdigit())), s))
    color = {pl: COLORS[i % len(COLORS)] for i, pl in enumerate(players)}
    print(f"rendering {len(players)} players: {players}")

    caps = {ang: cv2.VideoCapture(str(REPO / f"data/clips/{a.game}_{ang}_{a.tag}.mp4"))
            for ang in ANGLES}
    n = int(min(c.get(cv2.CAP_PROP_FRAME_COUNT) for c in caps.values()))

    base, to_px = draw_court(scale=0.30, margin=20)
    hb, wb = base.shape[:2]
    vc = np.rot90(base).copy()
    vh, vw_ = vc.shape[:2]

    def vpx(xy):
        x, y = to_px(xy)
        return (y, wb - 1 - x)

    def court_of(ang, box):
        (cx, cy), = project_pixels([((box[0] + box[2]) / 2, box[3])], calib[ang])
        return np.array([cx, cy])

    gy0 = 74 + (H - 74 - 720) // 2
    cx0 = 1280 + (640 - vw_) // 2
    cy0 = 74 + (H - 74 - vh) // 2
    raw = REPO / f"runs/tracking/{a.out}_{a.game}_raw.mp4"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    ZONE = {"FL": 0.6, "FR": 0.6, "NL": 1.0, "NR": 1.0}

    for f in range(n):
        canvas = np.zeros((H, W, 3), np.uint8)
        court_pts = {}                            # pl -> fused court pos this frame
        imgs = {}
        for ang in ANGLES:
            ok, img = caps[ang].read()
            imgs[ang] = img if ok else np.zeros((1080, 1920, 3), np.uint8)
        n_on = 0
        for pl in players:
            pts, ws = [], []
            for ang in ANGLES:
                box = tracks[pl].get(ang, {}).get(f)
                if not box:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(imgs[ang], (x1, y1), (x2, y2), color[pl], 6)
                cv2.putText(imgs[ang], pl, (x1, max(36, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, color[pl], 4, cv2.LINE_AA)
                pts.append(court_of(ang, box)); ws.append(ZONE[ang])
            if pts:
                court_pts[pl] = np.average(pts, axis=0, weights=ws)
                n_on += 1
        for k, ang in enumerate(ANGLES):
            tile = cv2.resize(imgs[ang], (640, 360))
            rr, cc = divmod(k, 2)
            y0 = gy0 + rr * 360
            canvas[y0:y0 + 360, cc * 640:cc * 640 + 640] = tile
            cv2.putText(canvas, ang, (cc * 640 + 10, y0 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (230, 230, 230), 2, cv2.LINE_AA)
        court = vc.copy()
        for pl, xy in court_pts.items():
            p = vpx(xy)
            cv2.circle(court, p, 13, color[pl], -1)
            cv2.putText(court, pl[1:], (p[0] - 12, p[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 15, 15), 2, cv2.LINE_AA)
        canvas[cy0:cy0 + vh, cx0:cx0 + vw_] = court
        cv2.rectangle(canvas, (0, 0), (W, 74), (12, 12, 12), -1)
        cv2.putText(canvas, "Full pipeline - every player, four cameras, one court map (no SAM3)",
                    (28, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"players on court map: {n_on}/{len(players)}",
                    (28, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(canvas)
    vw.release()
    for c in caps.values():
        c.release()
    final = REPO / f"runs/tracking/{a.out}_{a.game}.mp4"
    subprocess.run(["ffmpeg", "-y", "-i", str(raw), "-c:v", "libx264", "-preset", "fast",
                    "-crf", "23", "-pix_fmt", "yuv420p", str(final)], check=True, capture_output=True)
    raw.unlink()
    print(f"-> {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
