#!/usr/bin/env python3
"""Render ONE player's SAM3 single-object track: box highlighted in each camera that
holds him + fused court dot + trail. Shows the mechanism — a camera drops him, others
carry, fusion makes one court track.

  python scripts/render_sam3_player.py --game e6fba750 --tag 44_60 --player "#11"
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
from uball_cc.fusion.court import draw_court  # noqa: E402
from uball_cc.fusion.homography import load_calib, project_pixels  # noqa: E402

ANGLES = ("FL", "FR", "NL", "NR")
W, H, FPS = 1920, 1080, 30
HL = (0, 240, 255)
OFFS = {"e6fba750_44_60": {"FL": 0, "FR": -11, "NL": -1, "NR": -1},
        "e6fba750_44_180": {"FL": 0, "FR": -11, "NL": -1, "NR": -1}}
ZONE = {"FL": 0.6, "FR": 0.6, "NL": 1.0, "NR": 1.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--player", required=True)
    ap.add_argument("--sam3-dir", default="runs/sam3_players")
    ap.add_argument("--suffix", default="", help="output filename suffix (avoid overwrite)")
    a = ap.parse_args()
    key = f"{a.game}_{a.tag}"
    offs = OFFS[key]
    safe = a.player.replace("#", "n").replace(" ", "")
    calib = {ang: load_calib(str(REPO / f"configs/calib/{ang}.json")) for ang in ANGLES}

    sam = {}
    for ang in ANGLES:
        p = REPO / a.sam3_dir / f"{key}__{safe}__{ang}.json"
        if p.exists():
            sam[ang] = {int(f): r for f, r in json.loads(p.read_text())["frames"].items()}
    caps = {ang: cv2.VideoCapture(str(REPO / f"data/clips/{key.rsplit('_',2)[0]}_{ang}_{a.tag}.mp4"))
            for ang in ANGLES}
    caps = {ang: cv2.VideoCapture(str(REPO / f"data/clips/{a.game}_{ang}_{a.tag}.mp4"))
            for ang in ANGLES}
    n = int(min(c.get(cv2.CAP_PROP_FRAME_COUNT) for c in caps.values()))

    base, to_px = draw_court(scale=0.40, margin=26)
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
    out = REPO / f"runs/tracking/sam3player_{safe}{a.suffix}_{a.game}_raw.mp4"
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    trail = []
    for f in range(n):
        canvas = np.zeros((H, W, 3), np.uint8)
        held = []
        cpos, cw = [], []
        for k, ang in enumerate(ANGLES):
            ok, img = caps[ang].read()
            if not ok:
                img = np.zeros((1080, 1920, 3), np.uint8)
            cf = f  # camera-native frame index
            r = sam.get(ang, {}).get(cf)
            seen = r and r.get("present") and r.get("box")
            if seen:
                x1, y1, x2, y2 = (int(v) for v in r["box"])
                cv2.rectangle(img, (x1, y1), (x2, y2), HL, 8)
                cv2.putText(img, a.player, (x1, max(40, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX,
                            2.2, HL, 5, cv2.LINE_AA)
                held.append(ang)
                cpos.append(court_of(ang, r["box"])); cw.append(ZONE[ang])
            tile = cv2.resize(img, (640, 360))
            rr, cc = divmod(k, 2)
            y0 = gy0 + rr * 360
            canvas[y0:y0 + 360, cc * 640:cc * 640 + 640] = tile
            col = HL if seen else (110, 110, 110)
            cv2.putText(canvas, f"{ang}{' *' if seen else ''}", (cc * 640 + 10, y0 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2, cv2.LINE_AA)
        court = vc.copy()
        if cpos:
            fused = np.average(cpos, axis=0, weights=cw)
            trail.append(vpx(fused))
        for i in range(1, len(trail)):
            cv2.line(court, trail[i - 1], trail[i], (90, 200, 90), 2)
        if cpos:
            cv2.circle(court, trail[-1], 15, HL, -1)
        canvas[cy0:cy0 + vh, cx0:cx0 + vw_] = court
        cv2.rectangle(canvas, (0, 0), (W, 74), (12, 12, 12), -1)
        cv2.putText(canvas, f"SAM3 single-object + 4-camera fusion  —  {a.player}",
                    (28, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, HL, 2, cv2.LINE_AA)
        state = f"held by {', '.join(held)}" if held else "no camera — coasting"
        cv2.putText(canvas, state, (28, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(canvas)
    vw.release()
    for c in caps.values():
        c.release()
    final = REPO / f"runs/tracking/sam3player_{safe}{a.suffix}_{a.game}.mp4"
    subprocess.run(["ffmpeg", "-y", "-i", str(out), "-c:v", "libx264", "-preset", "fast",
                    "-crf", "23", "-pix_fmt", "yuv420p", str(final)], check=True, capture_output=True)
    out.unlink()
    print(f"-> {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
