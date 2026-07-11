#!/usr/bin/env python3
"""Extract the scoring-zone lines from court frames: WHITE = 3PT, RED = 4PT (venue
court-a). Color-masked floor pixels (player boxes excluded) are projected through
each camera's homography onto the court plane; the white cloud should land on the
modeled 3PT arc (radius 670.6 cm from each basket) — a free calibration check —
and the red cloud gives the 4PT line geometry (fit as an arc radius per side).

Output: runs/court_zones/{game}_zone_points.npz + verification image
        runs/court_zones/{game}_zones_verify.jpg (for user visual confirmation).

  python scripts/extract_court_zones.py --game e6fba750 --tag 44_60
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO / "src"))

ANGLES = ("FL", "FR", "NL", "NR")
FRAMES = (0, 400, 800, 1200, 1600)
PX_STRIDE = 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="e6fba750")
    ap.add_argument("--tag", default="44_60")
    a = ap.parse_args()
    from uball_cc.fusion.homography import load_calib, project_pixels
    from uball_cc.fusion.court import (draw_court, LEFT_BASKET, RIGHT_BASKET,
                                       THREE_PT_R, LENGTH, WIDTH)

    outd = REPO / "runs/court_zones"
    outd.mkdir(parents=True, exist_ok=True)
    pts = {"white": [], "red": []}
    for ang in ANGLES:
        calib = load_calib(str(REPO / f"configs/calib/{ang}.json"))
        z = np.load(REPO / f"runs/dets_cache/{a.game}_{ang}_{a.tag}_small_1280_t0.25.dets.npz")
        by_f = {}
        for b, f in zip(z["boxes"], z["frame_idx"]):
            by_f.setdefault(int(f), []).append(b)
        cap = cv2.VideoCapture(str(REPO / f"data/clips/{a.game}_{ang}_{a.tag}.mp4"))
        for fr in FRAMES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, img = cap.read()
            if not ok:
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
            white = (S < 45) & (V > 175)
            red = (((H <= 10) | (H >= 170)) & (S > 90) & (V > 60))
            # exclude player/ref boxes (dilated) — lines are on the FLOOR
            mask_ex = np.zeros(img.shape[:2], bool)
            for b in by_f.get(fr, []):
                x1, y1 = max(0, int(b[0]) - 12), max(0, int(b[1]) - 12)
                x2, y2 = min(img.shape[1], int(b[2]) + 12), min(img.shape[0], int(b[3]) + 12)
                mask_ex[y1:y2, x1:x2] = True
            for name, m in (("white", white), ("red", red)):
                mm = m & ~mask_ex
                ys, xs = np.nonzero(mm)
                if not len(ys):
                    continue
                sel = slice(None, None, PX_STRIDE)
                proj = project_pixels(list(zip(xs[sel].tolist(), ys[sel].tolist())), calib)
                for x, y in proj:
                    if -120 <= x <= LENGTH + 120 and -120 <= y <= WIDTH + 120:
                        pts[name].append((x, y))
        cap.release()
        print(f"{ang}: white {len(pts['white'])}, red {len(pts['red'])} (cumulative)")

    w = np.array(pts["white"], np.float32) if pts["white"] else np.zeros((0, 2))
    r = np.array(pts["red"], np.float32) if pts["red"] else np.zeros((0, 2))
    np.savez_compressed(outd / f"{a.game}_zone_points.npz", white=w, red=r)

    # fit 4PT radius per half: median distance of red points to that half's basket
    fits = {}
    for side, bk in (("L", LEFT_BASKET), ("R", RIGHT_BASKET)):
        half = r[(r[:, 0] < LENGTH / 2)] if side == "L" else r[(r[:, 0] >= LENGTH / 2)]
        if len(half) > 200:
            d = np.linalg.norm(half - np.array(bk), axis=1)
            d = d[(d > 400) & (d < 1200)]              # sane arc band only
            fits[side] = float(np.median(d))
    print("4PT radius fit (cm):", fits)
    # white sanity: distance of white points near the modeled arc
    wd = []
    for side, bk in (("L", LEFT_BASKET), ("R", RIGHT_BASKET)):
        half = w[(w[:, 0] < LENGTH / 2)] if side == "L" else w[(w[:, 0] >= LENGTH / 2)]
        if len(half):
            d = np.linalg.norm(half - np.array(bk), axis=1)
            near = d[np.abs(d - THREE_PT_R) < 80]
            if len(near):
                wd.append(float(np.median(near) - THREE_PT_R))
    print(f"white-vs-modeled-3PT arc offset (cm): {wd}")

    base, to_px = draw_court(scale=0.30, margin=20)
    canvas = base.copy()
    for x, y in w[::6]:
        cv2.circle(canvas, to_px((x, y)), 1, (230, 230, 230), -1)
    for x, y in r[::6]:
        cv2.circle(canvas, to_px((x, y)), 1, (0, 0, 255), -1)
    for side, bk in (("L", LEFT_BASKET), ("R", RIGHT_BASKET)):
        cv2.circle(canvas, to_px(bk), int(THREE_PT_R * 0.30), (0, 200, 0), 1)
        if side in fits:
            cv2.circle(canvas, to_px(bk), int(fits[side] * 0.30), (255, 0, 255), 1)
    cv2.putText(canvas, "white dots=3PT paint  red dots=4PT paint  green=model 3PT  magenta=fit 4PT",
                (10, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1)
    out = outd / f"{a.game}_zones_verify.jpg"
    cv2.imwrite(str(out), canvas)
    print(f"verify image -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
