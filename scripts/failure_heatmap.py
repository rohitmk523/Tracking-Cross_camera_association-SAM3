#!/usr/bin/env python3
"""Failure heatmap for the CURRENT pipeline: where on the court do we lose players?

For every GT (player, frame): position = fused GT court point; score = fraction of
GT-visible cameras where the pipeline's corrected box matches GT (IoU>=0.3) — the
same strict all-angles standard. Cells colored green (held) -> red (lost), with
sample counts. Both games combined (e6 tuned + c2a blind).

  python scripts/failure_heatmap.py --out runs/tracking/pipeline_failure_heatmap.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
ANGLES = ("FL", "FR", "NL", "NR")
GAMES = [
    ("e6fba750", "44_60", {"FL": 0, "FR": -11, "NL": -1, "NR": -1},
     "runs/demo_all", ["#11", "#22", "#43", "#6"]),
    ("c2a354fe", "300_60", {"FL": 0, "FR": 1, "NL": 2, "NR": -1},
     "runs/demo_c2a", ["#3B", "#3W", "#5B"]),
]
CELL = 150.0                                     # 1.5m cells


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/tracking/pipeline_failure_heatmap.jpg")
    a = ap.parse_args()
    from uball_cc.fusion.homography import load_calib, project_pixels
    from uball_cc.fusion.court import draw_court

    calib = {ang: load_calib(str(REPO / f"configs/calib/{ang}.json")) for ang in ANGLES}

    def court(ang, box):
        (x, y), = project_pixels([((box[0] + box[2]) / 2, box[3])], calib[ang])
        return np.array([x, y])

    hits = defaultdict(lambda: [0.0, 0])          # cell -> [sum score, n]
    for game, tag, offs, tracks_dir, players in GAMES:
        key = f"{game}_{tag}"
        gt = json.loads((REPO / f"data/gt_players/{key}.json").read_text())
        dets = {}
        for ang in ANGLES:
            z = np.load(REPO / f"runs/dets_cache/{game}_{ang}_{tag}_small_1280_t0.25.dets.npz")
            m = defaultdict(list)
            for b, c, f in zip(z["boxes"], z["classes"], z["frame_idx"]):
                m[int(f)].append([float(v) for v in b])
            dets[ang] = m
        for pl in players:
            safe = pl.replace("#", "n")
            track = {}
            for ang in ANGLES:
                p = REPO / tracks_dir / f"{key}__{safe}__{ang}.json"
                if p.exists():
                    track[ang] = {int(f): r["box"] for f, r in json.loads(p.read_text())["frames"].items()
                                  if r.get("present") and r.get("box")}
            sel = {int(f): s for f, s in gt[pl]["frames"].items()
                   if s and str(f) in gt[pl].get("approved", {})}
            for f, s in sel.items():
                gpos, ok, vis = [], 0, 0
                for ang in s:
                    cf = f + offs[ang]
                    gb = dets[ang].get(cf, [])
                    if s[ang] >= len(gb):
                        continue
                    vis += 1
                    gpos.append(court(ang, gb[s[ang]]))
                    cb = track.get(ang, {}).get(cf)
                    if cb and iou(gb[s[ang]], cb) >= 0.3:
                        ok += 1
                if not gpos or vis == 0:
                    continue
                pos = np.mean(gpos, axis=0)
                cell = (int(pos[0] // CELL), int(pos[1] // CELL))
                hits[cell][0] += ok / vis
                hits[cell][1] += 1

    base, to_px = draw_court(scale=0.55, margin=40)
    img = base.copy()
    for (cx, cy), (ssum, n) in hits.items():
        if n < 15:
            continue
        rate = ssum / n
        x0, y0 = to_px((cx * CELL, cy * CELL))
        x1, y1 = to_px(((cx + 1) * CELL, (cy + 1) * CELL))
        xa, xb = sorted((x0, x1)); ya, yb = sorted((y0, y1))
        col = (60, int(60 + rate * 170), int(60 + (1 - rate) * 160))   # BGR: red=lost, green=held
        overlay = img.copy()
        cv2.rectangle(overlay, (xa, ya), (xb, yb), col, -1)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        cv2.putText(img, f"{rate:.0%}", (xa + 4, (ya + yb) // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (250, 250, 250), 1, cv2.LINE_AA)
    cv2.putText(img, "CURRENT pipeline: strict hold-rate by court position (both games, 7 GT players)",
                (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    outp = REPO / a.out
    cv2.imwrite(str(outp), img)
    rates = sorted(((s / n, c, n) for c, (s, n) in hits.items() if n >= 15))
    print("weakest cells (rate, cell, n):")
    for r, c, n in rates[:6]:
        print(f"  {r:.0%}  cell {c}  n={n}")
    print(f"-> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
