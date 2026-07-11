#!/usr/bin/env python3
"""Simulate stride-2 DETECTION from a full dets cache (Phase-2 gate experiment).

Keeps even frames' detections as-is; rebuilds odd frames by IoU-greedy matching
of boxes between the surrounding even frames and interpolating linearly. Boxes
with no match across the gap are dropped on the odd frame (a real stride-2 run
would rely on the tracker to bridge those). Lets us GT-gate stride-2 detection
using the existing full cache — no new detection run needed; if the gate holds,
real detection time halves.

  python scripts/make_stride2_dets.py --game c2a354fe --tag 300_60 \
      --src runs/dets_cache --out-dir runs/dets_cache_s2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--src", default="runs/dets_cache")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--match-iou", type=float, default=0.3)
    a = ap.parse_args()
    outd = REPO / a.out_dir
    outd.mkdir(parents=True, exist_ok=True)

    for ang in ANGLES:
        name = f"{a.game}_{ang}_{a.tag}_small_1280_t0.25.dets.npz"
        z = np.load(REPO / a.src / name)
        by_f: dict[int, list] = {}
        for b, s, c, f in zip(z["boxes"], z["scores"], z["classes"], z["frame_idx"]):
            by_f.setdefault(int(f), []).append((b.astype(np.float32), float(s), int(c)))
        max_f = max(by_f) if by_f else -1
        boxes, scores, classes, fidx = [], [], [], []
        n_interp = 0
        for f in range(0, max_f + 1, 2):
            for b, s, c in by_f.get(f, []):
                boxes.append(b); scores.append(s); classes.append(c); fidx.append(f)
            if f + 1 > max_f:
                continue
            nxt = by_f.get(f + 2, [])
            used = set()
            for b, s, c in by_f.get(f, []):
                best, bj = a.match_iou, None
                for j, (b2, s2, c2) in enumerate(nxt):
                    if j in used or c2 != c:
                        continue
                    v = iou(b, b2)
                    if v > best:
                        best, bj = v, j
                if bj is None:
                    continue
                used.add(bj)
                b2, s2, _ = nxt[bj]
                boxes.append(((b + b2) / 2).astype(np.float32))
                scores.append(min(s, s2))
                classes.append(c)
                fidx.append(f + 1)
                n_interp += 1
        np.savez_compressed(outd / name,
                            boxes=np.stack(boxes) if boxes else np.zeros((0, 4), np.float32),
                            scores=np.array(scores), classes=np.array(classes),
                            frame_idx=np.array(fidx))
        print(f"{ang}: {len(z['frame_idx'])} full dets -> {len(boxes)} stride-2 "
              f"({n_interp} interpolated)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
