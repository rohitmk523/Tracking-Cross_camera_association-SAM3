#!/usr/bin/env python3
"""Score SAM3's RAW per-camera video-tracker identities against operator player GT.

For each annotated player and camera: match the operator's clicked box (cached-
detection coords) to SAM3's box on the same clip frame by IoU; measure how many
distinct SAM3 ids cover that one physical player (purity/switches). This is the
'is SAM3 alone a clean tracker on OUR footage' number.

  python scripts/score_sam3_raw_gt.py --game e6fba750 --tag 44_60 \
      --sam3-dir runs/sam3_demo60
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")
OFFS = {"e6fba750_44_60": {"FL": 0, "FR": -11, "NL": -1, "NR": -1}}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sam3-dir", default="runs/sam3_demo60")
    ap.add_argument("--min-iou", type=float, default=0.4)
    a = ap.parse_args()

    key = f"{a.game}_{a.tag}"
    offs = OFFS[key]
    dets = {}
    for ang in ANGLES:
        z = np.load(REPO / f"runs/dets_cache/{a.game}_{ang}_{a.tag}_small_1280_t0.25.dets.npz")
        m = {}
        for b, s, c, f in zip(z["boxes"], z["scores"], z["classes"], z["frame_idx"]):
            if int(c) in (0, 1) and float(s) >= 0.25:
                m.setdefault(int(f), []).append([float(v) for v in b])
        dets[ang] = m
    sam3 = {}
    for ang in ANGLES:
        p = REPO / a.sam3_dir / f"{a.game}_{ang}_{a.tag}.sam3.json"
        if p.exists():
            d = json.loads(p.read_text())
            sam3[ang] = {int(f): [(r.get("id"), r["box"]) for r in rows if r["score"] >= 0.3]
                         for f, rows in d["frames"].items()}

    gt = json.loads((REPO / f"data/gt_players/{key}.json").read_text())
    report = {}
    for player, rec in gt.items():
        per_cam = {}
        for ang in ANGLES:
            if ang not in sam3:
                continue
            seq = []
            n_gt = n_cov = 0
            for f_str, sels in rec.get("frames", {}).items():
                if str(f_str) not in rec.get("approved", {}) or ang not in sels:
                    continue
                f = int(f_str)
                clip_f = f + offs[ang]
                boxes = dets[ang].get(clip_f, [])
                bi = sels[ang]
                if bi >= len(boxes):
                    continue
                gt_box = boxes[bi]
                n_gt += 1
                cands = sam3[ang].get(clip_f, [])
                if not cands:
                    continue
                best = max(cands, key=lambda c: iou(gt_box, c[1]))
                if iou(gt_box, best[1]) >= a.min_iou:
                    n_cov += 1
                    seq.append((f, best[0]))
            if n_gt < 30:
                continue
            ids = Counter(i for _, i in seq)
            dom = ids.most_common(1)[0] if ids else (None, 0)
            ordered = [i for _, i in sorted(seq)]
            switches = sum(1 for x, y in zip(ordered, ordered[1:]) if x != y)
            per_cam[ang] = {"gt_frames": n_gt,
                            "sam3_coverage": round(n_cov / n_gt, 3),
                            "id_purity": round(dom[1] / max(1, len(seq)), 3),
                            "n_ids": len(ids), "switches": switches}
        if per_cam:
            report[player] = per_cam

    out = {"window": key, "sam3_raw": report}
    print(json.dumps(out, indent=1))
    led = REPO / f"runs/tracking/ledger/playerGT_sam3raw_{key}.json"
    led.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
