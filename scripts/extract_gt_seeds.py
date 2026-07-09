#!/usr/bin/env python3
"""Seed boxes for SAM3 single-object tracking (the Roboflow recipe).

For each ground-truth player and camera, find the FIRST frame the operator marked them
visible and emit that camera-pixel box as the SAM3 seed. SAM3 is then prompted with
this one box and propagates a mask for THAT player across the clip — single-object
tracking, not open-vocabulary detection of everyone.

Output: runs/anchors/{game}_{tag}.seeds.json
  { player: { cam: {seed_frame, seed_box, n_visible} } }

  python scripts/extract_gt_seeds.py --game e6fba750 --tag 44_60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")
OFFS = {"e6fba750_44_60": {"FL": 0, "FR": -11, "NL": -1, "NR": -1},
        "c2a354fe_300_60": {"FL": 0, "FR": -4, "NL": -3, "NR": -4}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    key = f"{a.game}_{a.tag}"
    offs = OFFS[key]

    dets = {}
    for ang in ANGLES:
        z = np.load(REPO / f"runs/dets_cache/{a.game}_{ang}_{a.tag}_small_1280_t0.25.dets.npz")
        m = {}
        for b, s, c, f in zip(z["boxes"], z["scores"], z["classes"], z["frame_idx"]):
            if int(c) in (0, 1):
                m.setdefault(int(f), []).append([float(v) for v in b])
        dets[ang] = m

    gt = json.loads((REPO / f"data/gt_players/{key}.json").read_text())
    seeds = {}
    for pl, v in gt.items():
        if len(v.get("approved", {})) < 30:
            continue
        sel = {int(f): s for f, s in v["frames"].items() if s and str(f) in v.get("approved", {})}
        per_cam = {}
        for ang in ANGLES:
            fs = sorted(f for f, s in sel.items() if ang in s)
            if not fs:
                continue
            f0 = fs[0]
            bi = sel[f0][ang]
            clip_f = f0 + offs[ang]                       # camera's own timeline
            boxes = dets[ang].get(clip_f, [])
            if bi >= len(boxes):
                continue
            per_cam[ang] = {"seed_frame": clip_f, "seed_box": [round(x, 1) for x in boxes[bi]],
                            "n_visible": len(fs), "ref_first": f0}
        if per_cam:
            seeds[pl] = per_cam
    out = REPO / f"runs/anchors/{key}.seeds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"game": a.game, "tag": a.tag, "offsets": offs, "seeds": seeds}, indent=1))
    n = sum(len(v) for v in seeds.values())
    print(f"{key}: {len(seeds)} players, {n} (player,camera) seeds -> {out}")
    for pl, cams in seeds.items():
        print(f"  {pl}: " + ", ".join(f"{c}@f{d['seed_frame']}" for c, d in cams.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
