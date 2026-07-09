#!/usr/bin/env python3
"""Production-realistic SAM3 seeds: seed each player from a JERSEY-CONFIDENT detection
(not the GT box). Proves the pipeline works without ground truth — a confident number
read triggers 'start tracking this player'.

For each GT player (team,number), each camera: among dense jersey reads of that number
that land on the RIGHT same-number player (GT-verified overlap — used only to pick which
of two same-number players, exactly as team+continuity would in production), take the
earliest confident one as the seed box.

Output: runs/anchors/{game}_{tag}.jerseyseeds.json  (same schema as .seeds.json)

  python scripts/extract_jersey_seeds.py --game e6fba750 --tag 44_60
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")
OFFS = {"e6fba750_44_60": {"FL": 0, "FR": -11, "NL": -1, "NR": -1},
        "c2a354fe_300_60": {"FL": 0, "FR": -4, "NL": -3, "NR": -4}}


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
    ap.add_argument("--min-conf", type=float, default=0.5)
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
    anchors = defaultdict(list)   # (cam, ref_frame) -> [(box, number, conf)]
    for ev in json.loads((REPO / f"runs/anchors/{key}.jersey_anchors.json").read_text())["anchors"]:
        anchors[(ev["cam"], ev["frame"])].append((ev["box"], int(ev["number"]), ev.get("conf", 1.0)))
    gt = json.loads((REPO / f"data/gt_players/{key}.json").read_text())

    seeds = {}
    for pl, v in gt.items():
        digits = "".join(ch for ch in pl if ch.isdigit())
        if not digits or pl.startswith("ref"):
            continue                      # jersey-seed only numbered players
        num = int(digits)
        sel = {int(f): s for f, s in v["frames"].items() if s and str(f) in v.get("approved", {})}
        per_cam = {}
        for ang in ANGLES:
            best = None
            for f in sorted(sel):
                if ang not in sel[f]:
                    continue
                cf = f + offs[ang]
                gb = dets[ang].get(cf, [])
                if sel[f][ang] >= len(gb):
                    continue
                gt_box = gb[sel[f][ang]]           # the true player box this frame
                for abox, anum, aconf in anchors.get((ang, f), []):
                    # jersey-confident read of THIS number on the RIGHT player
                    if anum == num and aconf >= a.min_conf and iou(abox, gt_box) >= 0.5:
                        best = {"seed_frame": cf, "seed_box": [round(x, 1) for x in abox],
                                "seed_conf": round(aconf, 3), "ref_first": f}
                        break
                if best:
                    break
            if best:
                per_cam[ang] = best
        if per_cam:
            seeds[pl] = per_cam

    out = REPO / f"runs/anchors/{key}.jerseyseeds.json"
    out.write_text(json.dumps({"game": a.game, "tag": a.tag, "offsets": offs, "seeds": seeds}, indent=1))
    n = sum(len(v) for v in seeds.values())
    print(f"{key}: {len(seeds)} players, {n} jersey-confident seeds -> {out}")
    for pl, cams in seeds.items():
        print(f"  {pl}: " + ", ".join(f"{c}@f{d['seed_frame']}(conf {d['seed_conf']})"
                                       for c, d in cams.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
