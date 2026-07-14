#!/usr/bin/env python3
"""Camera sync via ANCHOR-GEOMETRY sweep — the arbiter.

For each camera vs FL, sweep a residual frame offset and score cross-camera
agreement of jersey anchors: same number read in both cameras at (adjusted)
matching ref frames should project to the SAME court spot. The offset that
minimizes median court distance (and maximizes close pairs) is the truth the
PIPELINE cares about — no audio, no identity solving, just homography.

Anchors' "frame" fields are ref-baked with whatever offsets the prep used,
so the sweep returns the RESIDUAL vs that bake:
  true_offset = baked_offset + residual.
Validation: e6 (baked with pipeline-proven offsets) must sweep to ~0.

  .venv/bin/python scripts/sync_anchor_sweep.py --game e6fba750
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from game_meta import GAME_CHUNKS

ANGLES = ("FL", "FR", "NL", "NR")
SWEEP = range(-60, 61)
PAIR_TOL_F = 1          # ref-frame slop when pairing reads
CLOSE_CM = 120.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--min-conf", type=float, default=0.8)
    a = ap.parse_args()
    from uball_cc.fusion.homography import load_calib, project_pixels

    calib = {ang: load_calib(str(REPO / f"configs/calib/{ang}.json"))
             for ang in ANGLES}

    # (ang, num+kit) -> {ref_frame: court_xy} using highest-conf read per frame
    reads: dict[str, dict[tuple, dict[int, np.ndarray]]] = {
        ang: defaultdict(dict) for ang in ANGLES}
    conf_at: dict[str, dict[tuple, dict[int, float]]] = {
        ang: defaultdict(dict) for ang in ANGLES}
    n_files = 0
    for tag in GAME_CHUNKS[a.game]:
        p = REPO / f"runs/anchors/{a.game}_{tag}.jersey_anchors.json"
        if not p.exists():
            continue
        n_files += 1
        base = round(float(tag.split("_")[0]) * 29.97)
        doc = json.loads(p.read_text())
        for ev in doc["anchors"]:
            if ev.get("conf", 0) < a.min_conf:
                continue
            ang = ev["cam"]
            key = (int(ev["number"]), ev.get("kit"))
            f = base + int(ev["frame"])
            b = ev["box"]
            (x, y), = project_pixels([((b[0] + b[2]) / 2, b[3])], calib[ang])
            old = conf_at[ang][key].get(f, 0.0)
            if ev["conf"] > old:
                conf_at[ang][key][f] = ev["conf"]
                reads[ang][key][f] = np.array([x, y])
    if not n_files:
        raise SystemExit(f"no anchors for {a.game}")

    print(f"{a.game}: anchor-geometry sweep vs FL "
          f"(residual frames; true = baked + residual)")
    for ang in ("FR", "NL", "NR"):
        best = None
        curve = {}
        for d in SWEEP:
            dists = []
            for key, fl_reads in reads["FL"].items():
                oth = reads[ang].get(key)
                if not oth:
                    continue
                for f, pfl in fl_reads.items():
                    for df in range(-PAIR_TOL_F, PAIR_TOL_F + 1):
                        po = oth.get(f + d + df)
                        if po is not None:
                            dists.append(float(np.hypot(*(pfl - po))))
                            break
            if len(dists) >= 200:
                med = float(np.median(dists))
                close = float(np.mean(np.array(dists) <= CLOSE_CM))
                curve[d] = (med, close, len(dists))
        if not curve:
            print(f"  {ang}: UNRESOLVED (too few cross-cam pairs)")
            continue
        d_star = min(curve, key=lambda d: curve[d][0])
        med, close, n = curve[d_star]
        neigh = {d: curve[d][0] for d in (d_star - 1, d_star + 1) if d in curve}
        rivals = sorted((m, d) for d, (m, _, _) in curve.items()
                        if abs(d - d_star) > 3)
        margin = rivals[0][0] / med if rivals and med > 0 else float("inf")
        print(f"  {ang}: residual {d_star:+d}f  median {med:.0f}cm  "
              f"close@{CLOSE_CM:.0f}cm {close:.0%}  pairs {n}  "
              f"rival-margin {margin:.2f}x  neigh {neigh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
