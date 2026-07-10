#!/usr/bin/env python3
"""Result A: does ankle-midpoint projection beat bbox-bottom for court positions?

Uses GT correspondences as the oracle: the SAME player at the SAME instant seen by
two cameras should project to the SAME court point. Any distance between the two
cameras' projections is measurement error. Compare that cross-camera disagreement
for (a) bbox-bottom foot points (current) vs (b) RTMPose ankle midpoints.

  python scripts/eval_ankle_projection.py --game e6fba750 --tag 44_60
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")
OFFS = {"e6fba750_44_60": {"FL": 0, "FR": -11, "NL": -1, "NR": -1}}
L_ANK, R_ANK = 15, 16


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--kp-conf", type=float, default=0.3)
    a = ap.parse_args()
    from uball_cc.fusion.homography import load_calib, project_pixels

    key = f"{a.game}_{a.tag}"
    offs = OFFS[key]
    calib = {ang: load_calib(str(REPO / f"configs/calib/{ang}.json")) for ang in ANGLES}
    gt = json.loads((REPO / f"data/gt_players/{key}.json").read_text())

    dets, pose = {}, {}
    for ang in ANGLES:
        z = np.load(REPO / f"runs/dets_cache/{a.game}_{ang}_{a.tag}_small_1280_t0.25.dets.npz")
        m = defaultdict(list)
        for di, (b, c, f) in enumerate(zip(z["boxes"], z["classes"], z["frame_idx"])):
            m[int(f)].append((di, [float(v) for v in b]))
        dets[ang] = m
        p = np.load(REPO / f"runs/pose_cache/{a.game}_{ang}_{a.tag}.pose.npz")
        pose[ang] = {(int(f), int(d)): (k, s) for f, d, k, s in
                     zip(p["frame_idx"], p["det_idx"], p["kpts"], p["kscores"])}

    def court(ang, px, py):
        (x, y), = project_pixels([(px, py)], calib[ang])
        return np.array([x, y])

    diffs_box, diffs_ank = [], []
    per_pair = defaultdict(lambda: ([], []))
    for pl, v in gt.items():
        sel = {int(f): s for f, s in v.get("frames", {}).items()
               if s and str(f) in v.get("approved", {})}
        for f, s in sel.items():
            pts_box, pts_ank = {}, {}
            for ang in ANGLES:
                if ang not in s:
                    continue
                cf = f + offs[ang]
                cams = dets[ang].get(cf, [])
                idx = s[ang]
                if idx >= len(cams):
                    continue
                di, box = cams[idx]
                pts_box[ang] = court(ang, (box[0] + box[2]) / 2, box[3])
                kp = pose[ang].get((cf, di))
                if kp is not None:
                    k, ks = kp
                    good = [i for i in (L_ANK, R_ANK) if ks[i] >= a.kp_conf]
                    if good:
                        ax = float(np.mean([k[i][0] for i in good]))
                        ay = float(np.mean([k[i][1] for i in good]))
                        pts_ank[ang] = court(ang, ax, ay)
            for c1, c2 in combinations(sorted(pts_box), 2):
                if c1 in pts_ank and c2 in pts_ank:      # compare on identical support
                    db = float(np.linalg.norm(pts_box[c1] - pts_box[c2]))
                    da = float(np.linalg.norm(pts_ank[c1] - pts_ank[c2]))
                    diffs_box.append(db)
                    diffs_ank.append(da)
                    per_pair[(c1, c2)][0].append(db)
                    per_pair[(c1, c2)][1].append(da)

    b, k = np.array(diffs_box), np.array(diffs_ank)
    print(f"cross-camera court disagreement, same player same instant (n={len(b)} pairs):")
    print(f"  bbox-bottom : median {np.median(b):6.1f}cm   p75 {np.percentile(b,75):6.1f}   p90 {np.percentile(b,90):6.1f}")
    print(f"  ankle-mid   : median {np.median(k):6.1f}cm   p75 {np.percentile(k,75):6.1f}   p90 {np.percentile(k,90):6.1f}")
    print("  per camera pair (median cm, box -> ankle):")
    for (c1, c2), (lb, la) in sorted(per_pair.items()):
        print(f"    {c1}-{c2}: {np.median(lb):6.1f} -> {np.median(la):6.1f}   (n={len(lb)})")
    out = {"n_pairs": len(b),
           "bbox_median_cm": float(np.median(b)), "ankle_median_cm": float(np.median(k)),
           "bbox_p90_cm": float(np.percentile(b, 90)), "ankle_p90_cm": float(np.percentile(k, 90))}
    (REPO / f"runs/tracking/ledger/ankleproj_{key}.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
