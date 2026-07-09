#!/usr/bin/env python3
"""Re-seed points for drift-resistant SAM3 tracking.

A single seed drifts onto same-kit team-mates over time. This finds, per player per
camera, a SEQUENCE of jersey-confident detections spaced through the clip — the first
one seeds the track, each later one RE-ANCHORS it to the confirmed player. A drift can
then only survive from one confident number reading to the next.

Re-seed points = confident reads of the player's number that are (a) on the right
same-number player (GT-verified, standing in for team+continuity in production) and
(b) at least --gap frames apart, so we re-anchor periodically and after long gaps.

Output: runs/anchors/{game}_{tag}.reseedpts.json
  { player: { cam: {seed_frame, seed_box, reseeds:[{frame,box}, ...]} } }

  python scripts/extract_reseed_points.py --game e6fba750 --tag 44_60
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
    ap.add_argument("--gap", type=int, default=90, help="min frames between re-seeds (~3s)")
    ap.add_argument("--min-conf", type=float, default=0.6)
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
    anchors = defaultdict(list)
    for ev in json.loads((REPO / f"runs/anchors/{key}.jersey_anchors.json").read_text())["anchors"]:
        anchors[(ev["cam"], ev["frame"])].append((ev["box"], int(ev["number"]), ev.get("conf", 1.0)))
    gt = json.loads((REPO / f"data/gt_players/{key}.json").read_text())

    out = {}
    for pl, v in gt.items():
        digits = "".join(ch for ch in pl if ch.isdigit())
        if not digits or pl.startswith("ref"):
            continue
        num = int(digits)
        sel = {int(f): s for f, s in v["frames"].items() if s and str(f) in v.get("approved", {})}
        per_cam = {}
        for ang in ANGLES:
            pts = []                                   # (clip_frame, box) confident+correct reads
            for f in sorted(sel):
                if ang not in sel[f]:
                    continue
                cf = f + offs[ang]
                gb = dets[ang].get(cf, [])
                if sel[f][ang] >= len(gb):
                    continue
                gt_box = gb[sel[f][ang]]
                for abox, anum, aconf in anchors.get((ang, f), []):
                    if anum == num and aconf >= a.min_conf and iou(abox, gt_box) >= 0.5:
                        pts.append((cf, [round(x, 1) for x in abox]))
                        break
            if not pts:
                continue
            # thin to >= gap apart
            kept = [pts[0]]
            for cf, box in pts[1:]:
                if cf - kept[-1][0] >= a.gap:
                    kept.append((cf, box))
            per_cam[ang] = {"seed_frame": kept[0][0], "seed_box": kept[0][1],
                            "reseeds": [{"frame": cf, "box": box} for cf, box in kept[1:]]}
        if per_cam:
            out[pl] = per_cam

    p = REPO / f"runs/anchors/{key}.reseedpts.json"
    p.write_text(json.dumps({"game": a.game, "tag": a.tag, "offsets": offs, "seeds": out}, indent=1))
    tot = sum(1 + len(d["reseeds"]) for cams in out.values() for d in cams.values())
    print(f"{key}: {len(out)} players, {tot} total anchor points "
          f"({sum(len(c) for c in out.values())} tracks) -> {p}")
    for pl, cams in out.items():
        print(f"  {pl}: " + ", ".join(f"{c}:seed+{len(d['reseeds'])}reseeds" for c, d in cams.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
