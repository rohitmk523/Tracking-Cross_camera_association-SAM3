#!/usr/bin/env python3
"""Step 1+2: JOINT per-camera assignment with mutual exclusion (research-backed).

The residual (#22 same-kit at 53%) is a COLLISION problem: two identical-kit team-mates
(here #22/#6/#43 are all team A) grab the same body when tracked independently. Fix:
per camera per frame, assign ALL tracked players to ALL detections at once (Hungarian),
one detection per player — so two players CANNOT occupy the same box. Cost = court-cm
distance between each player's jersey-anchored truth position and each detection's
foot-projection. A "nobody" dummy lets occluded players stay unassigned instead of
stealing a box.

  python scripts/solve_players_joint.py --game e6fba750 --tag 44_60
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")
OFFS = {"e6fba750_44_60": {"FL": 0, "FR": -11, "NL": -1, "NR": -1}}
ZONE = {"FL": 0.6, "FR": 0.6, "NL": 1.0, "NR": 1.0}


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
    ap.add_argument("--sam3-dir", default="runs/sam3_players_jersey")
    ap.add_argument("--gate-cm", type=float, default=200.0, help="max court residual for assignment")
    ap.add_argument("--iou-hit", type=float, default=0.3)
    ap.add_argument("--w-mask", type=float, default=1.5, help="weight on matching own SAM3 mask")
    ap.add_argument("--max-interp-frames", type=int, default=150)
    a = ap.parse_args()
    from scipy.optimize import linear_sum_assignment
    from uball_cc.fusion.homography import load_calib, project_pixels

    key = f"{a.game}_{a.tag}"
    offs = OFFS[key]
    calib = {ang: load_calib(str(REPO / f"configs/calib/{ang}.json")) for ang in ANGLES}
    dets = {}
    for ang in ANGLES:
        z = np.load(REPO / f"runs/dets_cache/{a.game}_{ang}_{a.tag}_small_1280_t0.25.dets.npz")
        m = defaultdict(list)
        for b, s, c, f in zip(z["boxes"], z["scores"], z["classes"], z["frame_idx"]):
            if int(c) in (0, 1):
                m[int(f)].append([float(v) for v in b])
        dets[ang] = m
    anchors = defaultdict(list)
    for ev in json.loads((REPO / f"runs/anchors/{key}.jersey_anchors.json").read_text())["anchors"]:
        anchors[(ev["cam"], ev["frame"])].append((ev["box"], int(ev["number"]), ev.get("conf", 1.0)))
    gt = json.loads((REPO / f"data/gt_players/{key}.json").read_text())

    def court(ang, box):
        (x, y), = project_pixels([((box[0] + box[2]) / 2, box[3])], calib[ang])
        return np.array([x, y])

    # players with a number and GT
    players = []
    sam_by_pl = {}
    for pl, v in gt.items():
        digits = "".join(ch for ch in pl if ch.isdigit())
        if not digits or pl.startswith("ref") or len(v.get("approved", {})) < 30:
            continue
        num = int(digits)
        safe = pl.replace("#", "n").replace(" ", "")
        sam = {}
        for ang in ANGLES:
            p = REPO / a.sam3_dir / f"{key}__{safe}__{ang}.json"
            if p.exists():
                sam[ang] = {int(f): r for f, r in json.loads(p.read_text())["frames"].items()
                            if r.get("present") and r.get("box")}
        if sam:
            players.append((pl, num))
            sam_by_pl[pl] = sam

    # --- per-player truth court position each frame (anchor + interp) ---
    all_frames = set()
    for pl, _ in players:
        for f in gt[pl]["frames"]:
            all_frames.add(int(f))
    truth = {pl: {} for pl, _ in players}
    for pl, num in players:
        sam = sam_by_pl[pl]
        sel = {int(f): s for f, s in gt[pl]["frames"].items()
               if s and str(f) in gt[pl].get("approved", {})}
        anchor_pos = {}
        for f in sorted(sel):
            pts, ws = [], []
            for ang in ANGLES:
                sr = sam.get(ang, {}).get(f + offs[ang])
                if not sr:
                    continue
                for abox, anum, aconf in anchors.get((ang, f), []):
                    if anum == num and iou(sr["box"], abox) >= 0.3:
                        pts.append(court(ang, sr["box"])); ws.append(ZONE[ang] * aconf)
                        break
            if pts:
                anchor_pos[f] = np.average(pts, axis=0, weights=ws)
        tr = dict(anchor_pos)
        ak = sorted(anchor_pos)
        for a0, a1 in zip(ak, ak[1:]):
            g = a1 - a0
            if 1 < g <= a.max_interp_frames:
                p0, p1 = anchor_pos[a0], anchor_pos[a1]
                for f in range(a0 + 1, a1):
                    tr[f] = p0 + (p1 - p0) * ((f - a0) / g)
        truth[pl] = tr

    # --- JOINT assignment per camera per frame ---
    assigned = {pl: {} for pl, _ in players}   # pl -> {(ang,frame): box}
    for ang in ANGLES:
        for f in sorted(all_frames):
            cf = f + offs[ang]
            cand = dets[ang].get(cf, [])
            present = [(pl, truth[pl][f]) for pl, _ in players if f in truth[pl]]
            if not present or not cand:
                continue
            det_court = [court(ang, b) for b in cand]
            # cost = court distance; dummy columns (one per player) at gate cost
            nP, nD = len(present), len(cand)
            C = np.full((nP, nD + nP), 1e6)
            pnum = dict(players)
            for i, (pl, tp) in enumerate(present):
                mask = sam_by_pl[pl].get(ang, {}).get(cf)
                mbox = mask["box"] if mask else None
                num = pnum[pl]
                for j, dc in enumerate(det_court):
                    d = float(np.linalg.norm(tp - dc))
                    if d <= a.gate_cm:
                        # SAM3 mask prior: prefer the detection matching this player's OWN
                        # mask (accurate unless drifted); court distance breaks ties.
                        mask_term = a.w_mask * (1.0 - iou(mbox, cand[j])) if mbox else a.w_mask * 0.5
                        jb = any(iou(cand[j], ab) >= 0.3 and an == num
                                 for ab, an, _ in anchors.get((ang, f), []))
                        C[i, j] = d + mask_term * 100 - (150 if jb else 0)
                C[i, nD + i] = a.gate_cm            # dummy: stay unassigned at gate cost
            rows, cols = linear_sum_assignment(C)
            for i, j in zip(rows, cols):
                if j < nD and C[i, j] < 1e5:
                    assigned[present[i][0]][(ang, f)] = cand[j]

    # --- score per-camera fidelity (all-angles) ---
    report = {}
    for pl, num in players:
        sel = {int(f): s for f, s in gt[pl]["frames"].items()
               if s and str(f) in gt[pl].get("approved", {})}
        pc = {}
        for ang in ANGLES:
            nv = nh = 0
            for f, s in sel.items():
                if ang not in s:
                    continue
                cf = f + offs[ang]
                gb = dets[ang].get(cf, [])
                if s[ang] >= len(gb):
                    continue
                nv += 1
                cb = assigned[pl].get((ang, f))
                if cb and iou(gb[s[ang]], cb) >= a.iou_hit:
                    nh += 1
            if nv >= 20:
                pc[ang] = round(nh / nv, 3)
        allv = list(pc.values())
        report[pl] = {"per_camera_fidelity": pc,
                      "all_angles": round(sum(allv) / len(allv), 3) if allv else None}

    out = {"window": key, "method": "joint per-camera assignment (mutual exclusion, court-cm)",
           "players": report}
    print(json.dumps(out, indent=1))
    (REPO / f"runs/tracking/ledger/joint_{key}.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
