#!/usr/bin/env python3
"""Full-roster joint assignment from ANCHOR-DERIVED truths (no extra SAM3 needed).

Two fixes over prior attempts (Fable audit):
  A. SAME-NUMBER SPLIT: a jersey number can exist on BOTH teams (e6 has two #6s,
     #11s, #22s). Each number's anchor reads are split into spatially-coherent
     trajectories (players can't teleport; speed-gated greedy clustering) — each
     cluster is one PHYSICAL player. Prior code mixed both players' reads into one
     corrupted "truth" (why #22 was stuck at 53%).
  B. FULL-ROSTER EXCLUSION from anchors alone: every cluster of every number gets an
     interpolated truth track; the per-camera Hungarian assigns ALL of them to
     detections jointly (one detection -> one player).

Scoring: GT players only, per-camera all-angles fidelity (leakage-free — GT picks
which cluster is "our" player via the same seed read used for tracking, then grades).

  python scripts/solve_joint_anchors.py --game e6fba750 --tag 44_60
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
FPS = 29.97


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
    ap.add_argument("--gate-cm", type=float, default=200.0)
    ap.add_argument("--dummy-cm", type=float, default=160.0)
    ap.add_argument("--iou-hit", type=float, default=0.3)
    ap.add_argument("--split-speed-cms", type=float, default=900.0,
                    help="max plausible speed for same-trajectory reads (cm/s)")
    ap.add_argument("--min-cluster", type=int, default=15, help="min reads to keep a cluster")
    ap.add_argument("--max-interp-s", type=float, default=5.0)
    ap.add_argument("--sam3-dir", default="runs/sam3_players_jersey",
                    help="masklets used as per-camera prior for GT players when present")
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

    def court(ang, box):
        (x, y), = project_pixels([((box[0] + box[2]) / 2, box[3])], calib[ang])
        return np.array([x, y])

    # ---- A. split each number's anchors into spatially-coherent player trajectories ----
    anchors_raw = json.loads((REPO / f"runs/anchors/{key}.jersey_anchors.json").read_text())["anchors"]
    by_num = defaultdict(list)
    for ev in anchors_raw:
        pos = court(ev["cam"], ev["box"])
        by_num[int(ev["number"])].append(
            {"f": ev["frame"], "cam": ev["cam"], "box": ev["box"],
             "pos": pos, "conf": ev.get("conf", 1.0)})
    clusters = {}     # cluster_id -> {"num", "reads":[...]}
    cid = 0
    for num, evs in by_num.items():
        evs.sort(key=lambda e: e["f"])
        open_cl = []                              # [{last_f, last_pos, reads}]
        for e in evs:
            best, bestd = None, None
            for c in open_cl:
                dt = max(1, e["f"] - c["last_f"]) / FPS
                d = float(np.linalg.norm(e["pos"] - c["last_pos"]))
                if d <= max(120.0, a.split_speed_cms * dt):
                    if bestd is None or d < bestd:
                        best, bestd = c, d
            if best is None:
                open_cl.append({"last_f": e["f"], "last_pos": e["pos"], "reads": [e]})
            else:
                best["reads"].append(e)
                best["last_f"], best["last_pos"] = e["f"], e["pos"]
        for c in open_cl:
            if len(c["reads"]) >= a.min_cluster:
                clusters[cid] = {"num": num, "reads": c["reads"]}
                cid += 1
    print(f"anchor clusters (physical players): {len(clusters)} from {len(by_num)} numbers")
    per_num_counts = defaultdict(int)
    for c in clusters.values():
        per_num_counts[c["num"]] += 1
    print("  per number:", {f"#{n}": k for n, k in sorted(per_num_counts.items())})

    # ---- truth tracks per cluster (weighted per-frame consensus + interpolation) ----
    truth = {}
    for ci, c in clusters.items():
        byf = defaultdict(list)
        for e in c["reads"]:
            byf[e["f"]].append(e)
        tr = {}
        for f, es in byf.items():
            w = np.array([ZONE[e["cam"]] * e["conf"] for e in es])
            tr[f] = np.average([e["pos"] for e in es], axis=0, weights=w)
        ks = sorted(tr)
        full = dict(tr)
        for f0, f1 in zip(ks, ks[1:]):
            g = f1 - f0
            if 1 < g <= a.max_interp_s * FPS:
                for f in range(f0 + 1, f1):
                    full[f] = tr[f0] + (tr[f1] - tr[f0]) * ((f - f0) / g)
        truth[ci] = full

    # ---- map GT players to clusters via their (GT-verified) seed read ----
    gt = json.loads((REPO / f"data/gt_players/{key}.json").read_text())
    seeds = json.loads((REPO / f"runs/anchors/{key}.jerseyseeds.json").read_text())["seeds"]
    gt_cluster = {}
    for pl, cams in seeds.items():
        num = int("".join(ch for ch in pl if ch.isdigit()))
        best, bestn = None, -1
        for ci, c in clusters.items():
            if c["num"] != num:
                continue
            n = 0
            for ang, d in cams.items():
                sf = d["seed_frame"] - offs[ang]
                for e in c["reads"]:
                    if e["cam"] == ang and abs(e["f"] - sf) <= 2 and iou(e["box"], d["seed_box"]) >= 0.5:
                        n += 1
            if n > bestn:
                best, bestn = ci, n
        if best is not None:
            gt_cluster[pl] = best

    # masklet priors for GT players
    sam_by_pl = {}
    for pl in gt_cluster:
        safe = pl.replace("#", "n").replace(" ", "")
        sam = {}
        for ang in ANGLES:
            p = REPO / a.sam3_dir / f"{key}__{safe}__{ang}.json"
            if p.exists():
                sam[ang] = {int(f): r for f, r in json.loads(p.read_text())["frames"].items()
                            if r.get("present") and r.get("box")}
        sam_by_pl[pl] = sam
    cluster_pl = {ci: pl for pl, ci in gt_cluster.items()}

    # ---- B. joint per-camera assignment over ALL clusters ----
    assigned = defaultdict(dict)   # ci -> {(ang, f): box}
    all_frames = sorted({f for tr in truth.values() for f in tr})
    for ang in ANGLES:
        for f in all_frames:
            cf = f + offs[ang]
            cand = dets[ang].get(cf, [])
            present = [(ci, truth[ci][f]) for ci in truth if f in truth[ci]]
            if not cand or not present:
                continue
            det_court = [court(ang, b) for b in cand]
            nP, nD = len(present), len(cand)
            C = np.full((nP, nD + nP), 1e6)
            for i, (ci, tp) in enumerate(present):
                pl = cluster_pl.get(ci)
                mbox = None
                if pl:
                    mr = sam_by_pl.get(pl, {}).get(ang, {}).get(cf)
                    mbox = mr["box"] if mr else None
                for j, dc in enumerate(det_court):
                    d = float(np.linalg.norm(tp - dc))
                    if d <= a.gate_cm:
                        m = (1.0 - iou(mbox, cand[j])) * 60 if mbox else 30
                        C[i, j] = d + m
                C[i, nD + i] = a.dummy_cm
            rows, cols = linear_sum_assignment(C)
            for i, j in zip(rows, cols):
                if j < nD and C[i, j] < 1e5:
                    assigned[present[i][0]][(ang, f)] = cand[j]

    # ---- score GT players ----
    report = {}
    for pl, ci in gt_cluster.items():
        v = gt.get(pl, {})
        sel = {int(f): s for f, s in v.get("frames", {}).items()
               if s and str(f) in v.get("approved", {})}
        if len(sel) < 30:
            continue
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
                cb = assigned[ci].get((ang, f))
                if cb and iou(gb[s[ang]], cb) >= a.iou_hit:
                    nh += 1
            if nv >= 20:
                pc[ang] = round(nh / nv, 3)
        vals = list(pc.values())
        report[pl] = {"per_camera_fidelity": pc,
                      "all_angles": round(sum(vals) / len(vals), 3) if vals else None,
                      "cluster": ci, "truth_frames": len(truth[ci])}

    out = {"window": key,
           "method": "joint assignment, FULL-roster anchor truths, same-number split",
           "n_clusters": len(clusters), "players": report}
    print(json.dumps(out, indent=1))
    (REPO / f"runs/tracking/ledger/jointanchor_{key}.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
