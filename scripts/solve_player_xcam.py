#!/usr/bin/env python3
"""Cross-camera jersey-anchored correction (operator design).

If ANY camera confidently reads the tracked player's number, that camera's box IS the
player -> we know his court position that instant. Project it into every other camera
and pick the detection sitting there, overriding whatever SAM3 drifted onto. Because we
are offline, positions BETWEEN two number readings are interpolated (past + future
anchors), so a wrong track can only last from one reading to the next.

Per frame, per player (number N):
  1. ANCHOR cameras = those whose SAM3 mask is confirmed by a jersey read of N.
  2. TRUTH position = anchor cameras' court position; between anchored frames, linearly
     interpolated from the nearest anchored frames on each side.
  3. CORRECT every camera: anchor cameras keep their box; others take the detection
     nearest the truth position (this overrides drift). No detection near truth -> lost.
  4. Fuse corrected positions; score vs operator GT (leakage-free).

  python scripts/solve_player_xcam.py --game e6fba750 --tag 44_60 --sam3-dir runs/sam3_players_jersey
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
    ap.add_argument("--iou-hit", type=float, default=0.3)
    ap.add_argument("--reacq-cm", type=float, default=180.0)
    ap.add_argument("--max-interp-frames", type=int, default=150,
                    help="don't interpolate truth across gaps longer than this (~5s)")
    a = ap.parse_args()
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
    anchors = defaultdict(list)   # (cam, ref_frame) -> [(box, number)]
    for ev in json.loads((REPO / f"runs/anchors/{key}.jersey_anchors.json").read_text())["anchors"]:
        anchors[(ev["cam"], ev["frame"])].append((ev["box"], int(ev["number"])))
    gt = json.loads((REPO / f"data/gt_players/{key}.json").read_text())

    def court(ang, box):
        (x, y), = project_pixels([((box[0] + box[2]) / 2, box[3])], calib[ang])
        return np.array([x, y])

    report = {}
    for pl, v in gt.items():
        digits = "".join(ch for ch in pl if ch.isdigit())
        if not digits or pl.startswith("ref"):
            continue
        num = int(digits)
        safe = pl.replace("#", "n").replace(" ", "")
        sam = {}
        for ang in ANGLES:
            p = REPO / a.sam3_dir / f"{key}__{safe}__{ang}.json"
            if p.exists():
                sam[ang] = {int(f): r for f, r in json.loads(p.read_text())["frames"].items()
                            if r.get("present") and r.get("box")}
        if not sam:
            continue
        sel = {int(f): s for f, s in v["frames"].items() if s and str(f) in v.get("approved", {})}
        frames = sorted(sel)

        # --- 1. anchor court positions: frames where a jersey read of N confirms a SAM3 mask ---
        anchor_pos = {}                                # ref_frame -> court position (from anchors)
        for f in frames:
            pts, ws = [], []
            for ang in ANGLES:
                cf = f + offs[ang]
                sr = sam.get(ang, {}).get(cf)
                if not sr:
                    continue
                for abox, anum in anchors.get((ang, f), []):
                    if anum == num and iou(sr["box"], abox) >= 0.3:
                        pts.append(court(ang, sr["box"])); ws.append(ZONE[ang])
                        break
            if pts:
                anchor_pos[f] = np.average(pts, axis=0, weights=ws)

        # --- 2. truth position: anchors, linearly interpolated across short gaps ---
        akeys = sorted(anchor_pos)
        truth = dict(anchor_pos)
        for a0, a1 in zip(akeys, akeys[1:]):
            gap = a1 - a0
            if 1 < gap <= a.max_interp_frames:
                p0, p1 = anchor_pos[a0], anchor_pos[a1]
                for f in range(a0 + 1, a1):
                    if f in sel:
                        truth[f] = p0 + (p1 - p0) * ((f - a0) / gap)

        # --- 3. correct every camera each frame from truth ---
        corrected = {}   # (ang, f) -> box
        for f in frames:
            tp = truth.get(f)
            for ang in ANGLES:
                cf = f + offs[ang]
                sr = sam.get(ang, {}).get(cf)
                # anchor camera keeps its (confirmed) box
                is_anchor = sr and any(anum == num and iou(sr["box"], abox) >= 0.3
                                       for abox, anum in anchors.get((ang, f), []))
                if is_anchor:
                    corrected[(ang, f)] = sr["box"]
                    continue
                if tp is None:
                    # no truth this frame: trust SAM3 mask as-is (best we have)
                    if sr:
                        corrected[(ang, f)] = sr["box"]
                    continue
                # override: nearest detection in this camera to the truth position
                best, bestd = None, a.reacq_cm
                for b in dets[ang].get(cf, []):
                    d = float(np.linalg.norm(court(ang, b) - tp))
                    if d < bestd:
                        best, bestd = b, d
                if best is not None:
                    corrected[(ang, f)] = best
                elif sr:
                    corrected[(ang, f)] = sr["box"]

        # --- 4. score vs GT ---
        percam = {}
        fused_hit = fused_n = 0
        court_err = []
        for ang in ANGLES:
            n_vis = n_hit = 0
            for f in frames:
                if ang not in sel[f]:
                    continue
                cf = f + offs[ang]
                gb = dets[ang].get(cf, [])
                if sel[f][ang] >= len(gb):
                    continue
                n_vis += 1
                cb = corrected.get((ang, f))
                if cb and iou(gb[sel[f][ang]], cb) >= a.iou_hit:
                    n_hit += 1
            if n_vis >= 20:
                percam[ang] = round(n_hit / n_vis, 3)
        for f in frames:
            s = sel[f]
            if not s:
                continue
            fused_n += 1
            gpos, gw, spos, sw = [], [], [], []
            hit = False
            for ang in s:
                cf = f + offs[ang]
                gb = dets[ang].get(cf, [])
                if s[ang] < len(gb):
                    gpos.append(court(ang, gb[s[ang]])); gw.append(ZONE[ang])
                cb = corrected.get((ang, f))
                if cb and s[ang] < len(gb) and iou(gb[s[ang]], cb) >= a.iou_hit:
                    spos.append(court(ang, cb)); sw.append(ZONE[ang]); hit = True
            if hit:
                fused_hit += 1
                if gpos and spos:
                    court_err.append(float(np.linalg.norm(
                        np.average(gpos, axis=0, weights=gw) - np.average(spos, axis=0, weights=sw))))
        report[pl] = {"fused_coverage": round(fused_hit / max(1, fused_n), 3),
                      "per_camera_fidelity": percam,
                      "court_err_cm_median": round(float(np.median(court_err)), 1) if court_err else None,
                      "anchor_frames": len(anchor_pos), "truth_frames": len(truth),
                      "gt_frames": fused_n}

    out = {"window": key, "method": "cross-camera jersey-anchored correction (2D-mapped)",
           "players": report}
    print(json.dumps(out, indent=1))
    (REPO / f"runs/tracking/ledger/sam3xcam_{key}.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
