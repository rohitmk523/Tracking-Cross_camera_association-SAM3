#!/usr/bin/env python3
"""Re-ID / drift-correction layer over SAM3 single-object masklets (operator design).

SAM3 masks DRIFT onto same-kit team-mates in weak cameras. This layer catches and
corrects that using jersey number + geometry (no live dependency — offline, whole-clip):

  1. CONFIRM/DRIFT: attach dense jersey reads to each masklet frame. A read of the
     player's own number confirms; a read of a DIFFERENT roster number = drifted.
  2. TRUSTED POSITION: fuse the confirmed / uncontradicted cameras to a court position,
     rejecting geometric outliers (a drifted camera disagrees with the consensus).
  3. RE-ACQUIRE: where a camera drifted / lost / disagrees, project the trusted court
     position back into that camera and pick the nearest DETECTION whose jersey reads
     the player's number (or, absent a read, nearest of the right team) — replacing the
     drifted mask with the correct box.
  4. Re-fuse and score against operator GT (leakage-free: GT is the yardstick only).

  python scripts/solve_player_reid.py --game e6fba750 --tag 44_60
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
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
    ap.add_argument("--sam3-dir", default="runs/sam3_players")
    ap.add_argument("--iou-hit", type=float, default=0.3)
    ap.add_argument("--reacq-cm", type=float, default=150.0)
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
    # dense jersey anchors indexed by (cam, ref_frame) -> [(box, number)]
    anchors = defaultdict(list)
    ja = json.loads((REPO / f"runs/anchors/{key}.jersey_anchors.json").read_text())
    for ev in ja["anchors"]:
        anchors[(ev["cam"], ev["frame"])].append((ev["box"], int(ev["number"])))
    roster = {n for _, n in (x for lst in anchors.values() for x in lst)}
    gt = json.loads((REPO / f"data/gt_players/{key}.json").read_text())

    def court_of(ang, box):
        (cx, cy), = project_pixels([((box[0] + box[2]) / 2, box[3])], calib[ang])
        return np.array([cx, cy])

    report = {}
    for pl, v in gt.items():
        if len(v.get("approved", {})) < 30:
            continue
        num = None
        digits = "".join(ch for ch in pl if ch.isdigit())
        if digits and not pl.startswith("ref"):
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

        # --- per (cam, ref_frame): masklet box + confirm/drift by jersey ---
        state = {}       # (ang, f) -> ("box", box, flag)  flag in confirm/drift/none
        for ang in ANGLES:
            for f in frames:
                cf = f + offs[ang]
                r = sam.get(ang, {}).get(cf)
                if not r:
                    continue
                box = r["box"]
                flag = "none"
                for abox, anum in anchors.get((ang, f), []):
                    if iou(box, abox) >= 0.2:
                        if num is not None and anum == num:
                            flag = "confirm"
                        elif num is not None and anum in roster and anum != num:
                            flag = "drift"
                state[(ang, f)] = (box, flag)

        # --- trusted fused position per frame (confirmed/none, outlier-rejected) ---
        fused_pos = {}
        for f in frames:
            pts, ws = [], []
            for ang in ANGLES:
                st = state.get((ang, f))
                if st and st[1] != "drift":
                    pts.append(court_of(ang, st[0])); ws.append(ZONE[ang] * (2 if st[1] == "confirm" else 1))
            if not pts:
                continue
            pts = np.array(pts); ws = np.array(ws)
            med = np.median(pts, axis=0)
            keep = np.linalg.norm(pts - med, axis=1) <= 150.0
            if keep.sum() == 0:
                keep[:] = True
            fused_pos[f] = np.average(pts[keep], axis=0, weights=ws[keep])

        # --- re-acquire drifted/lost cameras from the trusted position + jersey ---
        corrected = {}   # (ang, f) -> box  (final box used for scoring in that cam)
        for ang in ANGLES:
            for f in frames:
                st = state.get((ang, f))
                if st and st[1] != "drift":
                    corrected[(ang, f)] = st[0]
                    continue
                # drifted or lost: re-acquire in COURT space — of this camera's cached
                # detections, pick the one whose court position is nearest the trusted
                # fused position (jersey read of the right number = strong bonus)
                if f not in fused_pos:
                    continue
                cf = f + offs[ang]
                best, bestd = None, a.reacq_cm
                for b in dets[ang].get(cf, []):
                    jmatch = any(iou(b, ab) >= 0.3 and an == num
                                 for ab, an in anchors.get((ang, f), []))
                    d = float(np.linalg.norm(court_of(ang, b) - fused_pos[f])) - (80 if jmatch else 0)
                    if d < bestd:
                        best, bestd = b, d
                if best is not None:
                    corrected[(ang, f)] = best

        # --- score vs GT: fused coverage + per-cam fidelity, using corrected boxes ---
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
                    gpos.append(court_of(ang, gb[s[ang]])); gw.append(ZONE[ang])
                cb = corrected.get((ang, f))
                if cb and s[ang] < len(gb) and iou(gb[s[ang]], cb) >= a.iou_hit:
                    spos.append(court_of(ang, cb)); sw.append(ZONE[ang]); hit = True
            if hit:
                fused_hit += 1
                if gpos and spos:
                    court_err.append(float(np.linalg.norm(
                        np.average(gpos, axis=0, weights=gw) - np.average(spos, axis=0, weights=sw))))
        report[pl] = {"fused_coverage": round(fused_hit / max(1, fused_n), 3),
                      "per_camera_fidelity": percam,
                      "court_err_cm_median": round(float(np.median(court_err)), 1) if court_err else None,
                      "gt_frames": fused_n, "number": num}

    out = {"window": key, "method": "SAM3 single-object + jersey/geometry re-ID (drift-corrected)",
           "players": report}
    print(json.dumps(out, indent=1))
    (REPO / f"runs/tracking/ledger/sam3reid_{key}.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
