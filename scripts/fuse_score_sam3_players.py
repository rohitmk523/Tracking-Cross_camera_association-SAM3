#!/usr/bin/env python3
"""Fuse the 4 per-camera SAM3 single-object masklets per player into ONE court track
(pure geometry, no jersey) and score against operator ground truth.

For each player:
  - project each camera's per-frame SAM3 box (foot point) to court cm via homography
  - per frame, fuse whichever cameras hold the player into one court position
    (region-of-confidence weighted: near cam on its own half > far cam)
  - the player is TRACKED whenever >=1 camera holds him (the union — a camera that
    loses him is carried by the others; when it regains him, geometry re-attaches)

Scoring (leakage-free — GT used only as the yardstick, never fed back):
  - per-camera fidelity : of GT frames where the player is visible in cam C, fraction
    where SAM3's box in C overlaps the GT box (IoU >= 0.3). Did SAM3 hold the RIGHT
    person in that view?
  - fused coverage       : of GT frames where the player is visible in ANY camera,
    fraction where the fused track is on him (>=1 camera matches). The union.
  - court error          : median cm between fused-SAM3 and fused-GT court position.

  python scripts/fuse_score_sam3_players.py --game e6fba750 --tag 44_60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
ANGLES = ("FL", "FR", "NL", "NR")
OFFS = {"e6fba750_44_60": {"FL": 0, "FR": -11, "NL": -1, "NR": -1},
        "c2a354fe_300_60": {"FL": 0, "FR": 1, "NL": 2, "NR": -1}}
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
    ap.add_argument("--out-worldstate", default=None)
    a = ap.parse_args()
    from uball_cc.fusion.homography import load_calib, project_pixels

    key = f"{a.game}_{a.tag}"
    offs = OFFS[key]
    calib = {ang: load_calib(str(REPO / f"configs/calib/{ang}.json")) for ang in ANGLES}

    # cached dets (to resolve GT click indices -> boxes)
    dets = {}
    for ang in ANGLES:
        z = np.load(REPO / f"runs/dets_cache/{a.game}_{ang}_{a.tag}_small_1280_t0.25.dets.npz")
        m = {}
        for b, s, c, f in zip(z["boxes"], z["scores"], z["classes"], z["frame_idx"]):
            if int(c) in (0, 1):
                m.setdefault(int(f), []).append([float(v) for v in b])
        dets[ang] = m
    gt = json.loads((REPO / f"data/gt_players/{key}.json").read_text())

    def foot_court(ang, box):
        (cx, cy), = project_pixels([((box[0] + box[2]) / 2, box[3])], calib[ang])
        return np.array([cx, cy])

    report = {}
    ws_players, ws_frames_acc = [], {}
    gid = 0
    for pl, v in gt.items():
        if len(v.get("approved", {})) < 30:
            continue
        # load SAM3 masklets for this player
        safe = pl.replace("#", "n").replace(" ", "")
        sam = {}
        for ang in ANGLES:
            p = REPO / a.sam3_dir / f"{a.game}_{a.tag}__{safe}__{ang}.json"
            if p.exists():
                d = json.loads(p.read_text())
                sam[ang] = {int(f): r for f, r in d["frames"].items() if r.get("present")}
        if not sam:
            continue
        sel = {int(f): s for f, s in v["frames"].items() if s and str(f) in v.get("approved", {})}

        percam = {}
        fused_hit = fused_n = 0
        court_err = []
        for ang in ANGLES:
            n_vis = n_hit = 0
            for f, s in sel.items():
                if ang not in s:
                    continue
                cf = f + offs[ang]
                gt_boxes = dets[ang].get(cf, [])
                if s[ang] >= len(gt_boxes):
                    continue
                n_vis += 1
                sr = sam.get(ang, {}).get(cf)
                if sr and sr.get("box") and iou(gt_boxes[s[ang]], sr["box"]) >= a.iou_hit:
                    n_hit += 1
            if n_vis >= 20:
                percam[ang] = {"gt_visible": n_vis, "sam3_held": n_hit,
                               "fidelity": round(n_hit / n_vis, 3)}
        # fused coverage: over GT frames where visible in ANY camera
        for f, s in sel.items():
            if not s:
                continue
            fused_n += 1
            # GT fused court position
            gpos, gw = [], []
            spos, sw = [], []
            for ang in s:
                cf = f + offs[ang]
                gb = dets[ang].get(cf, [])
                if s[ang] < len(gb):
                    gpos.append(foot_court(ang, gb[s[ang]])); gw.append(ZONE[ang])
                sr = sam.get(ang, {}).get(cf)
                if sr and sr.get("box"):
                    # count as a fused hit only if SAM3 box overlaps the GT box (right person)
                    if s[ang] < len(gb) and iou(gb[s[ang]], sr["box"]) >= a.iou_hit:
                        spos.append(foot_court(ang, sr["box"])); sw.append(ZONE[ang])
            if spos:
                fused_hit += 1
                if gpos:
                    gp = np.average(gpos, axis=0, weights=gw)
                    sp = np.average(spos, axis=0, weights=sw)
                    court_err.append(float(np.linalg.norm(gp - sp)))
        report[pl] = {
            "per_camera": percam,
            "fused_coverage": round(fused_hit / max(1, fused_n), 3),
            "gt_frames": fused_n,
            "court_err_cm_median": round(float(np.median(court_err)), 1) if court_err else None,
        }
        gid += 1

    out = {"window": key, "method": "SAM3 single-object x4 -> geometric fusion (no jersey)",
           "players": report}
    print(json.dumps(out, indent=1))
    led = REPO / f"runs/tracking/ledger/sam3players_{key}.json"
    led.parent.mkdir(parents=True, exist_ok=True)
    led.write_text(json.dumps(out, indent=1))
    print(f"-> {led}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
