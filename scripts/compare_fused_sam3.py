#!/usr/bin/env python3
"""FUSED-SYSTEM parity vs SAM3: is our real-time fusion as good as the big offline model?

Builds SAM3's own multi-camera CONSENSUS people (court-filtered boxes from every camera,
projected to the court, clustered; >=2 cameras agreeing = a real on-court person) and
scores our fused worldstate against them per frame:

  person recall   — of SAM3-consensus people, how many do we have within match_cm?
  person precision— of our fused identities, how many does SAM3 corroborate?
  position RMSE   — court-cm disagreement on matched pairs
  ID stability    — link SAM3 consensus clusters into pseudo-tracks; a good fused
                    system maps each pseudo-track to ONE global id (switch rate).

NOT ground truth — SAM3 is the strongest available reference; agreement = parity
evidence, disagreement = a ranked inspection list.

  python scripts/compare_fused_sam3.py --gid e6fba750 --tag 47_12 \
      --worldstate runs/tracking/e6_worldstate_v4.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ANGLES = ("FL", "FR", "NL", "NR")
CLUSTER_CM = 120.0            # SAM3 cross-camera consensus radius
LINK_CM = 100.0               # pseudo-track frame-to-frame link radius


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--worldstate", required=True)
    ap.add_argument("--sam3-dir", default="runs/sam3_ref")
    ap.add_argument("--clip-dir", default="data/clips")
    ap.add_argument("--calib", default="configs/calib")
    ap.add_argument("--ref", default="FL")
    ap.add_argument("--match-cm", type=float, default=200.0,
                    help="identity-presence radius: SAM3 mask-box feet vs our detector-box feet differ ~60cm RMSE systematically")
    ap.add_argument("--min-cams", type=int, default=2)
    a = ap.parse_args()

    from scipy.optimize import linear_sum_assignment

    from uball_cc.fusion.audiosync import audio_offset_seconds
    from uball_cc.fusion.court import LENGTH, WIDTH
    from uball_cc.fusion.homography import load_calib, project_pixels

    # --- SAM3 boxes -> court points per camera (synced to the ref timeline) ---
    pts_by_frame: dict[int, list] = defaultdict(list)     # frame -> [(cam, x, y)]
    sam3_tracks: dict[tuple, list] = defaultdict(list)    # (cam, sam3_id) -> [(frame, x, y)]
    for ang in ANGLES:
        sp = Path(a.sam3_dir) / f"{a.gid}_{ang}_{a.tag}.sam3.json"
        if not sp.exists():
            print(f"{ang}: no SAM3 json — skipped")
            continue
        calib = load_calib(f"{a.calib}/{ang}.json")
        off = 0
        ref_clip = Path(a.clip_dir) / f"{a.gid}_{a.ref}_{a.tag}.mp4"
        clip = Path(a.clip_dir) / f"{a.gid}_{ang}_{a.tag}.mp4"
        if ang != a.ref and ref_clip.exists() and clip.exists():
            off_s, _ = audio_offset_seconds(str(ref_clip), str(clip))
            off = int(round(off_s * 29.97))
        d = json.loads(sp.read_text())
        for f_str, rows in d["frames"].items():
            feet = [((r["box"][0] + r["box"][2]) / 2, r["box"][3]) for r in rows]
            if not feet:
                continue
            court = project_pixels(feet, calib)
            for r, (cx, cy) in zip(rows, court):
                if -100 <= cx <= LENGTH + 100 and -100 <= cy <= WIDTH + 100:
                    f = int(f_str) - off
                    pts_by_frame[f].append((ang, float(cx), float(cy)))
                    if r.get("id") is not None:
                        sam3_tracks[(ang, r["id"])].append((f, float(cx), float(cy)))

    # --- consensus clustering per frame (>= min_cams distinct cameras) ---
    consensus: dict[int, list] = {}
    for f, pts in pts_by_frame.items():
        groups: list[list] = []
        for cam, x, y in sorted(pts):
            placed = False
            for g in groups:
                gx = np.mean([p[1] for p in g])
                gy = np.mean([p[2] for p in g])
                if (x - gx) ** 2 + (y - gy) ** 2 <= CLUSTER_CM ** 2 \
                        and all(p[0] != cam for p in g):
                    g.append((cam, x, y))
                    placed = True
                    break
            if not placed:
                groups.append([(cam, x, y)])
        consensus[f] = [(float(np.mean([p[1] for p in g])), float(np.mean([p[2] for p in g])))
                        for g in groups if len({p[0] for p in g}) >= a.min_cams]

    # --- our fused worldstate ---
    ws = json.loads(Path(a.worldstate).read_text())
    ours = {fr["frame"]: [(t["global_id"], t["court_xy"][0], t["court_xy"][1])
                          for t in fr["tracks"]] for fr in ws["frames"]}

    # --- per-frame matching (person presence + position) ---
    rec_n = rec_hit = prec_n = prec_hit = 0
    sq = []
    for f in sorted(consensus):
        cons = consensus[f]
        mine = ours.get(f, [])
        if cons and mine:
            cost = np.array([[np.hypot(cx - x, cy - y) for _, x, y in mine] for cx, cy in cons])
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] <= a.match_cm:
                    rec_hit += 1
                    prec_hit += 1
                    sq.append(cost[i, j] ** 2)
        rec_n += len(cons)
        prec_n += len(mine)

    # --- ID stability via SAM3's OWN video-tracker ids: each long SAM3 track should map
    # to ONE of our global ids (purity) ---
    long_tracks = []
    for (cam, sid), obs in sam3_tracks.items():
        if len(obs) < 30:
            continue
        gids = Counter()
        for f, x, y in obs:
            mine = ours.get(f, [])
            if not mine:
                continue
            d2, gid = min(((mx - x) ** 2 + (my - y) ** 2, g) for g, mx, my in mine)
            if d2 <= a.match_cm ** 2:
                gids[gid] += 1
        if sum(gids.values()) >= 20:
            long_tracks.append(gids)
    pure = sum(1 for g in long_tracks
               if g.most_common(1)[0][1] / sum(g.values()) >= 0.9)
    switches = sum(sum(g.values()) - g.most_common(1)[0][1] for g in long_tracks)
    total_links = sum(sum(g.values()) for g in long_tracks)
    rmse = float(np.sqrt(np.mean(sq))) if sq else None
    report = {
        "window": f"{a.gid}_{a.tag}",
        "person_recall_vs_sam3_consensus": round(rec_hit / max(1, rec_n), 3),
        "person_precision_sam3_corroborated": round(prec_hit / max(1, prec_n), 3),
        "position_rmse_cm": round(rmse, 1) if rmse else None,
        "id_stability": {"sam3_tracks_30f+": len(long_tracks),
                         "mapped_to_one_gid_90pct": pure,
                         "impurity_rate": round(switches / max(1, total_links), 4)},
        "frames": len(consensus), "match_cm": a.match_cm, "min_cams": a.min_cams,
    }
    print(json.dumps(report, indent=2))
    out = REPO / f"runs/sam3_ref/fused_parity_{a.gid}_{a.tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
