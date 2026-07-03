#!/usr/bin/env python3
"""Cross-check our per-camera tracking against the SAM3 reference pass (runs locally).

NOT accuracy-vs-ground-truth — an AGREEMENT analysis between two independent model
families. High agreement -> confidence; disagreements -> a ranked list of specific
frames/cameras to eyeball.

Per camera: Hungarian IoU>=0.5 matching per frame ->
  agree      : boxes both systems found
  ours_only  : we detect, SAM3 doesn't  (our FP, or SAM3 miss)
  sam3_only  : SAM3 detects, we don't   (our MISS candidates — the interesting ones)
plus per-frame count MAE and (if SAM3 tracked ids) an ID-consistency cross-check:
how often our track_id <-> SAM3 id pairing flips (a proxy for ID switches).

  python scripts/compare_reference.py --gid e6fba750 --tag 47_12
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def compare_cam(ours_by_f: dict, sam3_by_f: dict, iou_thr: float) -> dict:
    from scipy.optimize import linear_sum_assignment
    agree = ours_only = sam3_only = 0
    count_err = []
    pair_votes: dict[int, Counter] = defaultdict(Counter)   # our id -> sam3 id votes
    worst: list[tuple] = []
    frames = sorted(set(ours_by_f) | {int(k) for k in sam3_by_f})
    for f in frames:
        ours = ours_by_f.get(f, [])
        sam3 = sam3_by_f.get(str(f), sam3_by_f.get(f, []))
        count_err.append(abs(len(ours) - len(sam3)))
        if ours and sam3:
            cost = np.array([[1.0 - _iou(o["box_xyxy"], s["box"]) for s in sam3] for o in ours])
            rows, cols = linear_sum_assignment(cost)
            hit_o, hit_s = set(), set()
            for i, j in zip(rows, cols):
                if 1.0 - cost[i, j] >= iou_thr:
                    hit_o.add(i)
                    hit_s.add(j)
                    if sam3[j].get("id") is not None:
                        pair_votes[ours[i]["track_id"]][sam3[j]["id"]] += 1
            agree += len(hit_o)
            ours_only += len(ours) - len(hit_o)
            sam3_only += len(sam3) - len(hit_s)
            miss = len(sam3) - len(hit_s)
        else:
            ours_only += len(ours)
            sam3_only += len(sam3)
            miss = len(sam3)
        if miss:
            worst.append((miss, f))
    # ID-consistency: frames where our track matched a NON-majority sam3 id
    flips = tot = 0
    for votes in pair_votes.values():
        n = sum(votes.values())
        tot += n
        flips += n - votes.most_common(1)[0][1]
    worst.sort(reverse=True)
    return {"agree": agree, "ours_only": ours_only, "sam3_only": sam3_only,
            "agreement_rate": round(agree / max(1, agree + ours_only + sam3_only), 3),
            "sam3_recall_of_ours": round(agree / max(1, agree + ours_only), 3),
            "our_recall_of_sam3": round(agree / max(1, agree + sam3_only), 3),
            "count_mae": round(float(np.mean(count_err)), 2),
            "id_pairing_flip_rate": (round(flips / tot, 4) if tot else None),
            "worst_frames": [f for _, f in worst[:8]]}


def _court_filter(boxes_by_f: dict, calib: dict, margin: float, box_key: str):
    """Keep only boxes whose FOOT point projects inside the court (+margin cm) —
    removes bench/spectator detections so both systems are scored on the same scope."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from uball_cc.fusion.court import LENGTH, WIDTH
    from uball_cc.fusion.homography import project_pixels

    out = {}
    for f, rows in boxes_by_f.items():
        if not rows:
            out[f] = rows
            continue
        feet = [((r[box_key][0] + r[box_key][2]) / 2.0, r[box_key][3]) for r in rows]
        court = project_pixels(feet, calib)
        out[f] = [r for r, c in zip(rows, court)
                  if -margin <= c[0] <= LENGTH + margin and -margin <= c[1] <= WIDTH + margin]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", default="e6fba750")
    ap.add_argument("--tag", default="47_12")
    ap.add_argument("--sam3-dir", default="runs/sam3_ref")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--court-filter", action="store_true",
                    help="score only boxes whose foot projects onto the court (+margin)")
    ap.add_argument("--margin", type=float, default=200.0)
    a = ap.parse_args()

    report = {}
    for ang in ANGLES:
        ours_p = REPO / f"runs/tracking/{a.gid}_{ang}_{a.tag}_teams.json"
        sam3_p = REPO / a.sam3_dir / f"{a.gid}_{ang}_{a.tag}.sam3.json"
        if not (ours_p.exists() and sam3_p.exists()):
            print(f"{ang}: missing {'ours' if not ours_p.exists() else 'sam3'} — skipped")
            continue
        by_f: dict[int, list] = defaultdict(list)
        for t in json.loads(ours_p.read_text())["tracks"]:
            by_f[t["frame"]].append(t)
        sam3 = json.loads(sam3_p.read_text())
        sam3_frames = sam3["frames"]
        if a.court_filter:
            calib = json.loads((REPO / f"configs/calib/{ang}.json").read_text())
            by_f = _court_filter(by_f, calib, a.margin, "box_xyxy")
            sam3_frames = _court_filter(sam3_frames, calib, a.margin, "box")
        report[ang] = compare_cam(by_f, sam3_frames, a.iou)
        report[ang]["sam3_mode"] = sam3.get("mode")
        r = report[ang]
        print(f"{ang} [{r['sam3_mode']}]: agreement {r['agreement_rate']:.0%} | "
              f"we-find-what-sam3-finds {r['our_recall_of_sam3']:.0%} | "
              f"sam3-confirms-ours {r['sam3_recall_of_ours']:.0%} | count MAE {r['count_mae']} | "
              f"id-flip {r['id_pairing_flip_rate']} | worst frames {r['worst_frames'][:5]}")
    out = REPO / a.sam3_dir / f"compare_{a.gid}_{a.tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
