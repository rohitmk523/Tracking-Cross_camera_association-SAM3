#!/usr/bin/env python3
"""HYBRID tracking — NO SAM3: ByteTrack over cached detections for continuity,
jersey checkpoints for identity, same output schema as the SAM3 masklets so the
cross-camera correction and every scorer run unchanged.

The A/B question this answers: how much of the SAM3 pipeline's accuracy is SAM3's
mask propagation vs. the identity machinery around it? Runs on cached data — CPU
only, seconds, $0.

Mechanism per camera:
  1. ByteTrack (the production docs/05 tracker) over the cached RF-DETR detections.
  2. Every confident jersey read that IoU-matches a live track CLAIMS that track for
     that player from that frame (same semantics as SAM3 re-seed checkpoints; a later
     conflicting read re-claims — drift can only survive until the next read).
  3. Per (player, camera) box stream -> masklet-schema JSON.

  python scripts/hybrid_track.py --game e6fba750 --tag 44_60 --out-dir runs/hybrid_e6
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
OFFS = {"e6fba750_44_60": {"FL": 0, "FR": -11, "NL": -1, "NR": -1}}


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
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-conf", type=float, default=0.7, help="jersey read confidence to claim")
    ap.add_argument("--iou-claim", type=float, default=0.35)
    ap.add_argument("--dual-numbers", default="",
                    help="comma list; these numbers claim per (number,kit) -> separate nNB/nNW streams")
    a = ap.parse_args()

    from uball_cc.detection.base import Detection
    from uball_cc.tracking.tracker import ByteTrackTracker

    key = f"{a.game}_{a.tag}"
    adoc = json.loads((REPO / f"runs/anchors/{key}.jersey_anchors.json").read_text())
    offs = OFFS.get(key) or adoc["offsets"]
    duals = {int(x) for x in a.dual_numbers.split(",") if x.strip()}
    reads = defaultdict(list)                       # (cam, clip_frame) -> [(key, box, conf)]
    for ev in adoc["anchors"]:
        if ev.get("conf", 0) >= a.min_conf:
            cf = ev["frame"] + offs[ev["cam"]]
            num = int(ev["number"])
            if num in duals:
                kit = ev.get("kit")
                if kit not in ("B", "W"):
                    continue            # untagged read cannot claim a dual (rare: tags ~100%)
                key = f"{num}{kit}"     # SEPARATE stream per (number, kit)
            else:
                key = str(num)
            reads[(ev["cam"], cf)].append((key, ev["box"], ev["conf"]))

    outd = REPO / a.out_dir
    outd.mkdir(parents=True, exist_ok=True)
    summary = {}
    for ang in ANGLES:
        z = np.load(REPO / f"runs/dets_cache/{a.game}_{ang}_{a.tag}_small_1280_t0.25.dets.npz")
        by_f = defaultdict(list)
        for b, s, c, f in zip(z["boxes"], z["scores"], z["classes"], z["frame_idx"]):
            by_f[int(f)].append(Detection(tuple(float(v) for v in b), float(s), int(c)))

        bt = ByteTrackTracker(ang)
        owner = {}                                  # track_id -> (number, since_frame)
        streams = defaultdict(dict)                 # number -> {frame: box}
        n_claims = 0
        for f in sorted(by_f):
            tracks = bt.update(by_f[f], f)
            # claims: each confident read matched to the best-IoU live track
            for num, rbox, rconf in reads.get((ang, f), []):
                best, bi = 0.0, None
                for t in tracks:
                    v = iou(t.box_xyxy, rbox)
                    if v > best:
                        best, bi = v, t.track_id
                if bi is not None and best >= a.iou_claim:
                    if owner.get(bi, (None,))[0] != num:
                        owner[bi] = (num, f)
                        n_claims += 1
            # one track per number per frame: highest-conf claimed track wins
            per_num = {}
            for t in tracks:
                num = owner.get(t.track_id, (None,))[0]
                if num is None:
                    continue
                if num not in per_num or t.score > per_num[num].score:
                    per_num[num] = t
            for num, t in per_num.items():
                streams[num][f] = [round(v, 1) for v in t.box_xyxy]

        for num, st in streams.items():
            (outd / f"{key}__n{num}__{ang}.json").write_text(json.dumps(
                {"player": f"#{num}", "cam": ang, "hybrid": True,
                 "frames": {f: {"box": b, "score": 1.0, "mask_area": 0, "present": True}
                            for f, b in st.items()}}))
        summary[ang] = {"players": len(streams), "claims": n_claims}
        print(f"{ang}: {len(streams)} identified players, {n_claims} identity claims", flush=True)
    print(f"-> {outd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
