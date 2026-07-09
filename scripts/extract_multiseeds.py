#!/usr/bin/env python3
"""Common-frame MULTI-OBJECT seeds: per camera, pick the frame where the most roster
players are identity-confirmed (confident jersey read within ±15 frames whose box
matches a detection at that frame), so ONE SAM3 pass can seed them all together.

No ground truth: reads + cached detections only (production recipe).
Output: runs/anchors/{game}_{tag}.multiseeds.json
  {cam: {"seed_frame": F(clip timeline), "players": {"#11": box, ...}}}

  python scripts/extract_multiseeds.py --game e6fba750 --tag 44_60
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
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
    ap.add_argument("--min-conf", type=float, default=0.7)
    ap.add_argument("--min-reads", type=int, default=25, help="reads for a number to count as roster")
    ap.add_argument("--near", type=int, default=15, help="read-to-frame max distance (frames)")
    ap.add_argument("--iou-attach", type=float, default=0.4)
    a = ap.parse_args()
    key = f"{a.game}_{a.tag}"
    adoc = json.loads((REPO / f"runs/anchors/{key}.jersey_anchors.json").read_text())
    offs = OFFS.get(key) or adoc["offsets"]

    anchors = adoc["anchors"]
    total = Counter(ev["number"] for ev in anchors if ev.get("conf", 0) >= a.min_conf)
    roster = {n for n, c in total.items() if c >= a.min_reads}

    out = {}
    for ang in ANGLES:
        z = np.load(REPO / f"runs/dets_cache/{a.game}_{ang}_{a.tag}_small_1280_t0.25.dets.npz")
        dets = defaultdict(list)
        for b, s, c, f in zip(z["boxes"], z["scores"], z["classes"], z["frame_idx"]):
            if int(c) == 0 and float(s) >= 0.3:
                dets[int(f)].append([float(v) for v in b])
        reads = [ev for ev in anchors
                 if ev["cam"] == ang and ev["number"] in roster and ev.get("conf", 0) >= a.min_conf]

        def confirmed_at(F):
            """{label: det_box} identity-confirmed at clip frame F. A number worn by
            two players (both teams) yields two instances: #3a, #3b — each a distinct
            read-confirmed detection."""
            claimed = []                           # [(number, det_box, read_frame)]
            for ev in sorted(reads, key=lambda e: abs(e["frame"] + offs[ang] - F)):
                cf = ev["frame"] + offs[ang]       # read on clip timeline
                if abs(cf - F) > a.near:
                    continue
                best, bi = 0.0, None
                for db in dets.get(F, []):
                    v = iou(db, ev["box"])
                    if v > best:
                        best, bi = v, db
                if best < a.iou_attach:
                    continue
                if any(iou(bi, b) > 0.5 for _, b, _ in claimed):
                    continue                       # detection already claimed
                prior = [c for c in claimed if c[0] == ev["number"]]
                # 2nd instance of a number = two players (both teams) wear it; require
                # near-simultaneous reads at distinct bodies, else it's the same player
                # re-matched after moving. Max 2 (one per team).
                if prior and (len(prior) >= 2 or abs(prior[0][2] - cf) > 4):
                    continue
                claimed.append((ev["number"], bi, cf))
            got, seen = {}, Counter()
            for num, b, _ in sorted(claimed, key=lambda c: (c[0], c[2])):
                suf = "ab"[seen[num]] if sum(1 for n, _, _ in claimed if n == num) > 1 else ""
                seen[num] += 1
                got[f"#{num}{suf}"] = b
            return got

        # candidate frames: sample every 5th frame with detections
        cand = sorted(dets)[::5]
        best_f, best_got = None, {}
        for F in cand:
            got = confirmed_at(F)
            if len(got) > len(best_got):
                best_f, best_got = F, got
        out[ang] = {"seed_frame": best_f,
                    "players": {f"#{n}": [round(v, 1) for v in b] for n, b in sorted(best_got.items())}}
        print(f"{ang}: seed frame {best_f} -> {len(best_got)} players confirmed "
              f"({sorted(best_got)})")

    p = REPO / f"runs/anchors/{key}.multiseeds.json"
    p.write_text(json.dumps({"game": a.game, "tag": a.tag, "offsets": offs, "cams": out}, indent=1))
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
