#!/usr/bin/env python3
"""Seed boxes for the FULL ROSTER (all players on court) from confident jersey reads —
no ground truth. Needed so joint assignment has the complete roster tracked (mutual
exclusion only works when every player competes for detections).

For each jersey number with enough confident reads in the window, take the earliest
high-confidence read per camera as the SAM3 seed box.

Output: runs/anchors/{game}_{tag}.rosterseeds.json (same schema as .seeds.json)

  python scripts/extract_roster_seeds.py --game e6fba750 --tag 44_60
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")
OFFS = {"e6fba750_44_60": {"FL": 0, "FR": -11, "NL": -1, "NR": -1}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--min-conf", type=float, default=0.7)
    ap.add_argument("--min-reads", type=int, default=25, help="min total confident reads to seed a number")
    a = ap.parse_args()
    key = f"{a.game}_{a.tag}"
    offs = OFFS[key]

    anchors = json.loads((REPO / f"runs/anchors/{key}.jersey_anchors.json").read_text())["anchors"]
    total = Counter(ev["number"] for ev in anchors if ev.get("conf", 0) >= a.min_conf)
    roster = sorted(n for n, c in total.items() if c >= a.min_reads)

    # earliest high-conf read per (number, camera) -> seed box
    seeds = {}
    for num in roster:
        per_cam = {}
        for ang in ANGLES:
            cands = sorted((ev for ev in anchors
                            if ev["number"] == num and ev["cam"] == ang
                            and ev.get("conf", 0) >= a.min_conf), key=lambda e: e["frame"])
            if not cands:
                continue
            ev = cands[0]
            f0 = ev["frame"]
            per_cam[ang] = {"seed_frame": f0 + offs[ang], "seed_box": ev["box"],
                            "seed_conf": ev["conf"]}
        if per_cam:
            seeds[f"#{num}"] = per_cam

    out = REPO / f"runs/anchors/{key}.rosterseeds.json"
    out.write_text(json.dumps({"game": a.game, "tag": a.tag, "offsets": offs, "seeds": seeds}, indent=1))
    n_tracks = sum(len(c) for c in seeds.values())
    print(f"{key}: {len(seeds)} roster players, {n_tracks} tracks (reads: "
          f"{dict(sorted(((n, total[n]) for n in roster), key=lambda x: -x[1]))}) -> {out}")
    for pl, cams in seeds.items():
        print(f"  {pl}: {','.join(cams)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
