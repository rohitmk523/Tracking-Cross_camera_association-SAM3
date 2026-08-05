#!/usr/bin/env python3
"""Score a holder timeline against annotator ground truth.

THE metric for the possession pipeline. GT format = docs/POSSESSION_GT_BRIEF.md
(contiguous segments of {start,end,holder,conf}, holder null = nobody has it,
conf "unsure" = excluded from scoring).

Reports, over the GT window:
  ALL-FRAME accuracy   — the honest number (nobody-frames included)
  HELD-frame accuracy  — of frames where a player really has the ball
  NOBODY-frame accuracy— do we correctly say nobody
  error anatomy        — wrong-player / wrong-kit / phantom-hold / missed-hold
  change-point recall  — did we switch holder when possession actually changed

  .venv/bin/python scripts/score_holders_gt.py --gt gt_c2a354fe_600_180.json \
      --holders runs/tracking/ledger/holders_c2ademo.json [--label OLD]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def num_of(sid):
    return int(str(sid).lstrip("#n").rstrip("BW"))


def load_gt(path):
    d = json.loads(Path(path).read_text())
    gt, unsure = {}, set()
    for s in d["segments"]:
        for f in range(s["start"], s["end"] + 1):
            if s["conf"] != "sure":
                unsure.add(f)
            else:
                gt[f] = s["holder"]
    return d, gt, unsure


def load_pred(path):
    d = json.loads(Path(path).read_text())
    p = {}
    for a, b, sid in d["segments"]:
        for f in range(a, b + 1):
            p[f] = sid
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--holders", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--roster", default=None,
                    help="score only players this roster can represent")
    a = ap.parse_args()
    doc, gt, unsure = load_gt(a.gt)
    pred = load_pred(a.holders)
    known = None
    if a.roster:
        r = json.loads(Path(a.roster).read_text())
        cnt = Counter(p["num"] for p in r["players"])
        known = set()
        for n, k in cnt.items():
            known |= {f"n{n}B", f"n{n}W"} if k > 1 else {f"n{n}"}

    frames = sorted(gt)
    held = [f for f in frames if gt[f] is not None]
    nobody = [f for f in frames if gt[f] is None]

    ok_held = sum(1 for f in held if pred.get(f) == gt[f])
    ok_nobody = sum(1 for f in nobody if pred.get(f) is None)
    ok_all = ok_held + ok_nobody

    err = Counter()
    for f in held:
        p, g = pred.get(f), gt[f]
        if p == g:
            continue
        err["missed-hold (we say nobody)"] += p is None
        if p is not None:
            err["wrong kit, right number" if num_of(p) == num_of(g)
                else "wrong player"] += 1
    phantom = sum(1 for f in nobody if pred.get(f) is not None)

    # change points: GT possession changes (collapse on holder, ignore referee)
    cps, prev = [], "INIT"
    for f in frames:
        if gt[f] != prev:
            if prev != "INIT":
                cps.append(f)
            prev = gt[f]
    hit = 0
    for f in cps:                       # did we switch to the right holder within 0.5s?
        if any(pred.get(g) == gt[f] for g in range(f, f + 15)):
            hit += 1

    lbl = f" [{a.label}]" if a.label else ""
    print(f"\n=== {Path(a.holders).name} vs {Path(a.gt).name}{lbl} ===")
    print(f"  scored frames {len(frames)} (excluded unsure: {len(unsure)})")
    print(f"  ALL-FRAME accuracy    {ok_all:5d}/{len(frames)} = {ok_all/len(frames):6.1%}   <- THE number")
    print(f"    held frames         {ok_held:5d}/{len(held)} = {ok_held/max(len(held),1):6.1%}")
    print(f"    nobody frames       {ok_nobody:5d}/{len(nobody)} = {ok_nobody/max(len(nobody),1):6.1%}")
    if known is not None:
        reach = [f for f in held if gt[f] in known]
        okr = sum(1 for f in reach if pred.get(f) == gt[f])
        print(f"    held & representable{okr:5d}/{len(reach)} = {okr/max(len(reach),1):6.1%} "
              f"(roster covers {len(reach)}/{len(held)} held frames)")
    print(f"  change points {len(cps)} | correct switch within 0.5s: {hit} ({hit/max(len(cps),1):.0%})")
    print("  error anatomy (of held frames):")
    for k, v in err.most_common():
        print(f"    {k:32s} {v:5d} ({v/max(len(held),1):5.1%})")
    print(f"    {'phantom hold (GT nobody)':32s} {phantom:5d} "
          f"({phantom/max(len(nobody),1):5.1%} of nobody frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
