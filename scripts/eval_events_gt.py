#!/usr/bin/env python3
"""Score derived possession/pass/turnover events against operator ground truth (audit P1).

GT comes from scripts/annotate_gt_events.py (label on the REF camera's clip so frame
indices line up with the fused timeline). Derived events come from a worldstate.json
(ws["events"]) or a demo/e2e JSON carrying {"events": {...}}.

Reports:
  - possession: frame-level team agreement (A/B/loose) + confusion matrix
  - pass/turnover: precision / recall / F1 with a ±tolerance frame match

  python scripts/eval_events_gt.py --gt data/gt_events/e6fba750_FL_47_12.json \
      --events runs/tracking/e6_worldstate_v2.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _holder_by_frame(gt: dict) -> dict[int, str]:
    out, holder = {}, "loose"
    bps = sorted(gt["breakpoints"], key=lambda b: b["frame"])
    j = 0
    for f in range(gt["n_frames"]):
        while j < len(bps) and bps[j]["frame"] <= f:
            holder = bps[j]["holder"]
            j += 1
        out[f] = holder
    return out


def _derived_team_by_frame(events: list[dict]) -> dict[int, str]:
    out: dict[int, str] = {}
    for e in events:
        if e.get("event") == "possession" and e.get("frame_window"):
            sf, ef = e["frame_window"]
            for f in range(int(sf), int(ef) + 1):
                out[f] = e.get("team") or "?"
    return out


def _match_moments(gt_m: list[dict], dv: list[dict], tol: int) -> dict:
    per = {}
    for kind in ("pass", "turnover"):
        g = sorted(m["frame"] for m in gt_m if m["type"] == kind)
        d = sorted(e["frame"] for e in dv if e.get("event") == kind and e.get("frame") is not None)
        used = set()
        tp = 0
        for gf in g:
            hit = next((k for k, df in enumerate(d)
                        if k not in used and abs(df - gf) <= tol), None)
            if hit is not None:
                used.add(hit)
                tp += 1
        p = tp / len(d) if d else (1.0 if not g else 0.0)
        r = tp / len(g) if g else 1.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per[kind] = {"gt": len(g), "derived": len(d), "tp": tp,
                     "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}
    return per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--events", required=True, help="worldstate.json or any JSON with ['events']")
    ap.add_argument("--tol", type=int, default=15, help="± frames to match a pass/turnover moment")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    gt = json.loads(Path(a.gt).read_text())
    src = json.loads(Path(a.events).read_text())
    ev = src.get("events", src)
    events = ev.get("events", [])

    gt_h = _holder_by_frame(gt)
    dv_h = _derived_team_by_frame(events)
    conf: Counter = Counter()
    agree = n = 0
    for f, g in gt_h.items():
        d = dv_h.get(f, "loose")
        conf[(g, d)] += 1
        agree += (g == d)
        n += 1
    moments = _match_moments(gt.get("moments", []), events, a.tol)

    report = {"n_frames": n, "possession_frame_accuracy": round(agree / n, 3) if n else None,
              "possession_confusion_gt_vs_derived":
                  {f"{g}->{d}": c for (g, d), c in conf.most_common()},
              "moments": moments, "tolerance_frames": a.tol}
    print(json.dumps(report, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
