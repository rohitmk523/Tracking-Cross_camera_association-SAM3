#!/usr/bin/env python3
"""Convert the annotation tool's internal STATE file into the export GT format.

The tool autosaves `state_<game>_<start>_<dur>.json` (change-points + review
flags + cursor). The scorer wants the exported format from
docs/POSSESSION_GT_BRIEF.md (contiguous segments). This does the expansion and
validates the result, so a state file is never scored by hand.

  .venv/bin/python scripts/gt_state_to_export.py state_c2a354fe_2400_300.json \
      --out data/gt/gt_c2a354fe_2400_300.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

FPS = 29.97


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--out", required=True)
    ap.add_argument("--annotator", default="akhilesh")
    a = ap.parse_args()
    d = json.loads(Path(a.state).read_text())
    base, n = d["frame_base"], d["nframes"]
    ch = sorted(d["changes"], key=lambda c: c["frame"])
    unreviewed = [c for c in ch if not c.get("reviewed")]
    if unreviewed:
        print(f"WARNING: {len(unreviewed)} unreviewed change(s) — "
              f"first at frame {unreviewed[0]['frame']}")
    if ch[0]["frame"] != base:
        print(f"WARNING: first change at {ch[0]['frame']} but window starts {base}; "
              f"frames {base}..{ch[0]['frame']-1} have no label")

    segs = []
    for c, nxt in zip(ch, ch[1:] + [None]):
        end = (nxt["frame"] - 1) if nxt else base + n - 1
        if end < c["frame"]:
            continue
        segs.append({"start": c["frame"], "end": end, "holder": c["holder"],
                     "conf": c.get("conf", "sure"),
                     "referee": bool(c.get("referee"))})

    # validate contiguity
    bad = 0
    for x, y in zip(segs, segs[1:]):
        if y["start"] != x["end"] + 1:
            bad += 1
    cov = segs[-1]["end"] - segs[0]["start"] + 1
    held = sum(s["end"] - s["start"] + 1 for s in segs
               if s["holder"] and s["conf"] == "sure")
    nul = sum(s["end"] - s["start"] + 1 for s in segs
              if not s["holder"] and s["conf"] == "sure")
    uns = sum(s["end"] - s["start"] + 1 for s in segs if s["conf"] != "sure")
    out = {
        "game": d["game"],
        "window": {"start_s": round(base / FPS), "dur_s": round(n / FPS),
                   "fps": FPS, "frame_base": base},
        "annotator": a.annotator,
        "frames_are": "global",
        "converted_from": Path(a.state).name,
        "segments": segs,
        "extra_players": d.get("extra_players", []),
        "blind_check": {"changes": len(d.get("blind", {}).get("changes", [])),
                        "spans": len(d.get("blind", {}).get("spans", []))},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    ids = Counter(s["holder"] for s in segs if s["holder"])
    print(f"{a.state} -> {a.out}")
    print(f"  segments {len(segs)} | discontinuities {bad} | covered {cov}/{n} frames")
    print(f"  held {held} ({held/max(cov,1):.0%}) | nobody {nul} ({nul/max(cov,1):.0%}) "
          f"| unsure {uns} ({uns/max(cov,1):.0%})")
    print(f"  change points {len(segs)} ({len(segs)/(n/FPS)*60:.0f}/min)")
    print(f"  ids: {sorted(ids)}")
    if out["blind_check"]["changes"] == 0:
        print("  NOTE: no blind spot-check recorded for this window")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
