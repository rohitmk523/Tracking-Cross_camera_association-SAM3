#!/usr/bin/env python3
"""Derive the event-stream JSON ("what happened") from a fused world-state (docs/08).

  python scripts/derive_events.py --worldstate runs/tracking/e6_worldstate.json \
      --out runs/tracking/e6_events.json

Optional --ball is a JSON {frame: [court_x, court_y]} ball trace; without it, possession
events are skipped (team-spatial events still emitted). See src/uball_cc/fusion/events.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worldstate", required=True)
    ap.add_argument("--ball", default=None, help="optional {frame: [x,y]} ball court trace JSON")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import sys
    sys.path.insert(0, "src")
    from uball_cc.fusion.events import derive_events

    ws = json.loads(Path(a.worldstate).read_text())
    ball = None
    if a.ball:
        ball = {int(k): tuple(v) for k, v in json.loads(Path(a.ball).read_text()).items()}
    out = derive_events(ws, ball_by_frame=ball)

    dest = Path(a.out) if a.out else Path(a.worldstate).with_name("events.json")
    dest.write_text(json.dumps(out, indent=2))
    s = out["summary"]
    print(f"events: {s['n_events']}  passes {s['n_passes']}  turnovers {s['n_turnovers']}  "
          f"transitions {s['n_transitions']}  (ball: {s['has_ball']})")
    for e in out["events"][:14]:
        extra = e.get("player") or f"{e.get('from','')}→{e.get('to','')}" or e.get("team", "")
        print(f"  t={e['t_sec']:>5}s  {e['event']:<14} {extra}")
    if out["caveats"]:
        print("caveats:", out["caveats"])
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
