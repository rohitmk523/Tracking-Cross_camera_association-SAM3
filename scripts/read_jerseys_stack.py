#!/usr/bin/env python3
"""Fill jersey numbers into a saved per-camera tracks JSON using the TRAINED stack
(legibility -> localizer -> PARSeq -> per-track voting). Near cameras only by design —
numbers are unreadable on far fisheye crops (docs/05).

  python scripts/read_jerseys_stack.py --video data/clips/e6fba750_NR_47_12.mp4 \
      --tracks runs/tracking/e6fba750_NR_47_12_teams.json          # updates in place
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--sample-per-track", type=int, default=12)
    ap.add_argument("--min-box-h", type=int, default=110)
    ap.add_argument("--out", default=None, help="default: update --tracks in place")
    a = ap.parse_args()

    from uball_cc.tracking import Track
    from uball_cc.tracking.jersey_stack import JerseyStack, read_track_jerseys

    data = json.loads(Path(a.tracks).read_text())
    tracks = [Track.from_record(r) for r in data["tracks"]]
    numbers = read_track_jerseys(a.video, tracks, stack=JerseyStack(),
                                 sample_per_track=a.sample_per_track,
                                 min_box_h=a.min_box_h)
    out_tracks = [t.with_attrs(jersey=numbers[t.track_id]) if t.track_id in numbers else t
                  for t in tracks]
    data["tracks"] = [t.to_record() for t in out_tracks]
    data["jersey_info"] = {"committed": {str(k): v for k, v in sorted(numbers.items())},
                           "n_tracks_named": len(numbers)}
    out = Path(a.out or a.tracks)
    out.write_text(json.dumps(data, indent=2))
    print(f"{Path(a.video).stem}: committed numbers for {len(numbers)} tracks: "
          f"{dict(sorted(numbers.items()))} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
