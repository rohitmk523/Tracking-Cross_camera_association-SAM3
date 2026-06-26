#!/usr/bin/env python3
"""Assign A/B teams to player tracklets (SigLIP + KMeans) and render a team-colored video.

  python scripts/assign_teams.py --video data/clips/e6fba750_FL_47_12.mp4 \
      --tracks runs/tracking/e6fba750_FL_47_12.json \
      --out runs/tracking/e6fba750_FL_47_12_teams.json \
      --save-video runs/tracking/e6fba750_FL_47_12_teams.mp4

Players -> A/B (clustered by appearance), referees -> REF. Reuses the tracklets from
scripts/track.py; the SigLIP model downloads once on first run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True, help="tracklets JSON from scripts/track.py")
    ap.add_argument("--sample-per-track", type=int, default=6)
    ap.add_argument("--model", default="google/siglip-base-patch16-224")
    ap.add_argument("--save-video", default=None)
    ap.add_argument("--out-fps", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
    from uball_cc.tracking import Track
    from uball_cc.tracking.teams import SiglipEmbedder, assign_teams

    data = json.loads(Path(a.tracks).read_text())
    tracks = [Track.from_record(r) for r in data["tracks"]]

    embedder = SiglipEmbedder(a.model)
    tracks, info = assign_teams(a.video, tracks, sample_per_track=a.sample_per_track,
                                embedder=embedder)
    print("team assignment:", info)

    out = Path(a.out) if a.out else Path(a.tracks).with_name(Path(a.tracks).stem + "_teams.json")
    out.write_text(json.dumps({**data, "team_info": info,
                               "tracks": [t.to_record() for t in tracks]}, indent=2))
    print(f"tracklets+teams -> {out}")

    if a.save_video:
        import cv2

        from uball_cc.tracking import render_frame
        by_frame: dict[int, list[Track]] = {}
        for t in tracks:
            by_frame.setdefault(t.frame, []).append(t)
        cap = cv2.VideoCapture(a.video)
        writer = None
        i = 0
        vid_out = Path(a.save_video)
        vid_out.parent.mkdir(parents=True, exist_ok=True)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vis = render_frame(frame, by_frame.get(i, []))
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(str(vid_out), cv2.VideoWriter_fourcc(*"mp4v"),
                                         a.out_fps, (w, h))
            writer.write(vis)
            i += 1
        cap.release()
        if writer:
            writer.release()
        print(f"team-colored video -> {vid_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
