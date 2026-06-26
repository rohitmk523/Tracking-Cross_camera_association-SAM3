#!/usr/bin/env python3
"""Read jersey numbers per player track (ResNet) + render a jersey-labeled video.

  python scripts/read_jerseys.py --video data/clips/e6fba750_FL_47_12.mp4 \
      --tracks runs/tracking/e6fba750_FL_47_12_teams.json \
      --out runs/tracking/e6fba750_FL_47_12_jersey.json \
      --save-video runs/tracking/e6fba750_FL_47_12_jersey.mp4

Reuses the teams tracklets (keeps team labels). The e6 reader only knows e6's 8
numbers (see jersey.py). Boxes show "<team> #<jersey>" when a number is committed,
else "<team> t<track_id>".
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--weights", default=None, help="override jersey weights path")
    ap.add_argument("--no-localizer", action="store_true", help="skip number-localizer (torso fallback)")
    ap.add_argument("--localizer-weights", default=None)
    ap.add_argument("--localizer-res", type=int, default=384)
    ap.add_argument("--localizer-threshold", type=float, default=0.3)
    ap.add_argument("--sample-per-track", type=int, default=10)
    ap.add_argument("--conf-threshold", type=float, default=0.6)
    ap.add_argument("--min-legible", type=int, default=2)
    ap.add_argument("--save-video", default=None)
    ap.add_argument("--out-fps", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # 0.0 disables the MPS cap (rfdetr localizer hits an invalid LOW<HIGH constraint otherwise).
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"
    from uball_cc.tracking import Track
    from uball_cc.tracking.jersey import JerseyReader, NumberLocalizer, read_jerseys

    data = json.loads(Path(a.tracks).read_text())
    tracks = [Track.from_record(r) for r in data["tracks"]]
    reader = JerseyReader(a.weights) if a.weights else JerseyReader()
    localizer = None if a.no_localizer else NumberLocalizer(
        *( (a.localizer_weights,) if a.localizer_weights else () ),
        resolution=a.localizer_res, threshold=a.localizer_threshold)
    tracks, info = read_jerseys(a.video, tracks, sample_per_track=a.sample_per_track,
                                reader=reader, localizer=localizer,
                                conf_threshold=a.conf_threshold, min_legible=a.min_legible)
    print("jersey reading:", info)
    # per-track committed numbers (player tracks only)
    committed = sorted({t.track_id: t.jersey for t in tracks if t.jersey is not None}.items())
    print("committed numbers (track_id -> #):", committed)

    out = Path(a.out) if a.out else Path(a.tracks).with_name(Path(a.tracks).stem + "_jersey.json")
    out.write_text(json.dumps({**data, "jersey_info": info,
                               "tracks": [t.to_record() for t in tracks]}, indent=2))
    print(f"tracklets+jersey -> {out}")

    if a.save_video:
        import cv2

        from uball_cc.tracking import render_frame
        by_frame: dict[int, list] = {}
        for t in tracks:
            by_frame.setdefault(t.frame, []).append(t)
        cap = cv2.VideoCapture(a.video)
        writer, i = None, 0
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
        print(f"jersey-labeled video -> {vid_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
