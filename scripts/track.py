#!/usr/bin/env python3
"""Per-camera tracking: detect -> ByteTrack -> tracklets (docs/05, Week-1 D5).

  # dependency-free smoke (dummy detector over any image dir)
  python scripts/track.py --detector dummy --frames data/detect_consolidated/test/images \
      --cam FL --max-frames 100

  # real RF-DETR over a camera video
  python scripts/track.py --detector rfdetr --weights runs/rfdetr-s-1280-ourdata-v1/best.pth \
      --model small --video data/games/<gid>/FL.mp4 --cam FL --out runs/tracking/<gid>_FL.json

Writes per-camera tracklets JSON (Tracking->Fusion contract) + prints stability stats.
Attributes (team/jersey/reid) are filled by the attribute stage (next sub-step).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_detector(a):
    if a.detector == "dummy":
        from uball_cc.detection.base import DummyDetector
        return DummyDetector()
    if a.detector == "rfdetr":
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
        os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
        from uball_cc.detection.base import RFDETRDetector
        if not a.weights:
            raise SystemExit("--weights required for --detector rfdetr")
        return RFDETRDetector(a.weights, resolution=a.resolution,
                              threshold=a.threshold, model=a.model)
    raise SystemExit(f"unknown detector: {a.detector}")


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="camera mp4")
    src.add_argument("--frames", help="directory of frames (sorted)")
    ap.add_argument("--cam", required=True, choices=("FL", "FR", "NL", "NR"))
    ap.add_argument("--detector", default="rfdetr", choices=("dummy", "rfdetr"))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--model", default="small", choices=("nano", "small"))
    ap.add_argument("--resolution", type=int, default=1280)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="video frame stride")
    ap.add_argument("--min-consecutive-frames", type=int, default=3)
    ap.add_argument("--save-video", default=None, help="write an annotated mp4 (boxes+ids) here")
    ap.add_argument("--out-fps", type=float, default=30.0, help="fps for --save-video")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from uball_cc.tracking import (
        iter_image_frames, iter_video_frames, render_frame, summarize,
        track_sequence, track_stream,
    )

    detector = _load_detector(a)
    if a.video:
        frames = iter_video_frames(a.video, max_frames=a.max_frames, stride=a.stride)
        source = a.video
    else:
        frames = iter_image_frames(a.frames, max_frames=a.max_frames)
        source = a.frames

    if a.save_video:
        import cv2
        vid_out = Path(a.save_video)
        vid_out.parent.mkdir(parents=True, exist_ok=True)
        writer = None
        tracks = []
        for _, img, frame_tracks in track_stream(
                detector, frames, a.cam,
                minimum_consecutive_frames=a.min_consecutive_frames):
            tracks.extend(frame_tracks)
            vis = render_frame(img, frame_tracks)
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(str(vid_out),
                                         cv2.VideoWriter_fourcc(*"mp4v"), a.out_fps, (w, h))
            writer.write(vis)
        if writer:
            writer.release()
        print(f"annotated video -> {vid_out}")
    else:
        tracks = track_sequence(detector, frames, a.cam,
                                minimum_consecutive_frames=a.min_consecutive_frames)
    stats = summarize(tracks)
    print(f"cam={a.cam} detector={a.detector} source={source}")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    out = Path(a.out) if a.out else (REPO / "runs" / "tracking" / f"{a.cam}_{a.detector}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "cam": a.cam, "detector": a.detector, "source": source,
        "stats": stats, "tracks": [t.to_record() for t in tracks],
    }, indent=2))
    print(f"tracklets -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
