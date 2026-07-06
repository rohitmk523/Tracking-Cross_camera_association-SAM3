#!/usr/bin/env python3
"""Extract a fused BALL court trace for the event stream (docs/08 possession).

Re-runs the detector keeping only the ball class (2), takes the top-scoring ball per
frame per camera, projects its center through each camera's homography (gated to the
calibrated region), and fuses across cameras (per-frame median) -> {frame: [court_x,
court_y]}. NOTE: the planar homography assumes the ball is near the floor; a ball in
flight (shot/long pass) projects with error -- fine for possession proximity, flagged
for shot detection.

  python scripts/extract_ball.py --game e6fba750 --out runs/tracking/e6_ball.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="e6fba750")
    ap.add_argument("--clip-dir", default="data/clips")
    ap.add_argument("--suffix", default="_47_12", help="clip name suffix after the angle")
    ap.add_argument("--calib-dir", default="configs/calib")
    ap.add_argument("--weights", default="runs/rfdetr-s-1280-ourdata-v1/best.pth")
    ap.add_argument("--cams", default="NL,NR",
                    help="near cams by default: the retrained ball AP (0.80) is near-basket")
    ap.add_argument("--ref", default="FL", help="reference angle for the fused timeline")
    ap.add_argument("--threshold", type=float, default=0.15, help="lower for the small ball")
    ap.add_argument("--region-pad", type=float, default=400.0, help="ball can leave the calib hull more")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")
    import sys
    sys.path.insert(0, "src")
    from uball_cc.detection.base import RFDETRDetector
    from uball_cc.fusion.audiosync import audio_offset_seconds
    from uball_cc.fusion.homography import calib_hull, in_calib_region, load_calib, project_pixels
    from uball_cc.tracking import iter_video_frames

    angles = tuple(a.cams.split(","))
    ref_clip = Path(a.clip_dir) / f"{a.game}_{a.ref}{a.suffix}.mp4"
    detector = RFDETRDetector(a.weights, resolution=1280, threshold=a.threshold, model="small")
    per_cam: dict[str, dict[int, tuple]] = {}     # angle -> {ref_frame: (court_x, court_y)}
    for ang in angles:
        clip = Path(a.clip_dir) / f"{a.game}_{ang}{a.suffix}.mp4"
        if not clip.exists():
            print(f"  {ang}: clip missing {clip}")
            continue
        off = 0
        if ang != a.ref and ref_clip.exists():    # put every camera on the REF timeline
            off_s, _ = audio_offset_seconds(str(ref_clip), str(clip))
            off = int(round(off_s * 29.97))
        calib = load_calib(Path(a.calib_dir) / f"{ang}.json")
        hull = calib_hull(calib)
        got = {}
        n_ball = 0
        for fi, img in enumerate(iter_video_frames(str(clip))):
            balls = [d for d in detector.predict(img) if d.class_id == 2]
            if not balls:
                continue
            d = max(balls, key=lambda b: b.score)
            cx = (d.box_xyxy[0] + d.box_xyxy[2]) / 2.0
            cy = (d.box_xyxy[1] + d.box_xyxy[3]) / 2.0
            court = project_pixels([(cx, cy)], calib)[0]   # undistort + homography
            if in_calib_region(court, hull, a.region_pad):
                got[fi - off] = (float(court[0]), float(court[1]))
                n_ball += 1
        per_cam[ang] = got
        print(f"  {ang}: ball detected+projected in {n_ball} frames (sync {off:+d}f)", flush=True)

    # fuse across cameras: per-frame median of projected ball positions
    all_frames = sorted({f for cam in per_cam.values() for f in cam})
    fused: dict[int, list] = {}
    for f in all_frames:
        pts = [cam[f] for cam in per_cam.values() if f in cam]
        if pts:
            arr = np.array(pts)
            fused[f] = [round(float(np.median(arr[:, 0])), 1), round(float(np.median(arr[:, 1])), 1)]
    print(f"fused ball trace: {len(fused)} frames "
          f"(per-cam union {len(all_frames)}; multi-cam agreement where available)")

    dest = Path(a.out) if a.out else Path("runs/tracking") / f"{a.game}_ball.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(fused))
    print(f"-> {dest}  ({len(fused)} frames with ball)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
