#!/usr/bin/env python3
"""SAHI-tiled ball detection + temporal tracking -> clean ball court trace (docs/08).

The ball is ~13 px at 1920x1080 -> nearly invisible to a 1280-input detector. Slicing the
frame into tiles and detecting per tile gives the ball ~3x the effective resolution, so it
clears a usable confidence. We keep ALL per-frame candidates (low threshold) and let the
temporal Kalman tracker (fusion/ball.py) gate out false positives + interpolate gaps.

  python scripts/track_ball.py --game e6fba750 --angles NR NL --grid 3x2 \
      --out runs/tracking/e6_ball_tracked.json

Near cameras only by default (they see the ball best + own the court).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _tiles(w, h, cols, rows, overlap):
    tw, th = w / cols, h / rows
    ox, oy = tw * overlap, th * overlap
    out = []
    for r in range(rows):
        for c in range(cols):
            x0, y0 = max(0, int(c * tw - ox)), max(0, int(r * th - oy))
            x1, y1 = min(w, int((c + 1) * tw + ox)), min(h, int((r + 1) * th + oy))
            out.append((x0, y0, x1, y1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="e6fba750")
    ap.add_argument("--clip-dir", default="data/clips")
    ap.add_argument("--suffix", default="_47_12")
    ap.add_argument("--calib-dir", default="configs/calib")
    ap.add_argument("--angles", nargs="+", default=["NR", "NL"])
    ap.add_argument("--weights", default="runs/rfdetr-s-1280-ourdata-v1/best.pth")
    ap.add_argument("--grid", default="3x2", help="tile grid COLSxROWS")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--det-thr", type=float, default=0.05, help="low: tracker gates false positives")
    ap.add_argument("--region-pad", type=float, default=500.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cols, rows = (int(x) for x in a.grid.lower().split("x"))

    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")
    import sys
    sys.path.insert(0, "src")
    from uball_cc.detection.base import RFDETRDetector
    from uball_cc.fusion.ball import track_ball
    from uball_cc.fusion.homography import (calib_hull, homography_from_calib, in_calib_region,
                                            load_calib, project)
    from uball_cc.tracking import iter_video_frames

    detector = RFDETRDetector(a.weights, resolution=1280, threshold=a.det_thr, model="small")
    candidates: dict[int, list] = {}                 # frame -> [(court_x, court_y, score)]
    for ang in a.angles:
        clip = Path(a.clip_dir) / f"{a.game}_{ang}{a.suffix}.mp4"
        if not clip.exists():
            print(f"  {ang}: clip missing {clip}", flush=True)
            continue
        calib = load_calib(Path(a.calib_dir) / f"{ang}.json")
        h, hull = homography_from_calib(calib), calib_hull(calib)
        n_hit = 0
        for fi, img in enumerate(iter_video_frames(str(clip))):
            H, W = img.shape[:2]
            frame_cands = []
            for (x0, y0, x1, y1) in _tiles(W, H, cols, rows, a.overlap):
                tile = img[y0:y1, x0:x1]
                for d in detector.predict(tile):
                    if d.class_id != 2:
                        continue
                    cx = x0 + (d.box_xyxy[0] + d.box_xyxy[2]) / 2.0
                    cy = y0 + (d.box_xyxy[1] + d.box_xyxy[3]) / 2.0
                    court = project([(cx, cy)], h)[0]
                    if in_calib_region(court, hull, a.region_pad):
                        frame_cands.append((float(court[0]), float(court[1]), float(d.score)))
            if frame_cands:
                candidates.setdefault(fi, []).extend(frame_cands)
                n_hit += 1
        print(f"  {ang}: ball candidates in {n_hit} frames (grid {cols}x{rows})", flush=True)

    n_cand_frames = len(candidates)
    from uball_cc.fusion.ball import reject_stationary
    candidates, banned = reject_stationary({f: c for f, c in candidates.items()})
    if banned:
        print(f"stationarity filter: dropped {len(banned)} fixed-FP cell(s) @ {banned[:3]}", flush=True)
    trace = track_ball(candidates)
    print(f"raw candidate frames: {n_cand_frames}  ->  tracked ball frames: {len(trace)}", flush=True)

    dest = Path(a.out) if a.out else Path("runs/tracking") / f"{a.game}_ball_tracked.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(trace))
    # also dump raw candidates for inspection / re-tuning the tracker without re-detecting
    raw = {str(f): c for f, c in candidates.items()}
    Path(str(dest).replace(".json", "_candidates.json")).write_text(json.dumps(raw))
    print(f"-> {dest}  ({len(trace)} tracked frames)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
