#!/usr/bin/env python3
"""Ball-only low-confidence detection cache. The main dets cache uses conf 0.25
(tuned for players); the ball is ~15-25px, motion-blurred, and needs its own
operating point. Writes runs/ball_cache/{game}_{ang}_{tag}.ball.npz
(boxes/scores/frame_idx, class-2 only, all dets >= --conf).

  python scripts/build_ball_cache.py --game e6fba750 --tag 44_60 --conf 0.08
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")
WEIGHTS = "runs/yolo26s-1280-ourdata-v1_fetch/runs/detect/runs/yolo26s-1280-ourdata-v1/weights/best.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--conf", type=float, default=0.08)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--ball-class", type=int, default=2, help="2=yolo26s, 0=Basketball specialist")
    a = ap.parse_args()
    from ultralytics import YOLO
    import cv2

    model = YOLO(str(REPO / a.weights) if not Path(a.weights).is_absolute() else a.weights)
    outd = REPO / "runs/ball_cache"
    outd.mkdir(parents=True, exist_ok=True)
    for ang in ANGLES:
        cap = cv2.VideoCapture(str(REPO / f"data/clips/{a.game}_{ang}_{a.tag}.mp4"))
        boxes, scores, fidx = [], [], []
        f, t0 = 0, time.time()
        while True:
            ok, img = cap.read()
            if not ok:
                break
            r = model.predict(img, imgsz=1280, conf=a.conf, device=a.device,
                              classes=[a.ball_class], verbose=False)[0]
            for b, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                boxes.append(b.astype(np.float32))
                scores.append(float(s))
                fidx.append(f)
            f += 1
        cap.release()
        out = outd / f"{a.game}_{ang}_{a.tag}.ball.npz"
        np.savez_compressed(out, boxes=np.stack(boxes) if boxes else np.zeros((0, 4), np.float32),
                            scores=np.array(scores), frame_idx=np.array(fidx))
        cov = len(set(fidx))
        print(f"{ang}: {len(boxes)} ball dets, {cov}/{f} frames covered "
              f"({100*cov/max(1,f):.0f}%), {f/(time.time()-t0):.0f} fps", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
