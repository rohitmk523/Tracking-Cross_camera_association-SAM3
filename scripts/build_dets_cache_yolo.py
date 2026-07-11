#!/usr/bin/env python3
"""Build detection caches with a YOLO model, in the exact npz format the pipeline
consumes (boxes/scores/classes/frame_idx; classes 0 player, 1 referee, 2 ball —
same order as data.yaml). Used for the Level-2 pipeline gate of YOLO_TRAINING_PLAN.

  python scripts/build_dets_cache_yolo.py --game e6fba750 --tag 44_60 \
      --weights runs/yolo11s-1280-ourdata-v1_fetch/.../best.pt \
      --out-dir runs/dets_cache_yolo11s
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    from ultralytics import YOLO
    import cv2

    model = YOLO(a.weights)
    outd = REPO / a.out_dir
    outd.mkdir(parents=True, exist_ok=True)
    for ang in ANGLES:
        clip = REPO / f"data/clips/{a.game}_{ang}_{a.tag}.mp4"
        cap = cv2.VideoCapture(str(clip))
        boxes, scores, classes, fidx = [], [], [], []
        f = 0
        t0 = time.time()
        while True:
            ok, img = cap.read()
            if not ok:
                break
            r = model.predict(img, imgsz=a.imgsz, conf=a.conf, device=a.device,
                              verbose=False)[0]
            for b, s, c in zip(r.boxes.xyxy.cpu().numpy(),
                               r.boxes.conf.cpu().numpy(),
                               r.boxes.cls.cpu().numpy()):
                boxes.append(b.astype(np.float32))
                scores.append(float(s))
                classes.append(int(c))
                fidx.append(f)
            f += 1
            if f % 300 == 0:
                print(f"{ang} {f} frames, {f/(time.time()-t0):.1f} fps", flush=True)
        cap.release()
        # canonical filename so every consumer works unchanged
        out = outd / f"{a.game}_{ang}_{a.tag}_small_1280_t0.25.dets.npz"
        np.savez_compressed(out, boxes=np.stack(boxes) if boxes else np.zeros((0, 4)),
                            scores=np.array(scores), classes=np.array(classes),
                            frame_idx=np.array(fidx))
        print(f"{ang}: {f} frames, {len(boxes)} detections -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
