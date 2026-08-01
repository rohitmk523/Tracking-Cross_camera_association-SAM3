#!/usr/bin/env python3
"""SPEED BUNDLE: one batched pass of the UNIFIED 4-class detector writes BOTH
caches — dets (player/referee) and ball (Basketball/Hoop) — replacing the two
separate full-video passes of build_dets_cache_yolo + build_ball_cache.

Unified model classes: 0 player, 1 referee, 2 ball, 3 hoop.
Outputs keep the LEGACY schemas/filenames so every consumer runs unchanged:
  runs/dets_cache/{game}_{ang}_{tag}_small_1280_t0.25.dets.npz   (classes 0,1)
  runs/ball_cache/{game}_{ang}_{tag}.ball.npz                    (classes 0=ball,1=hoop)

  .venv/bin/python scripts/build_caches_unified.py --game c2afast --tag 600_180 \
      --weights runs/unified_yolo26s_fetch/.../best.pt --device mps --batch 16
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")


def flush_batch(model, frames, idxs, a, out):
    if not frames:
        return
    res = model.predict(frames, imgsz=a.imgsz, conf=a.conf_floor,
                        device=a.device, verbose=False)
    for fi, r in zip(idxs, res):
        b = r.boxes
        if b is None or len(b) == 0:
            continue
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        for j in range(len(cls)):
            c, s = int(cls[j]), float(conf[j])
            if c in (0, 1) and s >= a.conf_player:
                out["dets"].append((xyxy[j], s, c, fi))
            elif c == 2 and s >= a.conf_ball:
                out["ball"].append((xyxy[j], s, 0, fi))
            elif c == 3 and s >= a.conf_ball:
                out["ball"].append((xyxy[j], s, 1, fi))


def save(rows, path):
    if rows:
        boxes = np.array([r[0] for r in rows], dtype=np.float32)
        scores = np.array([r[1] for r in rows], dtype=np.float32)
        classes = np.array([r[2] for r in rows], dtype=np.int16)
        fidx = np.array([r[3] for r in rows], dtype=np.int32)
    else:
        boxes = np.zeros((0, 4), np.float32)
        scores = np.zeros(0, np.float32)
        classes = np.zeros(0, np.int16)
        fidx = np.zeros(0, np.int32)
    np.savez_compressed(path, boxes=boxes, scores=scores,
                        classes=classes, frame_idx=fidx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf-player", type=float, default=0.25)
    ap.add_argument("--conf-ball", type=float, default=0.15)
    ap.add_argument("--conf-floor", type=float, default=0.15,
                    help="single inference floor; per-class filters applied at write")
    ap.add_argument("--angles", default=",".join(ANGLES))
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.weights)
    (REPO / "runs/dets_cache").mkdir(parents=True, exist_ok=True)
    (REPO / "runs/ball_cache").mkdir(parents=True, exist_ok=True)

    for ang in a.angles.split(","):
        clip = REPO / f"data/clips/{a.game}_{ang}_{a.tag}.mp4"
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            print(f"{ang}: MISSING {clip}")
            continue
        t0 = time.time()
        out = {"dets": [], "ball": []}
        frames, idxs, fi = [], [], 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
            idxs.append(fi)
            fi += 1
            if len(frames) >= a.batch:
                flush_batch(model, frames, idxs, a, out)
                frames, idxs = [], []
        flush_batch(model, frames, idxs, a, out)
        cap.release()
        save(out["dets"],
             REPO / f"runs/dets_cache/{a.game}_{ang}_{a.tag}_small_{a.imgsz}_t0.25.dets.npz")
        save(out["ball"], REPO / f"runs/ball_cache/{a.game}_{ang}_{a.tag}.ball.npz")
        dt = time.time() - t0
        print(f"{ang}: {fi} frames in {dt:.0f}s ({fi/max(dt,1e-9):.1f} fps) | "
              f"{len(out['dets'])} person dets, {len(out['ball'])} ball/hoop", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
