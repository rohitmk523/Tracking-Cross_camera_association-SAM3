#!/usr/bin/env python3
"""THE RACE (Level-3): detector speed on identical frames, one GPU, blind-game video.

Lanes: RF-DETR-Small FP16 (incumbent, optimize_for_inference) vs YOLO11 (half).
Same 1-minute x 4-camera blind-game clips, same 1280 input, same conf, sequential
on the same device. Warmup excluded. Reports fps, minutes per full game
(40min x 4 cams = 288k frames), $/game at $1.212/h.

  python scripts/race_detectors.py --game f66eb3b2 --tag race_60 \
      --rfdetr weights/best.pth --yolo weights/yolo11s.pt [--yolo2 weights/yolo11m.pt]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")
GAME_FRAMES = 40 * 60 * 30 * 4          # full game, 4 cams
GPU_USD_H = 1.212


def frames(game, tag):
    import cv2
    for ang in ANGLES:
        cap = cv2.VideoCapture(str(REPO / f"data/clips/{game}_{ang}_{tag}.mp4"))
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield img
        cap.release()


def lane_rfdetr(weights, game, tag, warmup=20):
    from uball_cc.detection.base import RFDETRDetector      # FP16 auto on CUDA
    det = RFDETRDetector(weights, resolution=1280, threshold=0.25, model="small")
    n = t0 = 0
    for i, img in enumerate(frames(game, tag)):
        if i == warmup:
            t0 = time.time()
        det.predict(img)
        n = i
    dt = time.time() - t0
    return (n + 1 - warmup) / dt


def lane_yolo(weights, game, tag, warmup=20):
    from ultralytics import YOLO
    model = YOLO(weights)
    n = t0 = 0
    for i, img in enumerate(frames(game, tag)):
        if i == warmup:
            t0 = time.time()
        model.predict(img, imgsz=1280, conf=0.25, half=True, verbose=False)
        n = i
    dt = time.time() - t0
    return (n + 1 - warmup) / dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rfdetr", required=True)
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--yolo2", default=None)
    ap.add_argument("--yolo3", default=None)
    a = ap.parse_args()

    def yname(path):
        stem = Path(path).stem            # e.g. yolo26s_best
        return stem.replace("_best", "") + "_fp16"

    results = {}
    lanes = [("rfdetr_s_fp16", lambda: lane_rfdetr(a.rfdetr, a.game, a.tag)),
             (yname(a.yolo), lambda: lane_yolo(a.yolo, a.game, a.tag))]
    if a.yolo2:
        lanes.append((yname(a.yolo2), lambda: lane_yolo(a.yolo2, a.game, a.tag)))
    if a.yolo3:
        lanes.append((yname(a.yolo3), lambda: lane_yolo(a.yolo3, a.game, a.tag)))
    for name, fn in lanes:
        print(f"=== lane {name} ===", flush=True)
        fps = fn()
        mins = GAME_FRAMES / fps / 60
        usd = mins / 60 * GPU_USD_H
        results[name] = {"fps": round(fps, 1), "full_game_minutes": round(mins, 1),
                         "usd_per_game_detection": round(usd, 2)}
        print(f"{name}: {fps:.1f} fps | full game {mins:.0f} min | ${usd:.2f}", flush=True)
    print("RACE_RESULTS_JSON " + json.dumps(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
