#!/usr/bin/env python3
"""SAM3 SINGLE-OBJECT tracking (Roboflow recipe) — runs on the AWS GPU.

Seed ONE box on ONE frame; SAM3 propagates a mask for THAT player across the clip.
This is SAM3VideoPredictor with a BOX PROMPT — not the open-vocabulary
SAM3VideoSemanticPredictor (which detects everyone and fragments). One clean masklet
per player per camera; fusion + jersey re-acquisition happen in our own logic afterward.

Seeds at an arbitrary frame by trimming the decode to [seed_frame:], seeding frame 0
of the trimmed stream with the box, and mapping results back to absolute frames.

Output per (player,camera):
  {"player","cam","seed_frame","seed_box","imgsz",
   "frames": {abs_frame: {"box":[x1,y1,x2,y2]|null, "score":s, "mask_area":a, "present":bool}}}

  python scripts/sam3_track_player.py --video clip.mp4 --seed-frame 27 \
      --seed-box 900,300,1000,600 --player "#11" --cam FR --out out.json --weights sam3.pt
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--seed-frame", type=int, required=True)
    ap.add_argument("--seed-box", required=True, help="x1,y1,x2,y2 in camera pixels")
    ap.add_argument("--player", required=True)
    ap.add_argument("--cam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="sam3.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    a = ap.parse_args()

    import cv2
    import numpy as np
    import torch
    from ultralytics.models.sam import SAM3VideoPredictor

    box = [float(v) for v in a.seed_box.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- trim decode to [seed_frame:] into a temp clip so the box seeds frame 0 ---
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.seed_frame)
    tmp = Path(tempfile.mkdtemp()) / "seg.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        vw.write(im)
        n += 1
    cap.release()
    vw.release()
    print(f"[{a.player} {a.cam}] trimmed {n} frames from seed {a.seed_frame}; seeding box {box}",
          flush=True)

    overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=a.imgsz,
                     model=a.weights, half=(device == "cuda"), save=False,
                     verbose=False, device=device)
    predictor = SAM3VideoPredictor(overrides=overrides)

    frames_out = {}
    results = predictor(source=str(tmp), bboxes=[box], stream=True)
    for i, r in enumerate(results):
        abs_f = a.seed_frame + i
        rec = {"box": None, "score": 0.0, "mask_area": 0, "present": False}
        masks = getattr(r, "masks", None)
        boxes = getattr(r, "boxes", None)
        if masks is not None and getattr(masks, "data", None) is not None and len(masks.data):
            m = masks.data[0].cpu().numpy() > 0.5
            area = int(m.sum())
            if area > 40:
                ys, xs = np.where(m)
                rec = {"box": [float(xs.min()), float(ys.min()),
                               float(xs.max()), float(ys.max())],
                       "score": (round(float(boxes.conf[0]), 3)
                                 if boxes is not None and getattr(boxes, "conf", None) is not None
                                 and len(boxes.conf) else 1.0),
                       "mask_area": area, "present": True}
        elif boxes is not None and len(boxes):
            xyxy = boxes.xyxy[0].cpu().numpy()
            rec = {"box": [round(float(v), 1) for v in xyxy],
                   "score": (round(float(boxes.conf[0]), 3)
                             if getattr(boxes, "conf", None) is not None and len(boxes.conf) else 1.0),
                   "mask_area": 0, "present": True}
        frames_out[str(abs_f)] = rec
        if i % 100 == 0:
            print(f"[{a.player} {a.cam}] frame {abs_f}: present={rec['present']} "
                  f"area={rec['mask_area']}", flush=True)
    del predictor

    n_present = sum(1 for v in frames_out.values() if v["present"])
    Path(a.out).write_text(json.dumps(
        {"player": a.player, "cam": a.cam, "seed_frame": a.seed_frame, "seed_box": box,
         "imgsz": a.imgsz, "n_frames": len(frames_out), "n_present": n_present,
         "frames": frames_out}))
    print(f"[{a.player} {a.cam}] done: present in {n_present}/{len(frames_out)} frames -> {a.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
