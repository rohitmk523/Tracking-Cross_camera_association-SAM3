#!/usr/bin/env python3
"""SAM3 REFERENCE pass (runs on the AWS GPU): independent detections/tracks to cross-check
our RF-DETR+ByteTrack pipeline. NOT ground truth — a second opinion from a different model
family: agreement builds confidence, disagreements are specific frames to inspect.

Uses the LOCAL sam3.pt (Ultralytics checkpoint) staged to S3 — no Hugging Face gating.
The inference adapter mirrors DEMO_UBALL's proven `sam3_track_camera` (defensive getattr
on the Ultralytics SAM3 results; text concept prompts; boxes in ORIGINAL pixel coords).

Output per clip:
  {"mode": "ultralytics-video", "prompts": [...],
   "frames": {frame: [{"box":[x1,y1,x2,y2], "id": i|null, "score": s, "cls": c}]}}

  python scripts/sam3_reference.py --video clip.mp4 --out clip.sam3.json --weights sam3.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROMPTS = ["basketball player", "basketball referee"]   # colour-neutral (DEMO finding:
                                                        # per-colour prompts under-detect)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="sam3.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.3)
    a = ap.parse_args()

    import numpy as np
    import torch
    from ultralytics.models.sam import SAM3VideoSemanticPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    overrides = dict(conf=a.conf, task="segment", mode="predict", imgsz=a.imgsz,
                     model=a.weights, half=(device == "cuda"), save=False,
                     verbose=False, device=device)
    predictor = SAM3VideoSemanticPredictor(overrides=overrides)
    print(f"[run] {a.video} on {device} (imgsz={a.imgsz}, conf={a.conf})", flush=True)

    out: dict[int, list] = {}
    results = predictor(source=a.video, text=PROMPTS, stream=True)
    for fidx, r in enumerate(results):
        boxes = getattr(r, "boxes", None)
        rows = []
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            ids = (boxes.id.int().cpu().numpy() if boxes.id is not None
                   else np.full(len(xyxy), -1))
            clss = (boxes.cls.int().cpu().numpy() if boxes.cls is not None
                    else np.zeros(len(xyxy), int))
            confs = (boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None
                     else np.ones(len(xyxy)))
            for k in range(len(xyxy)):
                rows.append({"box": [round(float(v), 1) for v in xyxy[k]],
                             "id": int(ids[k]) if ids[k] >= 0 else None,
                             "score": round(float(confs[k]), 3),
                             "cls": int(clss[k])})
        out[fidx] = rows
        if fidx == 0:
            print(f"[frame0] {len(rows)} dets, orig={getattr(r, 'orig_shape', '?')}", flush=True)
        elif fidx % 50 == 0:
            print(f"[frame {fidx}] {len(rows)} dets", flush=True)
    del predictor

    Path(a.out).write_text(json.dumps(
        {"mode": "ultralytics-video", "prompts": PROMPTS, "video": a.video,
         "n_frames": len(out), "frames": {str(k): v for k, v in sorted(out.items())}}))
    n = sum(len(v) for v in out.values())
    print(f"[done] {n} boxes over {len(out)} frames -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
