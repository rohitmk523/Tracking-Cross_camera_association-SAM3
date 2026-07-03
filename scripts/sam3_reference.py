#!/usr/bin/env python3
"""SAM3 REFERENCE pass (runs on the AWS GPU): independent detections/tracks to cross-check
our RF-DETR+ByteTrack pipeline (docs/05 kept SAM3 as an offline seeder/reference, not the
tracker). NOT ground truth — a second opinion from a different model family: agreement
builds confidence, disagreements are specific frames to inspect.

Tries, in order:
  1. transformers SAM3 VIDEO tracking (text prompt "person") -> boxes + track ids
  2. transformers SAM3 per-frame image concept detection      -> boxes only
Logs which path activated. Output per clip:
  {"mode": ..., "prompt": ..., "frames": {frame: [{"box":[x1,y1,x2,y2],"id":i|null,"score":s}]}}

  python scripts/sam3_reference.py --video clip.mp4 --out clip.sam3.json --prompt person
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def read_frames(path: str):
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def try_video_track(frames, prompt: str, device: str):
    """SAM3 video: text-prompted concept tracking -> per-frame boxes + stable object ids."""
    import torch
    import transformers
    names = [n for n in dir(transformers) if "sam3" in n.lower()]
    print(f"[probe] transformers {transformers.__version__} SAM3 symbols: {names}", flush=True)
    from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor  # may not exist

    model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3", torch_dtype=torch.bfloat16).to(device)
    processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
    session = processor.init_video_session(video=frames, inference_device=device)
    processor.add_text_prompt(session, text=prompt)
    out: dict[int, list] = {}
    for res in model.propagate_in_video_iterator(session):
        o = processor.postprocess_outputs(session, res)
        f = int(o["frame_idx"])
        rows = []
        for oid, box, score in zip(o["object_ids"], o["boxes"], o.get("scores", [1.0] * len(o["boxes"]))):
            rows.append({"box": [round(float(v), 1) for v in box], "id": int(oid),
                         "score": round(float(score), 3)})
        out[f] = rows
    return "video-track", out


def try_image_detect(frames, prompt: str, device: str):
    """SAM3 per-frame promptable concept detection (no tracking): boxes for 'person'."""
    import torch
    from transformers import Sam3Model, Sam3Processor

    model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.bfloat16).to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    out: dict[int, list] = {}
    for f, img in enumerate(frames):
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            res = model(**inputs)
        post = processor.post_process_instance_segmentation(
            res, threshold=0.4, target_sizes=[(img.shape[0], img.shape[1])])[0]
        rows = []
        boxes = post.get("boxes")
        scores = post.get("scores", [1.0] * (len(boxes) if boxes is not None else 0))
        if boxes is not None:
            for box, s in zip(boxes, scores):
                rows.append({"box": [round(float(v), 1) for v in box], "id": None,
                             "score": round(float(s), 3)})
        out[f] = rows
        if f % 50 == 0:
            print(f"[image-detect] frame {f}/{len(frames)} -> {len(rows)}", flush=True)
    return "image-detect", out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="person")
    a = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = read_frames(a.video)
    print(f"[run] {a.video}: {len(frames)} frames on {device}", flush=True)
    mode, data = None, None
    for fn in (try_video_track, try_image_detect):
        try:
            mode, data = fn(frames, a.prompt, device)
            break
        except Exception:
            print(f"[fallthrough] {fn.__name__} failed:\n{traceback.format_exc()}", flush=True)
    if data is None:
        sys.exit("all SAM3 paths failed — see probe output above")
    Path(a.out).write_text(json.dumps(
        {"mode": mode, "prompt": a.prompt, "video": a.video, "n_frames": len(frames),
         "frames": {str(k): v for k, v in sorted(data.items())}}))
    n = sum(len(v) for v in data.values())
    print(f"[done] mode={mode} {n} boxes over {len(data)} frames -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
