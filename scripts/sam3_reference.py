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


def _mask_to_box(mask) -> list | None:
    import numpy as np
    m = np.asarray(mask)
    if m.ndim > 2:
        m = m.squeeze()
    ys, xs = np.where(m > 0.5)
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def _rows_from(o) -> list:
    """Tolerant extraction: boxes if present, else masks -> boxes."""
    import torch
    ids = o.get("object_ids", o.get("obj_ids", []))
    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    scores = o.get("scores", [1.0] * len(ids))
    scores = scores.tolist() if hasattr(scores, "tolist") else list(scores)
    boxes = o.get("boxes")
    rows = []
    if boxes is not None and len(boxes):
        boxes = boxes.tolist() if hasattr(boxes, "tolist") else boxes
        for k, box in enumerate(boxes):
            rows.append({"box": [round(float(v), 1) for v in box],
                         "id": int(ids[k]) if k < len(ids) else None,
                         "score": round(float(scores[k]), 3) if k < len(scores) else 1.0})
        return rows
    masks = o.get("masks", o.get("pred_masks"))
    if masks is None:
        return rows
    if isinstance(masks, torch.Tensor):
        masks = masks.float().cpu().numpy()
    for k in range(len(masks)):
        box = _mask_to_box(masks[k])
        if box:
            rows.append({"box": [round(v, 1) for v in box],
                         "id": int(ids[k]) if k < len(ids) else None,
                         "score": round(float(scores[k]), 3) if k < len(scores) else 1.0})
    return rows


def try_video_track(frames, prompt: str, device: str):
    """SAM3 video: TEXT-prompted concept tracking (Sam3Video*) -> boxes + stable object ids."""
    import torch
    import transformers
    names = [n for n in dir(transformers) if "sam3" in n.lower()]
    print(f"[probe] transformers {transformers.__version__} SAM3 symbols: {names}", flush=True)
    from transformers import Sam3VideoModel, Sam3VideoProcessor

    model = Sam3VideoModel.from_pretrained("facebook/sam3", torch_dtype=torch.bfloat16).to(device)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    session = processor.init_video_session(video=frames, inference_device=device,
                                           video_storage_device="cpu",
                                           dtype=torch.bfloat16)
    if hasattr(processor, "add_text_prompt"):
        processor.add_text_prompt(session, text=prompt)
    else:
        session.add_text_prompt(prompt)
    propagate = getattr(model, "propagate_in_video_iterator",
                        getattr(model, "propagate_in_video", None))
    out: dict[int, list] = {}
    for res in propagate(session):
        o = processor.postprocess_outputs(session, res) if hasattr(processor, "postprocess_outputs") else res
        f = int(o.get("frame_idx", o.get("frame_index", len(out))))
        out[f] = _rows_from(o)
        if f % 50 == 0:
            print(f"[video-track] frame {f}/{len(frames)} -> {len(out[f])}", flush=True)
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
