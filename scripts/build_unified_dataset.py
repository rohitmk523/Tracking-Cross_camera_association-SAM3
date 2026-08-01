#!/usr/bin/env python3
"""Build the UNIFIED 4-class detection dataset (player/referee/ball/hoop) by
merging detect_consolidated (player/ref/ball) + ball_pooled (Basketball/Hoop).

The catch: each source labels only ITS classes, and the image sets are
disjoint. Concatenating raw would teach the model that players are background
on ball images and hoops are background on player images. Fix = CROSS
PSEUDO-LABELING with the two production detectors:
  - ball_pooled images   -> + player/referee boxes from yolo26s v1 @ conf>=0.60
  - consolidated images  -> + hoop boxes from the ball specialist @ conf>=0.50
Real labels always win: pseudo boxes overlapping a real box of the same class
(IoU>0.5) are dropped.

Unified classes: 0 player, 1 referee, 2 ball, 3 hoop.
Output: data/detect_unified/{train,valid,test}/{images,labels} + data.yaml
Images are HARDLINKED (same volume, ~0 extra bytes).

  .venv/bin/python scripts/build_unified_dataset.py --device mps
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
CONS = REPO / "data/detect_consolidated"
POOL = REPO / "data/ball_pooled"
OUT = REPO / "data/detect_unified"
DET_W = REPO / ("runs/yolo26s-1280-ourdata-v1_fetch/runs/detect/runs/"
                "yolo26s-1280-ourdata-v1/weights/best.pt")
BALL_W = REPO / ("runs/ball_yolo26s_fetch/runs/detect/runs/"
                 "ball-yolo26s-1280-v1/weights/best.pt")


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter)


def yolo_line(cls, box, w, h):
    x1, y1, x2, y2 = box
    return (f"{cls} {((x1 + x2) / 2 / w):.6f} {((y1 + y2) / 2 / h):.6f} "
            f"{((x2 - x1) / w):.6f} {((y2 - y1) / h):.6f}")


def parse_label(path, remap):
    rows = []
    if path.exists():
        for ln in path.read_text().splitlines():
            p = ln.split()
            if len(p) >= 5:
                rows.append((remap[int(p[0])], *[float(v) for v in p[1:5]]))
    return rows


def denorm(row, w, h):
    _, cx, cy, bw, bh = row
    return ((cx - bw / 2) * w, (cy - bh / 2) * h,
            (cx + bw / 2) * w, (cy + bh / 2) * h)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    from ultralytics import YOLO
    det = YOLO(str(DET_W))
    ball = YOLO(str(BALL_W))

    jobs = [  # (src_root, label_remap, pseudo_model, pseudo_classes->unified, conf)
        (CONS, {0: 0, 1: 1, 2: 2}, ball, {1: 3}, 0.50),   # +hoop
        (POOL, {0: 2, 1: 3}, det, {0: 0, 1: 1}, 0.60),    # +player/ref
    ]
    n_img = n_pseudo = 0
    for split in ("train", "valid", "test"):
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
        for root, remap, pmodel, pmap, pconf in jobs:
            imgs = sorted((root / split / "images").glob("*"))
            batch_paths = []
            for ip in imgs:
                batch_paths.append(ip)
                if len(batch_paths) >= a.batch:
                    n_pseudo += process(batch_paths, root, split, remap, pmodel,
                                        pmap, pconf, a)
                    n_img += len(batch_paths)
                    batch_paths = []
            if batch_paths:
                n_pseudo += process(batch_paths, root, split, remap, pmodel,
                                    pmap, pconf, a)
                n_img += len(batch_paths)
        print(f"{split}: done ({n_img} imgs so far, {n_pseudo} pseudo boxes)",
              flush=True)

    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        "nc: 4\nnames:\n  0: player\n  1: referee\n  2: ball\n  3: hoop\n")
    print(f"UNIFIED DATASET -> {OUT}: {n_img} images, {n_pseudo} pseudo boxes")
    return 0


def process(paths, root, split, remap, pmodel, pmap, pconf, a):
    import numpy as np  # noqa: F401
    imgs = [cv2.imread(str(p)) for p in paths]
    keep = [(p, im) for p, im in zip(paths, imgs) if im is not None]
    if not keep:
        return 0
    res = pmodel.predict([im for _, im in keep], imgsz=a.imgsz, conf=pconf,
                         device=a.device, verbose=False)
    added = 0
    for (ip, im), r in zip(keep, res):
        h, w = im.shape[:2]
        real = parse_label(root / split / "labels" / (ip.stem + ".txt"), remap)
        lines = [yolo_line(c, denorm((c, cx, cy, bw, bh), w, h), w, h)
                 for c, cx, cy, bw, bh in real]
        real_boxes = [(c, denorm(row, w, h)) for row in real for c in [row[0]]]
        if r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for j in range(len(cls)):
                if int(cls[j]) not in pmap:
                    continue
                uc = pmap[int(cls[j])]
                box = tuple(float(v) for v in xyxy[j])
                if any(rc == uc and iou(box, rb) > 0.5 for rc, rb in real_boxes):
                    continue
                lines.append(yolo_line(uc, box, w, h))
                added += 1
        dst_img = OUT / split / "images" / ip.name
        if not dst_img.exists():
            os.link(ip, dst_img)
        (OUT / split / "labels" / (ip.stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""))
    return added


if __name__ == "__main__":
    raise SystemExit(main())
