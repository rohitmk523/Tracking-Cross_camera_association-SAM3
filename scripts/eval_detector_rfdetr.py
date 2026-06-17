#!/usr/bin/env python3
"""RF-DETR port of uball_shot_detection_dual_fusion_v2/near_v0/eval_detector.py.

IDENTICAL metric logic (IoU@0.3 matching, per-class recall/precision, BALL recall
at the rim moment) -- only the detector is swapped YOLO -> RF-DETR, so the numbers
are apples-to-apples with the YOLO eval_detector.py output. Class 0=Basketball,
1=Basketball Hoop. conf=0.25, resolution=1280 (matching the YOLO eval).

  python scripts/eval_detector_rfdetr.py --weights runs/rfdetr-rim-near-v1/best.pth
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

BALL, HOOP = 0, 1
DEFAULT_SPLIT = ("/Users/rohitkale/Cellstrat/GitHub_Repositories/"
                 "Training_frameworks/Uball Near Angle/data/yolo_split/test")


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(lbl: Path, W, H):
    out = []
    if not lbl.exists():
        return out
    for line in lbl.read_text().splitlines():
        p = line.split()
        if not p:
            continue
        c = int(p[0])
        cx, cy, w, h = (float(v) for v in p[1:5])
        out.append((c, [(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--resolution", type=int, default=1280)
    a = ap.parse_args()

    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
    import cv2
    from rfdetr import RFDETRSmall
    model = RFDETRSmall(pretrain_weights=a.weights, resolution=a.resolution)

    imgs = sorted((Path(a.split) / "images").glob("*.jpg"))
    lbls = Path(a.split) / "labels"
    st = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0, "n": 0})
    rim_ball = {"tp": 0, "fn": 0}

    for img in imgs:
        kind = img.stem.rsplit("_", 1)[-1]
        bgr = cv2.imread(str(img))
        H, W = bgr.shape[:2]
        gt = load_gt(lbls / f"{img.stem}.txt", W, H)
        d = model.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), threshold=a.conf)
        preds = [(int(d.class_id[i]), [float(v) for v in d.xyxy[i]])
                 for i in range(len(d.xyxy))]
        for cls in (BALL, HOOP):
            g = [b for c, b in gt if c == cls]
            p = [b for c, b in preds if c == cls]
            matched = set()
            for gb in g:
                hit = any(iou(gb, pb) >= 0.3 and j not in matched
                          for j, pb in enumerate(p))
                if hit:
                    st[cls]["tp"] += 1
                    for j, pb in enumerate(p):
                        if iou(gb, pb) >= 0.3 and j not in matched:
                            matched.add(j)
                            break
                else:
                    st[cls]["fn"] += 1
            st[cls]["fp"] += len(p) - len(matched)
            st[cls]["n"] += len(g)
            if cls == BALL and kind == "rim" and g:
                if any(iou(gb, pb) >= 0.3 for gb in g for pb in p):
                    rim_ball["tp"] += 1
                else:
                    rim_ball["fn"] += 1

    print(f"weights={Path(a.weights).name}  test imgs={len(imgs)}  (RF-DETR)\n")
    for cls, name in [(HOOP, "HOOP"), (BALL, "BALL")]:
        s = st[cls]
        rec = s["tp"] / max(1, s["tp"] + s["fn"])
        prec = s["tp"] / max(1, s["tp"] + s["fp"])
        print(f"{name}: recall={rec:.3f} precision={prec:.3f} "
              f"(tp={s['tp']} fn={s['fn']} fp={s['fp']} gt={s['n']})")
    rb = rim_ball["tp"] / max(1, rim_ball["tp"] + rim_ball["fn"])
    print(f"\nBALL recall at RIM MOMENT: {rb:.3f} "
          f"(tp={rim_ball['tp']} fn={rim_ball['fn']})  <- decides make/miss")


if __name__ == "__main__":
    main()
