#!/usr/bin/env python3
"""Score candidate PRE-LABELING models against the operator-approved frames
(their corrected labels = ground truth). Per-class precision/recall/F1 for:
  - HF broadcast RF-DETR (koppolusameer/...): player/referee/ball in one model
  - our current: e6 RF-DETR (player/referee) + near rim RF-DETR (ball)

  python scripts/eval_prelabel_models.py            # all approved frames

Canonical ids: 0=player 1=referee 2=ball. Tells us which gives cleaner seeds.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TF = REPO.parent / "Training_frameworks"
POOL = REPO / "data" / "annotate_pool"
NAMES = {0: "player", 1: "referee", 2: "ball"}
HF_MODEL = "koppolusameer/rfdetr-basketball-player-ball-referee-detection"
HF_REMAP = {0: 2, 1: 0, 2: 1}   # HF (ball,player,referee) -> canonical (player,ref,ball)


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _load_gt(stem, w, h):
    lf = POOL / "labels" / f"{stem}.txt"
    out = []
    if not lf.exists():
        return out
    for ln in lf.read_text().splitlines():
        p = ln.split()
        if len(p) >= 5:
            c = int(p[0])
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            out.append((c, [(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h]))
    return out


def _score(preds_by_stem, gts_by_stem, iou_thr=0.5):
    st = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for stem, gt in gts_by_stem.items():
        preds = preds_by_stem.get(stem, [])
        for cls in (0, 1, 2):
            g = [b for c, b in gt if c == cls]
            p = sorted([(s, b) for c, s, b in preds if c == cls], key=lambda x: -x[0])
            used = [False] * len(g)
            for _, pb in p:
                hit = -1
                best = iou_thr
                for j, gb in enumerate(g):
                    if not used[j] and _iou(pb, gb) >= best:
                        best, hit = _iou(pb, gb), j
                if hit >= 0:
                    used[hit] = True
                    st[cls]["tp"] += 1
                else:
                    st[cls]["fp"] += 1
            st[cls]["fn"] += used.count(False)
    return st


def _prf(s):
    tp, fp, fn = s["tp"], s["fp"], s["fn"]
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2*pr*rc/(pr+rc) if pr+rc else 0.0
    return pr, rc, f1, tp, fp, fn


def main():
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
    import cv2
    import torch
    from PIL import Image

    state = json.loads((POOL / "review_state.json").read_text())
    stems = [k for k, v in state.items() if v.get("status") == "approved"]
    print(f"approved (GT) frames: {len(stems)}")
    sizes, gts = {}, {}
    for stem in stems:
        ip = POOL / "images" / f"{stem}.jpg"
        if not ip.exists():
            continue
        with Image.open(ip) as im:
            w, h = im.size
        sizes[stem] = (w, h)
        gts[stem] = _load_gt(stem, w, h)
    stems = list(sizes)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    # ---- HF broadcast RF-DETR (all 3 classes, one model) ----
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    proc = AutoImageProcessor.from_pretrained(HF_MODEL)
    hf = AutoModelForObjectDetection.from_pretrained(HF_MODEL).to(dev).eval()
    hf_preds = {}
    for i, stem in enumerate(stems, 1):
        w, h = sizes[stem]
        im = Image.open(POOL / "images" / f"{stem}.jpg").convert("RGB")
        with torch.no_grad():
            o = hf(**proc(images=im, return_tensors="pt").to(dev))
        r = proc.post_process_object_detection(o, threshold=0.3, target_sizes=[(h, w)])[0]
        hf_preds[stem] = [(HF_REMAP[int(c)], float(s), [float(v) for v in b])
                          for c, s, b in zip(r["labels"], r["scores"], r["boxes"])]
        if i % 60 == 0:
            print(f"  HF {i}/{len(stems)}")
    del hf

    # ---- our current: e6 (player/ref) + near rim (ball) ----
    from rfdetr import RFDETRNano, RFDETRSmall
    e6 = RFDETRNano(pretrain_weights=str(TF/"Uball E6 Demo"/"runs"/"e6_rfdetr_best.pth"),
                    resolution=1280)
    ball = RFDETRSmall(pretrain_weights=str(REPO/"runs"/"rfdetr-rim-near-v1"/"best.pth"),
                       resolution=1280)
    ours_preds = {}
    for i, stem in enumerate(stems, 1):
        bgr = cv2.imread(str(POOL / "images" / f"{stem}.jpg"))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pr = []
        d = e6.predict(rgb, threshold=0.35)
        for j in range(len(d.xyxy)):
            c = int(d.class_id[j])              # 0 player, 1 referee
            pr.append((1 if c == 1 else 0, float(d.confidence[j]),
                       [float(v) for v in d.xyxy[j]]))
        d = ball.predict(rgb, threshold=0.30)
        for j in range(len(d.xyxy)):
            if int(d.class_id[j]) == 0:         # Basketball -> ball(2)
                pr.append((2, float(d.confidence[j]), [float(v) for v in d.xyxy[j]]))
        ours_preds[stem] = pr
        if i % 60 == 0:
            print(f"  ours {i}/{len(stems)}")

    # ---- score + report ----
    print("\n" + "=" * 64)
    print(f"PRE-LABELER vs {len(stems)} approved-GT frames  (IoU>=0.5)")
    print("=" * 64)
    for name, preds in [("HF broadcast RF-DETR", hf_preds),
                        ("ours (e6 + near-ball)", ours_preds)]:
        st = _score(preds, gts)
        print(f"\n[{name}]")
        print(f"  {'class':9s} {'prec':>6s} {'recall':>7s} {'F1':>6s}   (tp/fp/fn)")
        for cls in (0, 1, 2):
            pr, rc, f1, tp, fp, fn = _prf(st[cls])
            print(f"  {NAMES[cls]:9s} {pr:6.3f} {rc:7.3f} {f1:6.3f}   ({tp}/{fp}/{fn})")
    print("\n>> Higher recall = fewer missed boxes to draw; higher precision = "
          "fewer wrong boxes to delete. Best seed = high both, esp. referee.")


if __name__ == "__main__":
    main()
