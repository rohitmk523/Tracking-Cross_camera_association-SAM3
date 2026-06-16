#!/usr/bin/env python3
"""Evaluate a detector on the consolidated dataset (docs/13).

  # smoke baseline (dependency-free; proves the harness runs)
  python scripts/eval_detection.py --detector dummy --split test --max-images 200

  # our trained RF-DETR weights (needs `pip install -e '.[baseline]'`)
  python scripts/eval_detection.py --detector rfdetr \
      --weights runs/rfdetr_s_1280/best.pth --model small --split test

Reports mAP@50 / mAP@50-95 per class (player/ref/ball) + the far-endline ROI
band player recall. Writes a JSON to runs/eval/.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def _load_detector(a):
    if a.detector == "dummy":
        from uball_cc.detection.base import DummyDetector
        return DummyDetector()
    if a.detector == "rfdetr":
        # RF-DETR inference runs on MPS/CPU; set MPS watermarks before torch import.
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
        os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
        from uball_cc.detection.base import RFDETRDetector
        if not a.weights:
            raise SystemExit("--weights required for --detector rfdetr")
        return RFDETRDetector(a.weights, resolution=a.resolution,
                              threshold=a.threshold, model=a.model)
    raise SystemExit(f"unknown detector: {a.detector}")


def _print_summary(res: dict) -> None:
    bar = "=" * 70
    print(bar)
    failed = res.get("n_images_failed", 0)
    print(f"DETECTION EVAL  detector={res['detector']}  images={res['n_images']}  "
          f"preds={res['n_predictions']}" + (f"  FAILED_LOADS={failed}" if failed else ""))
    print(bar)
    o = res["overall"]
    print(f"  overall   mAP@50={o['mAP_50']}  mAP@[50:95]={o['mAP_50_95']}  "
          f"AP_small={o['AP_small']}  AR100={o['AR_100']}")
    for name, m in res["per_class"].items():
        print(f"  {name:8s}  mAP@50={m['mAP_50']}  mAP@[50:95]={m['mAP_50_95']}  "
              f"AP_small={m['AP_small']}")
    b = res["far_endline_band"]
    print(f"  >> FAR-ENDLINE BAND  player recall@0.5={b['recall@0.5']}  "
          f"(matched {b['player_matched']}/{b['player_gt_in_band']})")
    for ang, d in sorted(b.get("per_angle", {}).items()):
        print(f"        {ang}: recall@0.5={d['recall@0.5']}  ({d['matched']}/{d['gt']})")
    print(bar)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(REPO / "data" / "detect_consolidated"))
    ap.add_argument("--split", default="test", choices=("train", "valid", "test"))
    ap.add_argument("--detector", default="dummy", choices=("dummy", "rfdetr"))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--model", default="nano", choices=("nano", "small"))
    ap.add_argument("--resolution", type=int, default=1280)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--roi-config", default=str(REPO / "configs" / "eval_roi.yaml"))
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from uball_cc.eval.detection import evaluate_detection

    ds = Path(a.dataset)
    gt = ds / a.split / "_annotations.coco.json"
    images = ds / a.split / "images"
    if not gt.exists():
        raise SystemExit(f"missing COCO GT: {gt} -- run build_detection_dataset.py")
    roi_cfg = yaml.safe_load(Path(a.roi_config).read_text())

    detector = _load_detector(a)
    res = evaluate_detection(detector, gt, images, roi_cfg, max_images=a.max_images)
    res["split"] = a.split
    _print_summary(res)

    out = Path(a.out) if a.out else (REPO / "runs" / "eval" /
                                     f"detection_{a.detector}_{a.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"metrics -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
