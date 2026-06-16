#!/usr/bin/env python3
"""Train RF-DETR (player/referee/ball) on OUR consolidated dataset.

RF-DETR reads the YOLO dataset directly (dataset_file="yolo"; needs train/valid/
test + data.yaml -> build_detection_dataset.py produces all three). TRAINING is
CUDA-only (RF-DETR ops deadlock on Apple MPS), so this runs on the AWS GPU box
(scripts/aws_train.py). `--model nano` can smoke-train locally on MPS.

  python scripts/train_rfdetr.py --config configs/train_rfdetr.yaml   # uses config
  python scripts/train_rfdetr.py --model nano --epochs 5              # quick override
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "train_rfdetr.yaml"))
    ap.add_argument("--model", choices=("nano", "small"))
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--resolution", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--grad-accum", type=int)
    ap.add_argument("--dataset")
    ap.add_argument("--run-name")
    ap.add_argument("--class-names", help="comma-separated; overrides config")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text()) if Path(a.config).exists() else {}
    model = a.model or cfg.get("model", "small")
    epochs = a.epochs or cfg.get("epochs", 60)
    resolution = a.resolution or cfg.get("resolution", 1280)
    batch_size = a.batch_size or cfg.get("batch_size", 8)
    grad_accum = a.grad_accum or cfg.get("grad_accum", 2)
    class_names = ([c.strip() for c in a.class_names.split(",")] if a.class_names
                   else list(cfg.get("class_names", ["player", "referee", "ball"])))
    dataset = Path(a.dataset or cfg.get("dataset", "data/detect_consolidated"))
    if not dataset.is_absolute():
        dataset = REPO / dataset
    run_name = a.run_name or cfg.get("run_name", "rfdetr-s-1280-ourdata-v1")
    run_dir = REPO / "runs" / run_name

    if not (dataset / "data.yaml").exists():
        sys.exit(f"{dataset}/data.yaml missing -- run build_detection_dataset.py first")

    import torch
    from rfdetr import RFDETRNano, RFDETRSmall
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cpu":
        sys.exit("CPU training not viable; need CUDA (AWS) or MPS (nano smoke only)")
    if model == "small" and device == "mps":
        print("WARNING: RF-DETR-Small training on MPS may deadlock; use CUDA (aws_train.py).")
    print(f"model={model} device={device} res={resolution} epochs={epochs} "
          f"batch={batch_size} dataset={dataset}")
    run_dir.mkdir(parents=True, exist_ok=True)

    net = (RFDETRSmall if model == "small" else RFDETRNano)()
    net.train(
        dataset_dir=str(dataset), dataset_file="yolo",
        epochs=epochs, resolution=resolution,
        batch_size=batch_size, grad_accum_steps=grad_accum,
        device=device, output_dir=str(run_dir),
        early_stopping=cfg.get("early_stopping", True),
        early_stopping_patience=cfg.get("early_stopping_patience", 12),
        num_workers=4, class_names=class_names, run_test=True,
    )
    print(f"\nRF-DETR weights + logs -> {run_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
