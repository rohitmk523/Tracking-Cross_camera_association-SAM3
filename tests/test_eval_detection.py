"""Smoke tests for the detection eval harness (docs/13 regression discipline)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from uball_cc.data.build_dataset import _yolo_to_coco_box
from uball_cc.detection.base import Detection, DummyDetector
from uball_cc.eval.detection import _iou_xyxy, evaluate_detection


def test_iou_basic():
    assert _iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert _iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert _iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3, abs=1e-6)


def test_yolo_to_coco_box():
    # centred half-size box on a 100x100 image -> [25, 25, 50, 50]
    x, y, w, h = _yolo_to_coco_box("0 0.5 0.5 0.5 0.5", 100, 100)
    assert (x, y, w, h) == pytest.approx((25, 25, 50, 50))


def _tiny_dataset(tmp: Path) -> tuple[Path, Path]:
    """Two 200x200 images, one player GT box each, + COCO GT."""
    images = tmp / "images"
    images.mkdir(parents=True)
    box = [80, 60, 40, 80]  # xywh, a player
    anns, imgs = [], []
    for i, ang in enumerate(("FL", "NL"), start=1):
        name = f"c2a354fe_{ang}_f0000{i}.jpg"
        cv2.imwrite(str(images / name), np.full((200, 200, 3), 127, np.uint8))
        imgs.append({"id": i, "file_name": name, "width": 200, "height": 200})
        anns.append({"id": i, "image_id": i, "category_id": 1, "bbox": box,
                     "area": box[2] * box[3], "iscrowd": 0})
    gt = tmp / "_annotations.coco.json"
    gt.write_text(json.dumps({"images": imgs, "annotations": anns,
                              "categories": [{"id": 1, "name": "player"},
                                             {"id": 2, "name": "referee"},
                                             {"id": 3, "name": "ball"}]}))
    return gt, images


class _PerfectDetector:
    name = "perfect"

    def predict(self, image_bgr):
        return [Detection((80, 60, 120, 140), 0.99, 0)]  # exactly the GT box


def test_perfect_detector_scores_high(tmp_path):
    gt, images = _tiny_dataset(tmp_path)
    roi = {"band_mode": "small_box_percentile", "small_box_percentile": 50}
    res = evaluate_detection(_PerfectDetector(), gt, images, roi)
    assert res["per_class"]["player"]["mAP_50"] == pytest.approx(1.0)
    assert res["far_endline_band"]["recall@0.5"] == pytest.approx(1.0)


def test_dummy_detector_scores_low(tmp_path):
    gt, images = _tiny_dataset(tmp_path)
    roi = {"band_mode": "small_box_percentile", "small_box_percentile": 50}
    res = evaluate_detection(DummyDetector(), gt, images, roi)
    assert res["per_class"]["player"]["mAP_50"] < 0.5
