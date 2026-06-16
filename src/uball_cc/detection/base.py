"""Detector interface + adapters (docs/04 output contract).

Any detector the eval harness consumes implements `predict(image_bgr)` and
returns a list of `Detection(box_xyxy, score, class_id)` where class_id indexes
the canonical taxonomy [0=player, 1=referee, 2=ball].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Detection:
    box_xyxy: tuple[float, float, float, float]
    score: float
    class_id: int


@runtime_checkable
class Detector(Protocol):
    def predict(self, image_bgr: np.ndarray) -> list[Detection]: ...


class DummyDetector:
    """Deterministic, dependency-free detector that proves the harness runs.

    Emits one centred player box per image (and nothing else). Useful as a
    smoke baseline and a control (its mAP should be ~0) when real weights are
    unavailable locally. NOT a model -- never report these numbers as quality.
    """

    name = "dummy"

    def __init__(self, score: float = 0.5):
        self.score = score

    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        h, w = image_bgr.shape[:2]
        bw, bh = 0.06 * w, 0.18 * h
        cx, cy = 0.5 * w, 0.55 * h
        box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
        return [Detection(box, self.score, 0)]


class RFDETRDetector:
    """Adapter over our trained RF-DETR weights (optional `baseline` deps).

    RF-DETR training is CUDA-only, but INFERENCE runs on MPS/CPU -- fine for the
    baseline eval. Set the MPS watermark env vars before importing torch on
    Apple silicon (PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5, LOW=0.4).
    """

    name = "rfdetr"

    def __init__(self, weights: str, resolution: int = 1280, threshold: float = 0.25,
                 model: str = "nano"):
        import cv2  # noqa: F401  (ensure cv2 available for color convert)
        from rfdetr import RFDETRNano, RFDETRSmall
        cls = RFDETRSmall if model == "small" else RFDETRNano
        self._model = cls(pretrain_weights=weights, resolution=resolution)
        self.threshold = threshold

    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        import cv2
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        d = self._model.predict(rgb, threshold=self.threshold)
        out: list[Detection] = []
        for i in range(len(d.xyxy)):
            x1, y1, x2, y2 = (float(v) for v in d.xyxy[i])
            out.append(Detection((x1, y1, x2, y2), float(d.confidence[i]),
                                 int(d.class_id[i])))
        return out
