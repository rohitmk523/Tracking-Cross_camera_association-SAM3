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
        # FP16 tensor-core path: ~4-8x throughput on NVIDIA GPUs at our thresholds.
        # CUDA-only; never block detection if unavailable.
        try:
            import torch
            if torch.cuda.is_available():
                self._model.optimize_for_inference(dtype=torch.float16)
        except Exception as e:
            print(f"[rfdetr] optimize_for_inference skipped: {e}")

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

class CachedDetector:
    """Wraps any Detector with an npz on-disk cache keyed by frame index.

    Detection is ~85% of pipeline wall time and is IDENTICAL across tracking /
    fusion iterations — cache once per clip, then association experiments re-run
    in seconds. Cache format: boxes (N,4) float32, scores (N,), classes (N,),
    frame_idx (N,) int32.
    """

    def __init__(self, inner, cache_path):
        import numpy as np
        from pathlib import Path as _P
        self._inner = inner
        self._path = _P(cache_path)
        self._cache: dict[int, list[Detection]] = {}
        self._next_frame = 0
        self._dirty = False
        if self._path.exists():
            z = np.load(self._path)
            for b, s, c, f in zip(z["boxes"], z["scores"], z["classes"], z["frame_idx"]):
                self._cache.setdefault(int(f), []).append(
                    Detection(tuple(float(v) for v in b), float(s), int(c)))
            # frames with zero detections still count as cached: track via max index
            self._max_cached = int(z["frame_idx"].max()) if len(z["frame_idx"]) else -1
            self._n_frames_cached = int(z.get("n_frames", [self._max_cached + 1])[0])
        else:
            self._n_frames_cached = 0

    def predict(self, image_bgr) -> "list[Detection]":
        f = self._next_frame
        self._next_frame += 1
        if f < self._n_frames_cached:
            return self._cache.get(f, [])
        dets = self._inner.predict(image_bgr)
        self._cache[f] = dets
        self._dirty = True
        return dets

    def flush(self) -> None:
        if not self._dirty:
            return
        import numpy as np
        rows = [(d.box_xyxy, d.score, d.class_id, f)
                for f, ds in sorted(self._cache.items()) for d in ds]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self._path,
            boxes=np.array([r[0] for r in rows], np.float32).reshape(-1, 4),
            scores=np.array([r[1] for r in rows], np.float32),
            classes=np.array([r[2] for r in rows], np.int32),
            frame_idx=np.array([r[3] for r in rows], np.int32),
            n_frames=np.array([self._next_frame], np.int32))
        self._dirty = False
