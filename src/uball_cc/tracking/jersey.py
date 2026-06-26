"""Jersey number reading (docs/05): ResNet-18 reader over per-track number crops.

Loads the e6 reader `e6_jersey_resnet_v2.pt` (ResNet-18, 96x96, Resize+ToTensor, NO
normalize; ckpt {state_dict, labels}). NOTE: this v2 is the **e6-specific** reader — its
`labels` are only the 8 numbers worn in e6fba750, so it cannot output numbers outside
that set (general OCR is a follow-on, docs/10). Without the number-localizer we
approximate the number region as the upper-torso of the player box; swap the localizer
in for accuracy.

Per track: read several sampled crops, keep legible (high-confidence) reads, and commit
the confidence-weighted majority number only if enough reads agree — the "legible on a
minority of frames" gate (docs/09). Numbers not consensual stay None (unknown).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .attributes import per_track_crops
from .types import Track

_TF = "/Users/rohitkale/Cellstrat/GitHub_Repositories/Training_frameworks/Uball E6 Demo/runs/"
DEFAULT_JERSEY_WEIGHTS = _TF + "e6_jersey_resnet_v2.pt"
DEFAULT_LOCALIZER_WEIGHTS = _TF + "e6_number_localizer_compat.pth"   # RF-DETR-Nano, class ['number']


def _torso(crop_bgr: np.ndarray, top: float = 0.12, bot: float = 0.55,
           lr: float = 0.15) -> np.ndarray:
    """Fallback number region = upper-central torso (no localizer; unreliable on far cams)."""
    h, w = crop_bgr.shape[:2]
    if h < 4 or w < 4:
        return crop_bgr
    sub = crop_bgr[int(top * h):int(bot * h), int(lr * w):int((1 - lr) * w)]
    return sub if sub.size else crop_bgr


class NumberLocalizer:
    """RF-DETR-Nano that crops the tight jersey-number box from a player crop (docs/05).

    The jersey reader needs a TIGHT number crop, not a loose torso. This finds it.
    Reliable only where the number is big enough — i.e. near cameras / close players.
    """

    def __init__(self, weights: str = DEFAULT_LOCALIZER_WEIGHTS, resolution: int = 384,
                 threshold: float = 0.3):
        from rfdetr import RFDETRNano  # noqa: PLC0415

        self._m = RFDETRNano(pretrain_weights=weights, resolution=resolution)
        self.threshold = threshold

    def locate(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Return the tight number-box sub-crop (highest conf), or None if not found."""
        import cv2  # noqa: PLC0415

        if crop_bgr is None or crop_bgr.size == 0:
            return None
        d = self._m.predict(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), threshold=self.threshold)
        if len(d.xyxy) == 0:
            return None
        i = max(range(len(d.xyxy)), key=lambda j: float(d.confidence[j]))
        x1, y1, x2, y2 = (int(v) for v in d.xyxy[i])
        h, w = crop_bgr.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        return crop_bgr[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None


class JerseyReader:
    """ResNet-18 jersey-number reader. predict(crops) -> [(label_str, confidence)]."""

    def __init__(self, weights: str = DEFAULT_JERSEY_WEIGHTS, device: str | None = None):
        import torch  # noqa: PLC0415
        import torch.nn as nn  # noqa: PLC0415
        from torchvision.models import resnet18  # noqa: PLC0415

        ck = torch.load(weights, map_location="cpu", weights_only=False)
        self.labels = [str(x) for x in ck["labels"]]
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, len(self.labels))
        m.load_state_dict(ck["state_dict"])
        self._m = m.to(self.device).eval()
        self._torch = torch

    def predict(self, crops_bgr: list[np.ndarray]) -> list[tuple[str, float]]:
        import cv2  # noqa: PLC0415
        import torch.nn.functional as fn  # noqa: PLC0415

        torch = self._torch
        tensors, idx_map = [], []
        for i, c in enumerate(crops_bgr):
            if c is None or c.size == 0:
                continue
            rgb = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (96, 96))
            tensors.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
            idx_map.append(i)
        out: list[tuple[str, float]] = [("", 0.0)] * len(crops_bgr)
        if not tensors:
            return out
        with torch.no_grad():
            probs = fn.softmax(self._m(torch.stack(tensors).to(self.device)), dim=1)
        conf, idx = probs.max(dim=1)
        for i, ci, ii in zip(idx_map, conf.cpu().tolist(), idx.cpu().tolist()):
            out[i] = (self.labels[ii], float(ci))
        return out


def read_jerseys(video_path, tracks: list[Track], *, sample_per_track: int = 10,
                 reader: JerseyReader | None = None, localizer: NumberLocalizer | None = None,
                 conf_threshold: float = 0.6, min_legible: int = 2) -> tuple[list[Track], dict]:
    """Set `jersey` on player tracks via confidence-gated consensus. Returns (tracks, info).

    With a `localizer`, each crop is reduced to its tight number box first (recommended);
    without one we fall back to a torso crop (unreliable). A track commits a number only
    if >= `min_legible` confident reads agree (the 'legible on a minority of frames' gate).
    """
    crops_by_id = per_track_crops(video_path, tracks, class_id=0,
                                  sample_per_track=sample_per_track)
    reader = reader or JerseyReader()
    jersey_of: dict[int, int | None] = {}
    n_boxes = 0
    for tid, crops in crops_by_id.items():
        if localizer is not None:
            num_crops = [c for c in (localizer.locate(c) for c in crops) if c is not None]
        else:
            num_crops = [_torso(c) for c in crops]
        n_boxes += len(num_crops)
        preds = reader.predict(num_crops) if num_crops else []
        legible = [(lab, conf) for lab, conf in preds if lab and conf >= conf_threshold]
        if len(legible) >= min_legible:
            score: dict[str, float] = defaultdict(float)
            for lab, conf in legible:
                score[lab] += conf
            jersey_of[tid] = int(max(score, key=score.get))
        else:
            jersey_of[tid] = None

    out = [t.with_attrs(jersey=jersey_of[t.track_id])
           if (t.class_id == 0 and jersey_of.get(t.track_id) is not None) else t
           for t in tracks]
    info = {"player_tracks": len(jersey_of),
            "jersey_read": sum(v is not None for v in jersey_of.values()),
            "number_boxes_found": n_boxes, "used_localizer": localizer is not None,
            "labels": reader.labels}
    return out, info
