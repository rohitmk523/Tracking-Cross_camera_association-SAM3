"""ReID appearance embeddings (docs/05): Torchreid OSNet vector per track.

A 512-d L2-normalized embedding per track (mean over sampled crops), for cross-camera
matching + re-entry in the fusion stage. Default backbone is imagenet-pretrained
OSNet; pass a ReID-trained `model_path` (e.g. osnet_x1_0_market1501, or the multi-source
domain-generalization osnet_ain_x1_0) for production-grade person ReID, or fine-tune on
basketball crops (docs/05).

Security: a ReID-trained `model_path` is loaded via PyTorch's `weights_only=True` safe
mode (tensors only, no pickle code-execution), and the OSNet architecture is built
directly rather than through torchreid's FeatureExtractor (whose internal loader does
NOT use weights_only). The imagenet default path keeps using FeatureExtractor.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .attributes import per_track_crops
from .types import Track

DEFAULT_OSNET = "osnet_x1_0"
_CACHED_MARKET = os.path.expanduser("~/.cache/torch/checkpoints/osnet_x1_0_market1501.pth")


def default_reid_weights() -> str:
    """ReID-trained weights to prefer over imagenet, if available: `UBALL_REID_WEIGHTS`
    env override, else the cached Market1501 OSNet, else "" (imagenet fallback). Loaded
    via weights_only=True (see OSNetEmbedder)."""
    env = os.environ.get("UBALL_REID_WEIGHTS", "")
    if env and os.path.exists(env):
        return env
    return _CACHED_MARKET if os.path.exists(_CACHED_MARKET) else ""
# torchreid / FeatureExtractor preprocessing constants (must match for the manual path)
_INPUT_HW = (256, 128)                                    # (h, w)
_PIX_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_PIX_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _safe_state_dict(model_path: str, torch) -> dict:
    """Load a ReID checkpoint with weights_only=True (no pickle code-execution), and
    normalize it to a {param_name: tensor} dict (unwrap 'state_dict', strip 'module.')."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


class OSNetEmbedder:
    """Torchreid OSNet feature extractor (MPS/CUDA/CPU). Embeds BGR crops -> (N, 512).

    `model_path` (ReID-trained weights) -> safe manual path; otherwise imagenet via
    FeatureExtractor."""

    def __init__(self, model_name: str = DEFAULT_OSNET, model_path: str = "",
                 device: str | None = None):
        import torch  # noqa: PLC0415

        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self._safe = bool(model_path)
        if not self._safe:
            from torchreid.reid.utils import FeatureExtractor  # noqa: PLC0415
            self._ex = FeatureExtractor(model_name=model_name, model_path="", device=self.device)
            return
        # --- safe ReID-trained path: build arch (no download), inject weights_only state dict ---
        from torchreid.reid.models import build_model  # noqa: PLC0415
        model = build_model(model_name, num_classes=1, loss="softmax", pretrained=False)
        sd = _safe_state_dict(model_path, torch)
        msd = model.state_dict()
        matched = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
        model.load_state_dict(matched, strict=False)        # classifier head intentionally skipped
        self._loaded, self._total = len(matched), len(msd)
        self._model = model.eval().to(self.device)

    def embed(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        import cv2  # noqa: PLC0415

        rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr if c.size]
        if not rgb:
            return np.zeros((0, 512), dtype=float)
        if not self._safe:
            feats = self._ex(rgb)                           # torch tensor (N, 512)
            return feats.float().cpu().numpy()
        # manual preprocessing identical to FeatureExtractor (resize -> /255 -> normalize)
        batch = np.stack([cv2.resize(r, (_INPUT_HW[1], _INPUT_HW[0])) for r in rgb]).astype(np.float32)
        batch = (batch / 255.0 - _PIX_MEAN) / _PIX_STD
        x = self._torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().to(self.device)
        with self._torch.no_grad():
            feats = self._model(x)
        return feats.float().cpu().numpy()


def track_embeddings(video_path: str | Path, tracks: list[Track], *,
                     sample_per_track: int = 6,
                     embedder: OSNetEmbedder | None = None) -> dict[int, np.ndarray]:
    """{player track_id -> L2-normalized mean OSNet embedding} for cross-camera matching."""
    crops_by_id = per_track_crops(video_path, tracks, class_id=0,
                                  sample_per_track=sample_per_track)
    embedder = embedder or OSNetEmbedder()
    out: dict[int, np.ndarray] = {}
    for tid, crops in crops_by_id.items():
        emb = embedder.embed(crops)
        if len(emb):
            v = emb.mean(axis=0)
            out[tid] = v / (np.linalg.norm(v) + 1e-8)
    return out
