"""Team classification (docs/05): A/B via SigLIP embeddings + KMeans(k=2).

Players (class 0) are clustered into two teams from crop appearance; referees
(class 1) are labelled REF directly (the detector already separates them). Team is
assigned PER TRACK (a player's team is constant) from a few sampled crops — robust
and cheap (~tens of tracks to cluster, not thousands of boxes).

SigLIP (transformers) -> mean embedding per track -> KMeans(k=2). UMAP (docs/05) is
an optional refinement we skip: fixed cameras give clean separation.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .attributes import per_track_crops
from .types import Track

DEFAULT_SIGLIP = "google/siglip-base-patch16-224"


class SiglipEmbedder:
    """Appearance embeddings from a SigLIP vision tower (MPS/CUDA/CPU)."""

    def __init__(self, model_name: str = DEFAULT_SIGLIP, device: str | None = None):
        import torch  # noqa: PLC0415
        # Image processor only — we embed crops (vision tower), so we skip the SigLIP
        # text tokenizer (which would pull in SentencePiece for nothing).
        from transformers import AutoImageProcessor, SiglipVisionModel  # noqa: PLC0415

        self._torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        self._proc = AutoImageProcessor.from_pretrained(model_name)
        self._model = SiglipVisionModel.from_pretrained(model_name).to(self.device).eval()

    def embed(self, crops_bgr: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        import cv2  # noqa: PLC0415

        torch = self._torch
        out: list[np.ndarray] = []
        for i in range(0, len(crops_bgr), batch_size):
            rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr[i:i + batch_size] if c.size]
            if not rgb:
                continue
            inputs = self._proc(images=rgb, return_tensors="pt").to(self.device)
            with torch.no_grad():
                pooled = self._model(**inputs).pooler_output
            out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 768), dtype=float)


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def assign_teams(video_path: str | Path, tracks: list[Track], *,
                 sample_per_track: int = 6, embedder: SiglipEmbedder | None = None,
                 seed: int = 0) -> tuple[list[Track], dict]:
    """Return (tracks_with_team, info). Players -> A/B (KMeans), referees -> REF."""
    crops_by_id = per_track_crops(video_path, tracks, class_id=0,
                                  sample_per_track=sample_per_track)
    embedder = embedder or SiglipEmbedder()
    track_mean: dict[int, np.ndarray] = {}
    for tid, crops in crops_by_id.items():
        emb = embedder.embed(crops)
        if len(emb):
            track_mean[tid] = emb.mean(axis=0)

    ids = list(track_mean.keys())
    player_ids = {t.track_id for t in tracks if t.class_id == 0}
    team_of: dict[int, str | None] = {}
    info = {"player_tracks": len(player_ids), "clustered_tracks": len(ids),
            "embedder": embedder.device}
    if len(ids) >= 2:
        from sklearn.cluster import KMeans  # noqa: PLC0415

        x = _l2(np.stack([track_mean[i] for i in ids]))
        labels = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(x)
        major = Counter(labels).most_common(1)[0][0]      # larger cluster -> "A" (deterministic)
        team_of = {tid: ("A" if labels[i] == major else "B") for i, tid in enumerate(ids)}
        info["team_counts"] = dict(Counter(team_of.values()))
    else:
        team_of = {tid: None for tid in ids}              # too few to cluster

    out = []
    for t in tracks:
        if t.class_id == 1:
            out.append(t.with_attrs(team="REF"))
        elif t.class_id == 0:
            out.append(t.with_attrs(team=team_of.get(t.track_id)))
        else:
            out.append(t)
    return out, info
