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


def _torso_color_feat(crops: list[np.ndarray]) -> np.ndarray | None:
    """Saturation-weighted torso jersey-colour feature [S, V, S·cosH, S·sinH], averaged over a
    track's crops. Separates teams by JERSEY COLOUR directly (a bright/saturated team vs a dark/
    low-saturation one) — robust where SigLIP's appearance embedding mis-clusters a dark team that
    blends into shadows. S and V carry bright-vs-dark; S·(cosH,sinH) carries the (circular) hue."""
    import cv2  # noqa: PLC0415

    feats = []
    for c in crops:
        if c.size == 0:
            continue
        h, w = c.shape[:2]
        torso = c[int(h * 0.15):int(h * 0.55), int(w * 0.25):int(w * 0.75)]
        if torso.size == 0:
            continue
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
        hue = hsv[:, 0] * np.pi / 90.0                      # 0..180 -> 0..2pi
        s, v = hsv[:, 1] / 255.0, hsv[:, 2] / 255.0
        wgt = s + 0.05                                      # weight by saturation (ignore grey floor/skin)
        feats.append([np.average(s, weights=wgt), np.average(v, weights=wgt),
                      np.average(s * np.cos(hue), weights=wgt), np.average(s * np.sin(hue), weights=wgt)])
    return np.mean(feats, axis=0) if feats else None


def _torso_hue(crops: list[np.ndarray]) -> float | None:
    """Median hue of saturated jersey (upper-torso) pixels across a track's crops — a
    within-camera team signature used to map clusters -> A/B deterministically (so the near
    cameras agree on which team is A; absolute hue still drifts across cameras, so this is
    paired with near-camera-only team voting in the fusion)."""
    import cv2  # noqa: PLC0415

    hues = []
    for c in crops:
        if c.size == 0:
            continue
        h, w = c.shape[:2]
        torso = c[int(h * 0.15):int(h * 0.55), int(w * 0.25):int(w * 0.75)]
        if torso.size == 0:
            continue
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        m = hsv[:, :, 1] > 60
        if int(m.sum()) > 10:
            hues.append(float(np.median(hsv[:, :, 0][m])))
    return float(np.median(hues)) if hues else None


def majority_class(tracks: list[Track]) -> dict[int, int]:
    """Per-track MAJORITY detector class. ByteTrack matches by IoU only, so a player track
    can carry scattered referee-class frames (and vice versa); labelling per OBSERVATION
    made one NR track flip A<->REF 42 times and leaked REF votes into the fusion team
    counters (2026-07-02 audit). The track's majority class decides ALL its frames."""
    votes: dict[int, Counter] = {}
    for t in tracks:
        votes.setdefault(t.track_id, Counter())[t.class_id] += 1
    return {tid: c.most_common(1)[0][0] for tid, c in votes.items()}


def assign_teams(video_path: str | Path, tracks: list[Track], *,
                 sample_per_track: int = 6, embedder: SiglipEmbedder | None = None,
                 seed: int = 0, method: str = "color") -> tuple[list[Track], dict]:
    """Return (tracks_with_team, info). Players -> A/B (KMeans), referees -> REF.

    method="color" (default): cluster on the torso JERSEY-COLOUR feature -- the team-defining
    signal -- robust where SigLIP mis-clusters a dark team. method="siglip": the old appearance
    embedding (kept as a fallback for same-coloured teams where colour can't separate)."""
    crops_by_id = per_track_crops(video_path, tracks, class_id=0,
                                  sample_per_track=sample_per_track)
    track_feat: dict[int, np.ndarray] = {}
    if method == "color":
        for tid, crops in crops_by_id.items():
            f = _torso_color_feat(crops)
            if f is not None:
                track_feat[tid] = f
    else:
        embedder = embedder or SiglipEmbedder()
        for tid, crops in crops_by_id.items():
            emb = embedder.embed(crops)
            if len(emb):
                track_feat[tid] = emb.mean(axis=0)

    ids = list(track_feat.keys())
    maj = majority_class(tracks)
    player_ids = {tid for tid, c in maj.items() if c == 0}
    team_of: dict[int, str | None] = {}
    info = {"player_tracks": len(player_ids), "clustered_tracks": len(ids), "method": method}
    if len(ids) >= 2:
        from sklearn.cluster import KMeans  # noqa: PLC0415

        x = np.stack([track_feat[i] for i in ids])
        if method != "color":
            x = _l2(x)                                      # colour feat is already scaled; don't L2 it
        labels = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(x)
        # Map clusters -> A/B by a within-camera jersey colour (median torso hue), NOT cluster
        # size ("larger cluster" differs per camera). Lower hue -> "A". Far cameras whose crops
        # are too small to separate teams are excluded from team voting in the fusion, so their
        # (unreliable) labels here don't matter.
        cl_hue = {}
        for cl in (0, 1):
            hs = [_torso_hue(crops_by_id.get(ids[i], [])) for i in range(len(ids)) if labels[i] == cl]
            hs = [h for h in hs if h is not None]
            cl_hue[cl] = float(np.median(hs)) if hs else 999.0
        a_cluster = min(cl_hue, key=cl_hue.get)
        team_of = {tid: ("A" if labels[i] == a_cluster else "B") for i, tid in enumerate(ids)}
        info["team_counts"] = dict(Counter(team_of.values()))
        info["cluster_hue"] = cl_hue
    else:
        team_of = {tid: None for tid in ids}              # too few to cluster

    out = []
    for t in tracks:
        mc = maj.get(t.track_id, t.class_id)          # track majority, not this frame's class
        if mc == 1:
            out.append(t.with_attrs(team="REF"))
        elif mc == 0:
            out.append(t.with_attrs(team=team_of.get(t.track_id)))
        else:
            out.append(t)
    return out, info
