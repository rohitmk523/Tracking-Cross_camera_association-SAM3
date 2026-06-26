"""Per-track attribute interface: team / jersey# / ReID (docs/05).

The tracker emits geometry (boxes + foot points + local ids). This stage enriches
each track from its image crop. We fix the INTERFACE now so the cross-camera fusion
contract is stable; the three concrete providers are the next sub-step (D5):

  - TeamClassifier : roboflow/sports SigLIP -> UMAP -> KMeans(k=2) -> A/B (+REF from
                     the detector's referee class).
  - JerseyReader   : our trained ResNet number reader (tight number-box crop -> digit).
  - ReIDEmbedder   : Torchreid OSNet appearance vector (cross-camera match / re-entry).

Each implements `AttributeProvider.annotate(track, crop_bgr) -> Track`. Until they
land, `NoOpAttributes` passes tracks through unchanged so the pipeline runs.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import Track


@runtime_checkable
class AttributeProvider(Protocol):
    def annotate(self, track: Track, crop_bgr: np.ndarray) -> Track:
        """Return a NEW Track with attribute(s) set from the crop (immutable)."""
        ...


class NoOpAttributes:
    """Passthrough provider (default until the real providers land)."""

    name = "noop"

    def annotate(self, track: Track, crop_bgr: np.ndarray) -> Track:
        return track


def crop_of(track: Track, image_bgr: np.ndarray, pad_frac: float = 0.0) -> np.ndarray:
    """Box crop for a track, clamped to image bounds, optionally padded."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = track.box_xyxy
    if pad_frac:
        pw, ph = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
        x1, y1, x2, y2 = x1 - pw, y1 - ph, x2 + pw, y2 + ph
    xi1, yi1 = max(0, int(x1)), max(0, int(y1))
    xi2, yi2 = min(w, int(round(x2))), min(h, int(round(y2)))
    if xi2 <= xi1 or yi2 <= yi1:
        return image_bgr[0:0, 0:0]
    return image_bgr[yi1:yi2, xi1:xi2]


# --- shared crop-sampling for the batch attribute stages (team, reid) ---

def sample_track(tracks: list[Track], k: int) -> list[Track]:
    """Evenly sample up to k observations of one track (by frame order)."""
    ts = sorted(tracks, key=lambda x: x.frame)
    if len(ts) <= k:
        return ts
    idx = np.linspace(0, len(ts) - 1, k).astype(int)
    return [ts[i] for i in idx]


def read_frames(video_path, need_frames: set[int]) -> dict[int, np.ndarray]:
    """Sequential read keeping only the wanted frame indices (matches tracker indexing)."""
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(video_path))
    out: dict[int, np.ndarray] = {}
    i, last = 0, (max(need_frames) if need_frames else -1)
    try:
        while i <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if i in need_frames:
                out[i] = frame
            i += 1
    finally:
        cap.release()
    return out


def per_track_crops(video_path, tracks: list[Track], *, class_id: int = 0,
                    sample_per_track: int = 6) -> dict[int, list[np.ndarray]]:
    """{track_id: [crop_bgr, ...]} for tracks of `class_id`, sampled across frames."""
    from collections import defaultdict  # noqa: PLC0415

    by_id: dict[int, list[Track]] = defaultdict(list)
    for t in tracks:
        if t.class_id == class_id:
            by_id[t.track_id].append(t)
    sampled = {tid: sample_track(ts, sample_per_track) for tid, ts in by_id.items()}
    frames = read_frames(video_path, {t.frame for ts in sampled.values() for t in ts})
    out: dict[int, list[np.ndarray]] = {}
    for tid, ts in sampled.items():
        crops = [c for t in ts if (img := frames.get(t.frame)) is not None
                 and (c := crop_of(t, img)).size]
        if crops:
            out[tid] = crops
    return out
