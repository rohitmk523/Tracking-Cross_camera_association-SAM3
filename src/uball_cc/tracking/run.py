"""Run a per-camera tracker over a frame sequence (video or image dir).

detect (per frame) -> ByteTrack -> tracklets. Frame iteration is decoupled so the
same `track_sequence` works on an mp4, a directory of frames, or any iterator.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..detection.base import Detector
from .tracker import ByteTrackTracker
from .types import Track


def iter_video_frames(path: str | Path, max_frames: int | None = None,
                      stride: int = 1) -> Iterator[np.ndarray]:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(path))
    i = yielded = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                yield frame
                yielded += 1
                if max_frames and yielded >= max_frames:
                    break
            i += 1
    finally:
        cap.release()


def iter_image_frames(directory: str | Path,
                      max_frames: int | None = None) -> Iterator[np.ndarray]:
    import cv2  # noqa: PLC0415

    files = sorted(p for p in Path(directory).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if max_frames:
        files = files[:max_frames]
    for f in files:
        img = cv2.imread(str(f))
        if img is not None:
            yield img


def track_stream(detector: Detector, frames: Iterator[np.ndarray], cam: str,
                 tracker: ByteTrackTracker | None = None, **tracker_kw):
    """Yield (frame_idx, frame_bgr, tracks_this_frame) — for rendering / streaming."""
    tracker = tracker or ByteTrackTracker(cam, **tracker_kw)
    for frame_idx, img in enumerate(frames):
        yield frame_idx, img, tracker.update(detector.predict(img), frame_idx)


def track_sequence(detector: Detector, frames: Iterator[np.ndarray], cam: str,
                   tracker: ByteTrackTracker | None = None, **tracker_kw) -> list[Track]:
    """Detect+track every frame in order; return the flat list of per-frame tracks."""
    out: list[Track] = []
    for _, _, tracks in track_stream(detector, frames, cam, tracker, **tracker_kw):
        out.extend(tracks)
    return out


# Distinct BGR colors so adjacent track ids are easy to tell apart when eyeballing.
_PALETTE = [
    (245, 135, 66), (90, 66, 245), (131, 245, 66), (66, 206, 245), (245, 66, 206),
    (66, 245, 138), (245, 66, 90), (206, 245, 66), (138, 66, 245), (66, 245, 239),
]


def _color(track_id: int):
    return _PALETTE[track_id % len(_PALETTE)]


# Team overlay colors (BGR): A=orange, B=blue, REF=yellow. Used when a track has a team.
_TEAM_COLOR = {"A": (0, 140, 255), "B": (255, 120, 40), "REF": (0, 255, 255)}


def render_frame(image_bgr: np.ndarray, tracks: list[Track]):
    """Draw boxes + labels. Color by TEAM (A/B/REF) when set, else by track id."""
    import cv2  # noqa: PLC0415

    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.box_xyxy)
        c = _TEAM_COLOR.get(t.team) if t.team else _color(t.track_id)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), c, 2)
        tag = t.team if t.team else t.class_name[:3]
        num = f"#{t.jersey}" if t.jersey is not None else f"t{t.track_id}"   # jersey if read, else track id
        label = f"{tag} {num}"
        ytxt = max(13, y1 - 5)
        cv2.putText(image_bgr, label, (x1, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)
    return image_bgr


def summarize(tracks: list[Track]) -> dict:
    """Quick stability stats (a pre-eval sanity signal: id count, track lengths)."""
    from collections import Counter

    frames = {t.frame for t in tracks}
    per_id = Counter(t.track_id for t in tracks)
    lengths = sorted(per_id.values())
    n = len(lengths)
    return {
        "n_frames_with_tracks": len(frames),
        "n_unique_track_ids": n,
        "n_track_observations": len(tracks),
        "track_len_min": lengths[0] if n else 0,
        "track_len_median": lengths[n // 2] if n else 0,
        "track_len_max": lengths[-1] if n else 0,
        "avg_tracks_per_frame": round(len(tracks) / max(1, len(frames)), 2),
    }
