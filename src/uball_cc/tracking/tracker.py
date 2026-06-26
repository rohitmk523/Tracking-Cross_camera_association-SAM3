"""Per-camera ByteTrack over our detector output (docs/05).

Uses the Roboflow **`trackers`** package (`ByteTrackTracker`) — the supported
successor to supervision's now-deprecated bundled `ByteTrack` (update() instead of
update_with_detections()). Fixed cameras → ByteTrack is a strong baseline; BoT-SORT
/ ReID (also in `trackers`) is the occlusion upgrade path. Tracks ENTITIES
(player + referee); the ball (class 2) is not a tracklet here.
`minimum_consecutive_frames` debounces flicker births (blog uses 3, docs/05).
"""
from __future__ import annotations

import numpy as np

from ..detection.base import Detection
from .types import Track

# Track players + referees; the ball (class 2) is handled separately, not as a track.
DEFAULT_TRACK_CLASSES = (0, 1)


class ByteTrackTracker:
    """Stateful per-camera tracker. One instance per camera; call `update` per frame.

    Defaults are permissive (activation 0.25, high-conf split 0.5) so our
    0.25-threshold detector actually initiates tracks — `trackers` defaults to a
    0.7 activation that would drop most basketball detections.
    """

    def __init__(
        self,
        cam: str,
        *,
        track_classes: tuple[int, ...] = DEFAULT_TRACK_CLASSES,
        track_activation_threshold: float = 0.25,
        high_conf_det_threshold: float = 0.5,
        minimum_iou_threshold: float = 0.2,
        lost_track_buffer: int = 30,
        minimum_consecutive_frames: int = 3,
        frame_rate: int = 30,
    ):
        import supervision as sv  # noqa: PLC0415
        from trackers import ByteTrackTracker as _ByteTrack  # noqa: PLC0415

        self.cam = cam
        self.track_classes = set(track_classes)
        self._sv = sv
        self._bt = _ByteTrack(
            lost_track_buffer=lost_track_buffer,
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            minimum_iou_threshold=minimum_iou_threshold,
            high_conf_det_threshold=high_conf_det_threshold,
        )

    def reset(self) -> None:
        """Clear tracker state (call between independent clips/games), if supported."""
        r = getattr(self._bt, "reset", None)
        if callable(r):
            r()

    def update(self, detections: list[Detection], frame: int) -> list[Track]:
        """Advance the tracker by one frame; return the confirmed tracks for it."""
        sv = self._sv
        dets = [d for d in detections if d.class_id in self.track_classes]
        if not dets:
            self._bt.update(sv.Detections.empty())          # advance lost buffers
            return []
        sd = sv.Detections(
            xyxy=np.array([d.box_xyxy for d in dets], dtype=float),
            confidence=np.array([d.score for d in dets], dtype=float),
            class_id=np.array([d.class_id for d in dets], dtype=int),
        )
        tracked = self._bt.update(sd)
        out: list[Track] = []
        for i in range(len(tracked)):
            tid = tracked.tracker_id[i] if tracked.tracker_id is not None else None
            if tid is None or int(tid) < 0:                 # unconfirmed (< min_consecutive)
                continue
            x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[i])
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            cls = int(tracked.class_id[i]) if tracked.class_id is not None else 0
            out.append(Track(self.cam, frame, int(tid), (x1, y1, x2, y2), conf, cls))
        return out
