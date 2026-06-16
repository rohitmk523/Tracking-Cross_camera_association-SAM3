"""Per-camera tracking metrics (docs/13) -- STUB.

Implemented in Week-1 D5 once the ByteTrack/BoT-SORT stage emits per-camera
tracklets. Benchmarks: SportsMOT basketball split / TrackID3x3 (research-license,
EVAL ONLY -- never trained on, per docs/10).

Metrics: MOTA, IDF1, ID-switches per camera; plus jersey-read precision and
team-classification accuracy (the identity cues feeding cross-camera fusion).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackBox:
    """One tracked detection in one camera (Tracking->Fusion contract, docs/01)."""
    frame: int
    track_id: int
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class TrackingMetrics:
    mota: float | None = None
    idf1: float | None = None
    id_switches: int | None = None
    jersey_precision: float | None = None
    team_accuracy: float | None = None


def evaluate_tracking(pred: list[TrackBox], gt: list[TrackBox],
                      iou_threshold: float = 0.5) -> TrackingMetrics:
    """MOTA/IDF1/ID-switches for one camera. NOT YET IMPLEMENTED (Week-1 D5).

    Will wrap `motmetrics`/`trackeval` on the (pred, gt) tracklets. Signature is
    stable so the harness and `runs/` logging can be wired now.
    """
    raise NotImplementedError(
        "per-camera tracking metrics land with the tracker stage (docs/12 D5); "
        "interface is fixed -- wrap motmetrics/TrackEval here.")
