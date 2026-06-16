"""Cross-camera fusion metrics (docs/13) -- the HEADLINE, STUB.

Implemented in Week-2 once fusion emits global tracks. These quantify exactly
the DEMO_UBALL failure modes ("#1 is two people", "#25 single-angle"):

  - cross_camera_id_consistency: fraction of players that keep ONE global id
    across all cameras for the whole clip.
  - global_idf1 / global_id_switches: IDF1 at the fused level.
  - occlusion_recovery_rate: after occlusion in a view, does the global id
    persist and re-attach correctly.
  - duplicate_id_rate: two on-court dots never share a label; one (team, number)
    -> one track (uniqueness).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GlobalTrack:
    """Fusion->World-model contract (docs/01): one persistent player identity."""
    frame: int
    global_id: int
    court_xy_cm: tuple[float, float]
    team: str | None = None
    jersey: int | None = None
    members: tuple[tuple[str, int], ...] = field(default_factory=tuple)  # (cam, track_id)


@dataclass(frozen=True)
class CrossCamMetrics:
    id_consistency: float | None = None
    global_idf1: float | None = None
    global_id_switches: int | None = None
    occlusion_recovery_rate: float | None = None
    duplicate_id_rate: float | None = None


def evaluate_crosscam(pred: list[GlobalTrack], gt: list[GlobalTrack]) -> CrossCamMetrics:
    """Cross-camera ID-consistency + global IDF1. NOT YET IMPLEMENTED (Week-2).

    Interface is fixed now so fusion can be scored the moment it produces global
    tracks. `duplicate_id_rate` and `id_consistency` are computable directly from
    `pred`; IDF1/occlusion-recovery need aligned GT.
    """
    raise NotImplementedError(
        "cross-camera fusion metrics land with the fusion stage (docs/12 W2, "
        "docs/06); interface is fixed -- implement id-consistency + duplicate-id "
        "from pred, IDF1 vs GT.")
