"""Per-camera tracker smoke (docs/05). Dependency-free: DummyDetector + ByteTrack.

A single static box every frame must collapse to ONE stable local id, with the
track confirmed after the `minimum_consecutive_frames` debounce.
"""
from __future__ import annotations

import numpy as np
import pytest

from uball_cc.detection.base import Detection, DummyDetector
from uball_cc.tracking import ByteTrackTracker, Track, summarize, track_sequence

pytest.importorskip("supervision")             # Detections container (MIT)
pytest.importorskip("trackers")                # Roboflow ByteTrackTracker (MIT)


def _frames(n: int, w: int = 640, h: int = 360):
    for _ in range(n):
        yield np.zeros((h, w, 3), dtype=np.uint8)


def test_stable_single_track():
    tracks = track_sequence(DummyDetector(), _frames(40), cam="FL",
                            minimum_consecutive_frames=3)
    assert tracks, "expected some confirmed tracks"
    ids = {t.track_id for t in tracks}
    assert len(ids) == 1, f"static box should yield ONE id, got {ids}"
    assert all(t.cam == "FL" and t.class_id == 0 for t in tracks)
    # confirmed after the debounce, so fewer than all 40 frames produce a track
    assert len(tracks) <= 40


def test_foot_point_is_bottom_centre():
    t = Track("NR", 0, 1, (100.0, 40.0, 140.0, 240.0), 0.9, 0)
    assert t.foot_xy == (120.0, 240.0)
    assert t.class_name == "player"


def test_ball_is_not_tracked_by_default():
    tr = ByteTrackTracker("FL", minimum_consecutive_frames=1)
    ball = [Detection((10.0, 10.0, 20.0, 20.0), 0.9, 2)]   # class 2 = ball
    assert tr.update(ball, frame=0) == []                   # excluded from tracklets


def test_summarize_shape():
    s = summarize([Track("FL", 0, 1, (0, 0, 10, 10), 0.5, 0),
                   Track("FL", 1, 1, (0, 0, 10, 10), 0.5, 0)])
    assert s["n_unique_track_ids"] == 1 and s["n_track_observations"] == 2


def test_record_roundtrip_with_team():
    t = Track("FR", 5, 9, (1.0, 2.0, 3.0, 4.0), 0.81, 0, team="A", jersey=7)
    back = Track.from_record(t.to_record())
    assert back.cam == "FR" and back.track_id == 9 and back.class_id == 0
    assert back.team == "A" and back.jersey == 7
    assert back.box_xyxy == (1.0, 2.0, 3.0, 4.0)
