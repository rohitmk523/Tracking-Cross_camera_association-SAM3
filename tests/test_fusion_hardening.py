"""2026-07-02 audit hardening: the specific measured defects, pinned by tests.

Each test encodes one audited failure scenario: axis-coupled Kalman noise (D8),
player<->referee class flicker (D2), sticky attribute counters (D10), silent track
deletion on a (team,number) collision (D11 / jersey blast-radius), and re-entry
identity theft through the never-failing ReID gate (D5).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
from uball_cc.fusion.engine import FusionEngine, GlobalTrack, Observation  # noqa: E402
from uball_cc.fusion.kalman import CVKalman2D  # noqa: E402
from uball_cc.tracking.teams import majority_class  # noqa: E402
from uball_cc.tracking.types import Track  # noqa: E402


def test_kalman_q_axes_decoupled():
    """Process noise must be block-diagonal per axis — the old rank-1 g@g^T coupled x/y."""
    q = CVKalman2D((0, 0)).Q
    assert q[0, 2] > 0 and q[1, 3] > 0                     # position<->velocity within an axis
    for i, j in ((0, 1), (0, 3), (1, 2), (2, 3)):          # never across axes
        assert q[i, j] == 0 and q[j, i] == 0


def _trk(tid: int, frame: int, cls: int) -> Track:
    return Track(cam="NR", frame=frame, track_id=tid, box_xyxy=(0, 0, 10, 20),
                 score=0.9, class_id=cls)


def test_majority_class_stops_ref_flicker():
    """A player track with a few referee-class frames is a PLAYER on every frame."""
    tracks = [_trk(1, f, 0) for f in range(9)] + [_trk(1, 9, 1), _trk(1, 10, 1)] \
        + [_trk(2, f, 1) for f in range(5)] + [_trk(2, 5, 0)]
    maj = majority_class(tracks)
    assert maj[1] == 0 and maj[2] == 1


def test_attr_decay_lets_recent_evidence_win():
    """With decay, a mislabeled start flips to the sustained recent team; without it,
    the early majority persists much longer (the audit's 'counters never decay')."""
    def flip_after(decay: float) -> int:
        t = GlobalTrack(1, attr_decay=decay)
        t.init([Observation("NL", 1, (0.0, 0.0), team="A")], 0)
        for f in range(1, 5):
            t.update([Observation("NL", 1, (0.0, 0.0), team="A")], f)
        for n in range(1, 30):
            t.update([Observation("NL", 1, (0.0, 0.0), team="B")], 4 + n)
            if t.team == "B":
                return n
        return 99
    assert flip_after(0.7) <= 3                             # decayed: recent B evidence wins fast
    assert flip_after(1.0) >= 5                             # sticky: must out-count all history


def _run(eng: FusionEngine, frames: list[list[Observation]]) -> list:
    live = []
    for f, obs in enumerate(frames):
        live = eng.step(f, obs)
    return live


def test_enforce_unique_strips_number_but_keeps_track():
    """Duplicate committed (team, number): the weaker claimant loses the NUMBER, not its life."""
    eng = FusionEngine(min_hits=1, jersey_min_votes=2)
    frames = [[Observation("NL", 1, (500.0, 500.0), team="A", jersey=7),
               Observation("NR", 2, (2200.0, 900.0), team="A", jersey=7)] for _ in range(6)]
    live = _run(eng, frames)
    assert len(live) == 2, "a live identity was deleted on a number collision"
    assert [t.jersey for t in live].count(7) == 1          # exactly one keeps the number


def test_none_team_duplicates_are_exempt():
    """(None, 7) on two tracks proves nothing about identity — both keep their number."""
    eng = FusionEngine(min_hits=1, jersey_min_votes=2)
    frames = [[Observation("NL", 1, (500.0, 500.0), jersey=7),
               Observation("NR", 2, (2200.0, 900.0), jersey=7)] for _ in range(6)]
    live = _run(eng, frames)
    assert [t.jersey for t in live].count(7) == 2


def _reentry_scenario(returning_team: str) -> tuple[set, set]:
    """A team-A player tracks, disappears past the lost buffer, then someone reappears
    nearby with an IDENTICAL reid (the audit's camera-biased-ReID scenario). Returns
    (ids_before, ids_after)."""
    eng = FusionEngine(min_hits=1, lost_buffer=2, reentry_frames=100)
    reid = np.ones(16)
    before = set()
    for f in range(5):
        before |= {t.id for t in eng.step(f, [Observation("NL", 1, (500.0, 500.0),
                                                          team="A", reid=reid)])}
    for f in range(5, 12):
        eng.step(f, [])                                     # gone past lost_buffer
    after = {t.id for t in eng.step(12, [Observation("NL", 9, (520.0, 510.0),
                                                     team=returning_team, reid=reid)])}
    return before, after


def test_reentry_rejects_team_mismatch():
    before, after = _reentry_scenario("B")                  # opponent walks into the same spot
    assert not (before & after), "identity stolen across teams through the reid gate"


def test_reentry_accepts_true_return():
    before, after = _reentry_scenario("A")                  # the same player comes back
    assert before & after, "true re-entry was refused"
