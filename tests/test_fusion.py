"""Cross-camera fusion engine (docs/06) on synthetic multi-camera scenes (known GT).

We simulate players on the court seen by 3 cameras with projection noise, partial
jersey legibility, occlusion, and per-camera id churn, then assert the engine gives
ONE stable global id per player, keeps same-team neighbours separate (jersey
authority), and re-attaches a player after a long full occlusion (re-entry).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

pytest.importorskip("scipy")
from uball_cc.fusion.engine import FusionEngine, Observation  # noqa: E402

CAMS = ("FL", "NL", "NR")
META = {0: ("A", 7), 1: ("A", 10), 2: ("B", 5), 3: ("B", 23)}      # pid -> (team, jersey)
BASES = {0: (300, 300), 1: (1800, 400), 2: (600, 1200), 3: (1500, 1000)}


def _gt(pid, f):
    bx, by = BASES[pid]
    return (bx + 3.0 * f, by + 2.0 * f * (1 if pid % 2 == 0 else -1))


def _observe(eng, n_frames, *, occ_pid=None, occ_range=(), rng=None, reids=None):
    rng = rng or np.random.default_rng(1)
    reids = reids or {pid: rng.normal(size=64) for pid in META}
    hist = {pid: [] for pid in META}
    for f in range(n_frames):
        obs = []
        for pid, (team, jersey) in META.items():
            if pid == occ_pid and f in occ_range:
                continue
            gx, gy = _gt(pid, f)
            for cam in CAMS:
                nx, ny = rng.normal(scale=25, size=2)
                jersey_seen = jersey if rng.random() < 0.5 else None        # legible minority
                obs.append(Observation(cam, pid * 10 + f % 3, (gx + nx, gy + ny),
                                       team=team, jersey=jersey_seen,
                                       reid=reids[pid] + rng.normal(scale=0.1, size=64)))
        for t in eng.step(f, obs):
            nearest = min(META, key=lambda p: np.linalg.norm(t.pos - np.array(_gt(p, f))))
            hist[nearest].append(t.id)
    return hist


def test_one_global_id_per_player():
    hist = _observe(FusionEngine(jersey_min_votes=2, min_hits=2), 50)
    dominant = {}
    for pid, ids in hist.items():
        assert ids, f"player {pid} never tracked"
        top, cnt = Counter(ids).most_common(1)[0]
        assert cnt / len(ids) > 0.9, f"player {pid} id unstable: {Counter(ids)}"
        dominant[pid] = top
    assert len(set(dominant.values())) == 4, f"players collapsed to {dominant}"


def test_same_team_neighbours_stay_separate():
    """Two team-A players a court-step apart, different numbers -> never merged."""
    eng = FusionEngine(jersey_min_votes=2, min_hits=2)
    rng = np.random.default_rng(3)
    r7, r10 = rng.normal(size=64), rng.normal(size=64)
    seen_ids = set()
    for f in range(30):
        obs = []
        for (jersey, reid, off) in ((7, r7, 0.0), (10, r10, 80.0)):     # 80cm apart, same team A
            for cam in CAMS:
                xy = (1000 + off + rng.normal(scale=15), 800 + rng.normal(scale=15))
                obs.append(Observation(cam, jersey, xy, team="A", jersey=jersey,
                                       reid=reid + rng.normal(scale=0.1, size=64)))
        live = eng.step(f, obs)
        if f > 5:
            seen_ids.update(t.id for t in live)
            assert len({t.jersey for t in live if t.jersey}) == 2  # both numbers present, distinct
    assert len(seen_ids) == 2, f"same-team neighbours merged/duplicated: {seen_ids}"


def test_reentry_after_long_occlusion():
    """Player fully occluded past the lost-buffer must revive with the SAME id."""
    eng = FusionEngine(jersey_min_votes=2, min_hits=2, lost_buffer=5, reentry_frames=200)
    hist = _observe(eng, 70, occ_pid=2, occ_range=set(range(20, 45)))   # 25-frame blackout > buffer
    ids2 = hist[2]
    before = [i for i in ids2[:18]]
    after = [i for i in ids2[-15:]]
    assert before and after, "player 2 not tracked before/after occlusion"
    assert Counter(before).most_common(1)[0][0] == Counter(after).most_common(1)[0][0], \
        f"re-entry assigned a new id: before={Counter(before)} after={Counter(after)}"
