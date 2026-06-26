"""Ball tracker (fusion/ball.py) + event deriver (fusion/events.py), docs/08.

The ball detector is noisy (low-confidence true ball + scattered false positives); the
Kalman tracker must follow the real trajectory, gate the false positives, interpolate
short gaps, and re-init on a real jump (long pass). The event deriver must turn a ball
trace + world-state into possession / pass events, and degrade honestly with no ball.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
from uball_cc.fusion.ball import track_ball  # noqa: E402
from uball_cc.fusion.events import derive_events  # noqa: E402


def test_ball_tracker_gates_false_positives_and_interpolates():
    cands = {}
    for f in range(60):
        c = []
        if f not in range(20, 25):                       # 5-frame detection gap
            c.append((500 + 30 * f + (f % 2) * 5, 700, 0.12))   # noisy low-conf true ball
        c.append(((f * 137) % 2000, (f * 91) % 1200, 0.06))     # a false positive every frame
        cands[f] = c
    tr = track_ball(cands, fps=30)
    assert len(tr) >= 55                                 # near-full coverage incl. interpolation
    assert 22 in tr                                      # gap interpolated
    xs = [tr[f][0] for f in sorted(tr)]
    assert all(xs[i] <= xs[i + 1] + 80 for i in range(len(xs) - 1))   # followed the true line
    assert xs[0] < 700 and xs[-1] > 1800                 # not stuck on the random FPs


def test_ball_tracker_reinit_on_real_jump():
    """A persistent far detection (a long pass) re-acquires; one-frame blips do not."""
    cands = {}
    for f in range(40):
        if f < 20:
            cands[f] = [(500, 700, 0.3)]
        else:
            cands[f] = [(1900, 700, 0.3)]                # ball teleports (inbound) and stays
    tr = track_ball(cands, fps=30)
    assert tr[5][0] < 700                                # first location
    assert tr[39][0] > 1700                              # re-acquired at the new location


def _world(n=51):
    """Two team-A players: id7 ~x500, id10 ~x1500, both present every frame."""
    rng = np.random.default_rng(0)
    frames = []
    for f in range(n):
        tracks = [{"global_id": 7, "team": "A", "jersey": 7,
                   "court_xy": [500 + rng.normal(scale=5), 700 + rng.normal(scale=5)]},
                  {"global_id": 10, "team": "A", "jersey": 10,
                   "court_xy": [1500 + rng.normal(scale=5), 700 + rng.normal(scale=5)]}]
        frames.append({"frame": f, "tracks": tracks})
    players = [{"global_id": 7, "team": "A", "jersey": 7},
               {"global_id": 10, "team": "A", "jersey": 10}]
    return {"players": players, "frames": frames, "fps": 30}


def test_events_possession_and_pass_from_ball():
    ws = _world(51)
    ball = {}
    for f in range(0, 21):                               # ball with player 7
        ball[f] = (505, 700)
    for f in range(25, 51):                              # ...then with player 10 (pass)
        ball[f] = (1495, 700)
    out = derive_events(ws, ball_by_frame=ball, fps=30)
    kinds = [e["event"] for e in out["events"]]
    assert kinds.count("possession") == 2
    assert "pass" in kinds and "turnover" not in kinds   # same team -> pass
    poss = [e for e in out["events"] if e["event"] == "possession"]
    assert {p["player_id"] for p in poss} == {7, 10}
    assert out["summary"]["has_ball"] is True


def test_events_degrade_without_ball():
    out = derive_events(_world(40), ball_by_frame=None, fps=30)
    assert out["summary"]["has_ball"] is False
    assert not [e for e in out["events"] if e["event"] in ("possession", "pass", "turnover")]
    assert "ball" in out["caveats"].lower()
