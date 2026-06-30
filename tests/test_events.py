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
from uball_cc.fusion.ball import reject_stationary, track_ball  # noqa: E402
from uball_cc.fusion.events import derive_events  # noqa: E402


def test_fuse_ball_candidates_rewards_cross_camera_agreement():
    """The ball seen by 2 cameras at one court point outscores a lone single-camera FP."""
    from uball_cc.fusion.ball_fuse import fuse_ball_candidates
    per_cam = {
        "NL": {5: [(1000.0, 700.0, 0.3)]},          # ball seen by NL ...
        "NR": {5: [(1010.0, 690.0, 0.3)]},          # ... and NR at ~the same court point
        "FL": {5: [(200.0, 200.0, 0.3)]},           # a lone false positive elsewhere
    }
    fused = fuse_ball_candidates(per_cam)
    top = max(fused[5], key=lambda c: c[2])
    assert abs(top[0] - 1005) < 50                  # the agreed ball location is the top candidate
    assert top[2] > 0.3                             # ...with an agreement-boosted score (> a single cam)


def test_motion_candidates_finds_moving_orange_ball():
    """Motion detection surfaces a small fast orange ball that moves against a static court."""
    import cv2
    from uball_cc.fusion.ball_motion import motion_candidates
    frames = []
    for i in range(6):
        img = np.full((200, 400, 3), 100, np.uint8)        # static gray court
        cv2.circle(img, (60 + i * 35, 100), 5, (20, 120, 240), -1)   # orange ball moving right
        frames.append(img)
    cands = motion_candidates(frames)
    assert len(cands) >= 3                                  # detected in the interior frames
    xs = [cands[t][0][0] for t in sorted(cands)]
    assert xs[-1] > xs[0]                                   # candidate tracks the rightward motion


def test_reject_stationary_drops_fixed_fp_keeps_moving_ball():
    """A fixed false positive (same court spot every frame) is removed; the moving ball kept."""
    cands = {}
    for f in range(60):
        cands[f] = [(900.0, 600.0, 0.10),                 # fixed FP — same cell all clip
                    (500.0 + 25 * f, 700.0, 0.10)]        # real ball moving across the court
    filt, banned = reject_stationary(cands)
    assert len(banned) == 1                                # exactly the FP cell flagged
    flat = [c for v in filt.values() for c in v]
    assert all(abs(c[0] - 900.0) > 80 or abs(c[1] - 600.0) > 80 for c in flat)  # no FP survives
    assert any(c[0] > 1500 for c in flat)                  # late moving-ball positions kept


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
