"""Event stream (docs/08): turn the fused world-state into a deterministic JSON of
"what happened" — the low-volume, structured substrate the VLM narrates over.

Design principle (docs/08): build state deterministically in CV; the LLM only *reasons*
over it. So this module emits discrete, timestamped events with `confidence` and a
`needs_frame_check` flag (pixels required to confirm) — never prose.

Ball-optional: possession-based events (possession, pass, turnover) need a ball court
trace; team-spatial events (transition/fast-break, possession-side) need only player
positions. With no ball, possession events are skipped and a caveat is emitted — honest
degradation rather than guessing who has the ball.
"""
from __future__ import annotations

import numpy as np

from .court import CENTER, LENGTH

# --- tunables (court cm / seconds) ---
POSS_RADIUS_CM = 180.0          # a player within this of the ball is a possession candidate
MIN_POSS_FRAMES = 8             # frames a player must hold nearest-to-ball to confirm possession
STICKY_MARGIN_CM = 70.0         # holder KEEPS possession unless a challenger is this much closer (hysteresis)
SWITCH_FRAMES = 6               # ...and stays clearly-closest this many consecutive frames -> kills jitter
PASS_MAX_GAP_FRAMES = 30        # max gap between two possessions to call it a pass (not a reset)
FASTBREAK_CM = 800.0            # team-centroid x travel to flag a transition
FASTBREAK_WINDOW_S = 2.0        # ...within this window


def _frame_index(world_state: dict) -> tuple[list[int], dict]:
    """frames sorted + {frame: {gid: {"xy": (x,y), "team": t}}}."""
    by_frame: dict[int, dict] = {}
    for fr in world_state["frames"]:
        by_frame[fr["frame"]] = {t["global_id"]: {"xy": tuple(t["court_xy"]), "team": t.get("team")}
                                 for t in fr["tracks"] if t.get("court_xy")}
    return sorted(by_frame), by_frame


def _roster_label(world_state: dict) -> dict[int, str]:
    """global_id -> human label ('Blue #7' / 'A 3') from the roster, for readable events."""
    name = {"A": "Blue", "B": "Green"}
    out = {}
    for p in world_state.get("players", []):
        team = name.get(p.get("team"), p.get("team") or "?")
        jersey = p.get("jersey")
        out[p["global_id"]] = f"{team} #{jersey}" if jersey is not None else f"{team} id{p['global_id']}"
    return out


def _possession_runs(frames, by_frame, ball_by_frame):
    """[(start_frame, end_frame, gid, team)] — STICKY nearest-player-to-ball.

    Plain nearest-player jitters when players bunch around the ball (every tiny ball
    wobble flips the 'holder'), spawning fake sub-second passes. Hysteresis fixes it: the
    current holder keeps possession unless a *challenger* is closer by STICKY_MARGIN_CM and
    stays clearly-closest for SWITCH_FRAMES consecutive frames."""
    raw = {}
    holder, holder_team = None, None
    challenger, ch_count = None, 0
    for f in frames:
        ball = ball_by_frame.get(f)
        if ball is None:
            holder, holder_team, challenger, ch_count = None, None, None, 0
            continue
        dists = {gid: float(np.hypot(p["xy"][0] - ball[0], p["xy"][1] - ball[1]))
                 for gid, p in by_frame[f].items()}
        if not dists:
            continue
        nearest = min(dists, key=dists.get)
        if dists[nearest] > POSS_RADIUS_CM:                 # ball is loose -> no possession
            holder, holder_team, challenger, ch_count = None, None, None, 0
            continue
        keep = (holder is not None and holder in dists
                and dists[holder] <= POSS_RADIUS_CM
                and dists[holder] <= dists[nearest] + STICKY_MARGIN_CM)
        if keep:
            challenger, ch_count = None, 0                  # holder stays sticky
        else:
            ch_count = ch_count + 1 if nearest == challenger else 1
            challenger = nearest
            if holder is None or ch_count >= SWITCH_FRAMES:  # confirmed takeover
                holder, holder_team = nearest, by_frame[f][nearest]["team"]
                challenger, ch_count = None, 0
        if holder is not None:                              # survives a 1-frame holder dropout
            raw[f] = (holder, holder_team)
    # collapse consecutive same-holder frames into runs, then drop runs shorter than MIN_POSS_FRAMES
    runs, cur = [], None
    for f in frames:
        h = raw.get(f)
        if h is None:
            if cur:
                runs.append(cur)
                cur = None
            continue
        if cur and cur[2] == h[0]:
            cur = (cur[0], f, h[0], h[1])
        else:
            if cur:
                runs.append(cur)
            cur = (f, f, h[0], h[1])
    if cur:
        runs.append(cur)
    return [r for r in runs if (r[1] - r[0] + 1) >= MIN_POSS_FRAMES]


def derive_events(world_state: dict, ball_by_frame: dict[int, tuple] | None = None,
                  fps: float | None = None) -> dict:
    """World-state (+ optional {frame: (court_x, court_y)} ball trace) -> event stream JSON."""
    fps = fps or world_state.get("fps", 29.97)
    frames, by_frame = _frame_index(world_state)
    label = _roster_label(world_state)
    events: list[dict] = []
    caveats: list[str] = []

    def t_of(f):
        return round((f - frames[0]) / fps, 2) if frames else 0.0

    # ---- possession-based events (need a ball trace) ----
    has_ball = bool(ball_by_frame)
    n_pass = n_to = 0
    if has_ball:
        runs = _possession_runs(frames, by_frame, ball_by_frame)
        for i, (sf, ef, gid, team) in enumerate(runs):
            events.append({"event": "possession", "t_sec": t_of(sf), "frame_window": [sf, ef],
                           "player_id": gid, "player": label.get(gid, str(gid)), "team": team,
                           "confidence": 0.6, "needs_frame_check": False})
            if i > 0:
                psf, pef, pgid, pteam = runs[i - 1]
                if sf - pef <= PASS_MAX_GAP_FRAMES and pgid != gid:
                    same = pteam == team
                    events.append({"event": "pass" if same else "turnover", "t_sec": t_of(pef),
                                   "frame": pef, "from_id": pgid, "from": label.get(pgid, str(pgid)),
                                   "to_id": gid, "to": label.get(gid, str(gid)),
                                   "from_team": pteam, "to_team": team,
                                   "confidence": 0.45, "needs_frame_check": True})
                    if same:
                        n_pass += 1
                    else:
                        n_to += 1
    else:
        caveats.append("No ball trace: possession / pass / turnover events skipped. "
                       "Ball detection on the far cameras is the blocker (Stage 4 next step).")

    # ---- team-spatial events (player positions only) ----
    teams = ("A", "B")
    centroid = {tm: {} for tm in teams}
    for f in frames:
        for tm in teams:
            xs = [p["xy"][0] for p in by_frame[f].values() if p["team"] == tm]
            if xs:
                centroid[tm][f] = float(np.mean(xs))
    win = max(1, int(FASTBREAK_WINDOW_S * fps))
    for tm in teams:
        cf = centroid[tm]
        fs = [f for f in frames if f in cf]
        for j in range(len(fs)):
            f0 = fs[j]
            f1 = next((fs[k] for k in range(j + 1, len(fs)) if fs[k] - f0 >= win), None)
            if f1 is None:
                break
            dx = cf[f1] - cf[f0]
            if abs(dx) >= FASTBREAK_CM:
                direction = "right" if dx > 0 else "left"
                events.append({"event": "transition", "t_sec": t_of(f0), "frame_window": [f0, f1],
                               "team": tm, "direction": direction, "centroid_dx_cm": round(dx),
                               "confidence": 0.5, "needs_frame_check": False})
                break  # one per team per pass of the sweep is enough for a v1

    events.sort(key=lambda e: e.get("t_sec", 0.0))
    # possession share (which side the play sat on) — proxy for control when no ball
    side = {"A_left": 0, "A_right": 0, "B_left": 0, "B_right": 0}
    for f in frames:
        for tm in teams:
            if f in centroid[tm]:
                side[f"{tm}_{'right' if centroid[tm][f] > CENTER[0] else 'left'}"] += 1
    summary = {"has_ball": has_ball, "n_events": len(events), "n_passes": n_pass,
               "n_turnovers": n_to, "n_transitions": sum(1 for e in events if e["event"] == "transition"),
               "court_length_cm": LENGTH}
    return {"fps": fps, "n_frames": len(frames), "summary": summary,
            "events": events, "caveats": " ".join(caveats)}
