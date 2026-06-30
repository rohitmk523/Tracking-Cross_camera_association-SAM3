"""Temporal ball tracker (docs/08): turn noisy per-frame ball detections into a clean,
gap-filled court trace for possession.

The ball detector fires on the ball in most frames but at low confidence, with false
positives scattered elsewhere (small-object signature). A constant-velocity Kalman filter
on the court plane fixes this: it **gates** measurements (a detection far from the
predicted ball is rejected as a false positive), **interpolates** short gaps with the
prediction, and **re-initialises** when strong detections persist far from the track (a
real long pass / inbound). Input is *all* candidates per frame so the gate — not the raw
score — decides which detection is the ball.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .kalman import CVKalman2D

GATE_CM = 250.0          # a detection within this of the prediction is a candidate ball
MAX_COAST = 12           # frames to interpolate with the prediction before declaring the ball lost
REINIT_SCORE = 0.2       # a detection this strong, far from the track, can seed a re-init
REINIT_FRAMES = 3        # ...if it persists this many frames (a real pass/inbound, not a blip)
STATIONARY_CELL_CM = 80.0    # fine court grid: a fixed FP lands in one cell, a moving ball spreads
STATIONARY_MAX_OCC = 0.35    # reject a cell whose detections span > this fraction of the clip's frames


def reject_stationary(candidates_by_frame: dict[int, list[tuple]]) -> tuple[dict, list]:
    """Drop candidates fixed at one court spot across most of the clip — a real ball MOVES;
    a persistent same-cell detection is a fixed false positive (logo / marking / ball-coloured
    object). Returns (filtered_candidates, [banned_cell_centers]). Distinguishes a fixed FP from
    a genuinely-held ball: a held ball still passes/dribbles across cells, so no single fine cell
    is occupied in >STATIONARY_MAX_OCC of frames; a rock-steady FP concentrates in one cell."""
    if not candidates_by_frame:
        return candidates_by_frame, []
    span = max(1, max(candidates_by_frame) - min(candidates_by_frame) + 1)
    cell_frames: dict[tuple, set] = defaultdict(set)
    for f, cands in candidates_by_frame.items():
        for (x, y, _s) in cands:
            cell_frames[(round(x / STATIONARY_CELL_CM), round(y / STATIONARY_CELL_CM))].add(f)
    banned = {c for c, fr in cell_frames.items() if len(fr) / span > STATIONARY_MAX_OCC}
    if not banned:
        return candidates_by_frame, []
    out: dict[int, list] = {}
    for f, cands in candidates_by_frame.items():
        kept = [c for c in cands
                if (round(c[0] / STATIONARY_CELL_CM), round(c[1] / STATIONARY_CELL_CM)) not in banned]
        if kept:
            out[f] = kept
    centers = [(round(cx * STATIONARY_CELL_CM), round(cy * STATIONARY_CELL_CM)) for cx, cy in banned]
    return out, centers


def _best_in_gate(cands, pred, gate_cm=GATE_CM):
    ing = [(c, float(np.hypot(c[0] - pred[0], c[1] - pred[1]))) for c in cands]
    ing = [(c, d) for c, d in ing if d <= gate_cm]
    return min(ing, key=lambda cd: cd[1])[0] if ing else None


def track_ball(candidates_by_frame: dict[int, list[tuple]], fps: float = 29.97, *,
               gate_cm: float = GATE_CM, max_coast: int = MAX_COAST,
               reinit_score: float = REINIT_SCORE, reinit_frames: int = REINIT_FRAMES) -> dict[int, list]:
    """{frame: [(court_x, court_y, score), ...]} (court cm) -> {frame: [x, y]} clean trace.

    A frame appears in the output only while the ball is actively tracked (real detection
    in-gate, or a short coast). Long gaps are left empty rather than hallucinated. For the
    multi-camera fused pipeline pass a higher `reinit_score` so a re-acquire requires
    CROSS-CAMERA AGREEMENT, not a single-camera blip (prevents end-of-clip teleport glitches)."""
    if not candidates_by_frame:
        return {}
    f0, f1 = min(candidates_by_frame), max(candidates_by_frame)
    kf: CVKalman2D | None = None
    coast = 0
    strong_run: list = []
    trace: dict[int, list] = {}
    for f in range(f0, f1 + 1):
        cands = candidates_by_frame.get(f, [])
        if kf is None:                                   # (re)acquire on the strongest candidate
            if cands:
                seed = max(cands, key=lambda c: c[2])
                kf = CVKalman2D((seed[0], seed[1]), dt=1.0 / fps)
                trace[f] = [round(seed[0], 1), round(seed[1], 1)]
                coast, strong_run = 0, []
            continue
        kf.predict()
        pred = kf.pos
        pick = _best_in_gate(cands, pred, gate_cm)
        if pick is not None:
            kf.update((pick[0], pick[1]), weight=float(np.clip(pick[2] * 3.0, 0.3, 1.5)))
            p = kf.pos
            trace[f] = [round(float(p[0]), 1), round(float(p[1]), 1)]
            coast, strong_run = 0, []
            continue
        # no in-gate detection: coast, and watch for a persistent far detection (real jump)
        coast += 1
        if coast <= max_coast:
            trace[f] = [round(float(pred[0]), 1), round(float(pred[1]), 1)]
        # a re-acquire needs the strong candidates CLUSTERED (a real ball at a new spot), not
        # scattered noise that merely happens to be strong on different frames.
        strong = [c for c in cands if c[2] >= reinit_score]
        if strong and (not strong_run or
                       np.hypot(strong_run[-1][0] - max(strong, key=lambda c: c[2])[0],
                                strong_run[-1][1] - max(strong, key=lambda c: c[2])[1]) <= gate_cm):
            strong_run.append(max(strong, key=lambda c: c[2]))
            if len(strong_run) >= reinit_frames:
                s = strong_run[-1]
                kf = CVKalman2D((s[0], s[1]), dt=1.0 / fps)
                trace[f] = [round(s[0], 1), round(s[1], 1)]
                coast, strong_run = 0, []
        else:
            strong_run = [max(strong, key=lambda c: c[2])] if strong else []
        if coast > max_coast and not strong_run:
            kf = None                                    # lost; re-acquire on the next detection
            coast = 0
    return trace
