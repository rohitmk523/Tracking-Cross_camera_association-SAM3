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

import numpy as np

from .kalman import CVKalman2D

GATE_CM = 250.0          # a detection within this of the prediction is a candidate ball
MAX_COAST = 12           # frames to interpolate with the prediction before declaring the ball lost
REINIT_SCORE = 0.2       # a detection this strong, far from the track, can seed a re-init
REINIT_FRAMES = 3        # ...if it persists this many frames (a real pass/inbound, not a blip)


def _best_in_gate(cands, pred):
    ing = [(c, float(np.hypot(c[0] - pred[0], c[1] - pred[1]))) for c in cands]
    ing = [(c, d) for c, d in ing if d <= GATE_CM]
    return min(ing, key=lambda cd: cd[1])[0] if ing else None


def track_ball(candidates_by_frame: dict[int, list[tuple]], fps: float = 29.97) -> dict[int, list]:
    """{frame: [(court_x, court_y, score), ...]} (court cm) -> {frame: [x, y]} clean trace.

    A frame appears in the output only while the ball is actively tracked (real detection
    in-gate, or a short coast). Long gaps are left empty rather than hallucinated."""
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
        pick = _best_in_gate(cands, pred)
        if pick is not None:
            kf.update((pick[0], pick[1]), weight=float(np.clip(pick[2] * 3.0, 0.3, 1.5)))
            p = kf.pos
            trace[f] = [round(float(p[0]), 1), round(float(p[1]), 1)]
            coast, strong_run = 0, []
            continue
        # no in-gate detection: coast, and watch for a persistent far detection (real jump)
        coast += 1
        if coast <= MAX_COAST:
            trace[f] = [round(float(pred[0]), 1), round(float(pred[1]), 1)]
        strong = [c for c in cands if c[2] >= REINIT_SCORE]
        if strong:
            strong_run.append(max(strong, key=lambda c: c[2]))
            if len(strong_run) >= REINIT_FRAMES:
                s = strong_run[-1]
                kf = CVKalman2D((s[0], s[1]), dt=1.0 / fps)
                trace[f] = [round(s[0], 1), round(s[1], 1)]
                coast, strong_run = 0, []
        else:
            strong_run = []
        if coast > MAX_COAST and not strong_run:
            kf = None                                    # lost; re-acquire on the next detection
            coast = 0
    return trace
