"""Motion-based ball candidate detection (docs/08): find the small/fast ball by its MOTION.

The hard truth (verified): appearance detection finds the ball only near the basket, where
it's big. Mid-court it's too small/blurred for a per-frame detector — and even for a human
annotator — so more appearance training data can't fix it. But the ball MOVES fast against a
near-static court, so 3-frame differencing reveals it exactly where appearance fails. On an
e6 fast break this surfaced a ball candidate in 86% of frames spanning the full court width,
where SAHI appearance detection pinned to one spot.

These are IMAGE-space candidates; the caller projects them to court (per-camera homography)
and feeds them to the cross-camera ball fusion + Kalman tracker (fusion/ball.py). Noisy by
design (some land on moving jerseys) — the trajectory gate + stationarity filter clean it.
"""
from __future__ import annotations

import cv2
import numpy as np

MOTION_THRESH = 28              # per-pixel 3-frame motion to count as "moving"
MIN_AREA, MAX_AREA = 6, 250     # ball-sized blob (px area) at 1920x1080
ROUND_LO, ROUND_HI = 0.4, 2.5   # w/h aspect — roughly round
MIN_ORANGE = 0.12               # fraction of the patch that is basketball-orange


def _orange_frac(hsv: np.ndarray, x: int, y: int, r: int = 8) -> float:
    p = hsv[max(0, y - r):y + r, max(0, x - r):x + r]
    if p.size == 0:
        return 0.0
    return float(((p[:, :, 0] < 22) & (p[:, :, 1] > 80) & (p[:, :, 2] > 80)).mean())


def motion_candidates(frames: list[np.ndarray]) -> dict[int, list[tuple]]:
    """[BGR frames] -> {frame_idx: [(x, y, orange_score)]} ball candidates from 3-frame motion.

    A pixel counts as moving only if it changed in BOTH the previous AND next gap (min of the
    two abs-diffs) — kills single-frame noise/flicker, keeps the consistently-moving ball."""
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    hsv = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in frames]
    kernel = np.ones((2, 2), np.uint8)
    out: dict[int, list[tuple]] = {}
    for t in range(1, len(frames) - 1):
        motion = np.minimum(cv2.absdiff(gray[t], gray[t - 1]), cv2.absdiff(gray[t + 1], gray[t]))
        _, mask = cv2.threshold(motion, MOTION_THRESH, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
        cands = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if MIN_AREA <= area <= MAX_AREA and ROUND_LO <= w / max(h, 1) <= ROUND_HI:
                x, y = int(cent[i][0]), int(cent[i][1])
                o = _orange_frac(hsv[t], x, y)
                if o > MIN_ORANGE:
                    cands.append((x, y, o))
        if cands:
            out[t] = cands
    return out
