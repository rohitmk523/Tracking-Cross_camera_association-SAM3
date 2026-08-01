"""Kit shade (B dark / W bright) from a player crop — pose-guided torso brightness.

Shared by the anchor extractor (inline shading, Phase-1 optimization) and
annotate_anchor_kits (clustering + the legacy re-decode path).
"""
from __future__ import annotations

import cv2
import numpy as np


def jersey_shade(crop, kpts, kscores) -> float | None:
    """Median V (HSV) of the torso region. kpts are crop-local pixel coords."""
    sh = [i for i in (5, 6) if kscores[i] >= 0.3]
    hp = [i for i in (11, 12) if kscores[i] >= 0.3]
    h, w = crop.shape[:2]
    if sh and hp:
        y1 = int(max(0, min(kpts[i][1] for i in sh)))
        y2 = int(min(h, max(kpts[i][1] for i in hp)))
        x1 = int(max(0, min(kpts[i][0] for i in sh + hp) - 5))
        x2 = int(min(w, max(kpts[i][0] for i in sh + hp) + 5))
    else:
        y1, y2, x1, x2 = int(0.2 * h), int(0.5 * h), int(0.25 * w), int(0.75 * w)
    if y2 <= y1 or x2 <= x1:
        return None
    hsv = cv2.cvtColor(crop[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    return float(np.median(hsv[:, :, 2]))


def jersey_shade_hv(crop, kpts, kscores) -> tuple[float, float] | None:
    """(median V, median H) of the torso region — H separates colored kits
    (blue vs green) that brightness alone cannot."""
    sh = [i for i in (5, 6) if kscores[i] >= 0.3]
    hp = [i for i in (11, 12) if kscores[i] >= 0.3]
    h, w = crop.shape[:2]
    if sh and hp:
        y1 = int(max(0, min(kpts[i][1] for i in sh)))
        y2 = int(min(h, max(kpts[i][1] for i in hp)))
        x1 = int(max(0, min(kpts[i][0] for i in sh + hp) - 5))
        x2 = int(min(w, max(kpts[i][0] for i in sh + hp) + 5))
    else:
        y1, y2, x1, x2 = int(0.2 * h), int(0.5 * h), int(0.25 * w), int(0.75 * w)
    if y2 <= y1 or x2 <= x1:
        return None
    hsv = cv2.cvtColor(crop[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    return float(np.median(hsv[:, :, 2])), float(np.median(hsv[:, :, 0]))


def jersey_shade_hv_bbox(crop) -> tuple[float, float] | None:
    """(median V, median H) from a GEOMETRIC torso region — pose-free fallback
    for the speed bundle: upper-middle of the person box (y 18-55%, x middle
    60%). Noisier than pose-guided but keeps kit tags alive without skeletons."""
    h, w = crop.shape[:2]
    if h < 24 or w < 12:
        return None
    y1, y2 = int(0.18 * h), int(0.55 * h)
    x1, x2 = int(0.20 * w), int(0.80 * w)
    if y2 - y1 < 6 or x2 - x1 < 6:
        return None
    hsv = cv2.cvtColor(crop[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    return float(np.median(hsv[:, :, 2])), float(np.median(hsv[:, :, 0]))
