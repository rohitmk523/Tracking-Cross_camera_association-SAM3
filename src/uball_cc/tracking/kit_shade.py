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
