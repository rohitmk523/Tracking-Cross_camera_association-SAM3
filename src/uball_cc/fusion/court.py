"""Court model (docs/03/08) — court_2.dxf geometry in CENTIMETRES, corner-origin.

Matches the DEMO_UBALL calibration convention (the court-rectangle corner shifted to
(0,0)), so the per-angle image->court homographies in DEMO_UBALL/demo/calibration drop
straight in. Playing court = 2143.7 x 1426.4 cm.

Axes: x = LENGTH (baseline..baseline, 0..2143.7), y = WIDTH (sideline..sideline, 0..1426.4).
"""
from __future__ import annotations

import numpy as np

LENGTH, WIDTH = 2143.7, 1426.4
X0, X1, Y0, Y1 = 0.0, LENGTH, 0.0, WIDTH
CENTER = (LENGTH / 2.0, WIDTH / 2.0)                       # (1071.85, 713.2)
CENTER_CIRCLE_R, CENTER_DOT_R, FT_CIRCLE_R, THREE_PT_R = 182.9, 61.0, 182.9, 670.6
LANE_HALF, FT_DISTANCE, HOOP_INSET = 194.3, 594.3, 107.7
LEFT_BASKET, RIGHT_BASKET = (HOOP_INSET, CENTER[1]), (LENGTH - HOOP_INSET, CENTER[1])
LEFT_FT_CENTER, RIGHT_FT_CENTER = (FT_DISTANCE, CENTER[1]), (LENGTH - FT_DISTANCE, CENTER[1])
LEFT_KEY = (0.0, CENTER[1] - LANE_HALF, FT_DISTANCE, CENTER[1] + LANE_HALF)
RIGHT_KEY = (LENGTH - FT_DISTANCE, CENTER[1] - LANE_HALF, LENGTH, CENTER[1] + LANE_HALF)

LANDMARKS: dict[str, tuple[float, float]] = {
    "L_baseline_top": (0.0, 0.0), "L_baseline_bot": (0.0, WIDTH),
    "R_baseline_top": (LENGTH, 0.0), "R_baseline_bot": (LENGTH, WIDTH),
    "center": CENTER, "center_top": (CENTER[0], 0.0), "center_bot": (CENTER[0], WIDTH),
    "L_ft_center": LEFT_FT_CENTER, "R_ft_center": RIGHT_FT_CENTER,
    "L_basket": LEFT_BASKET, "R_basket": RIGHT_BASKET,
}


def make_to_px(scale: float = 0.34, margin: int = 26):
    """Return (canvas_hw, to_px). to_px(court_xy)->(col,row) for the top-down image."""
    w = int(LENGTH * scale) + 2 * margin
    h = int(WIDTH * scale) + 2 * margin

    def to_px(xy):
        return (int(xy[0] * scale) + margin, int(xy[1] * scale) + margin)

    return (h, w), to_px


def draw_court(scale: float = 0.34, margin: int = 26):
    """Render the top-down court (lines/circles/keys/arcs). Returns (img_bgr, to_px)."""
    import cv2  # noqa: PLC0415

    (h, w), to_px = make_to_px(scale, margin)
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    line = (180, 180, 180)
    rad = lambda v: int(round(v * scale))  # noqa: E731

    cv2.rectangle(img, to_px((X0, Y0)), to_px((X1, Y1)), line, 2)
    cv2.line(img, to_px((CENTER[0], Y0)), to_px((CENTER[0], Y1)), line, 1)
    cv2.circle(img, to_px(CENTER), rad(CENTER_CIRCLE_R), line, 1)
    for kx0, ky0, kx1, ky1 in (LEFT_KEY, RIGHT_KEY):
        cv2.rectangle(img, to_px((kx0, ky0)), to_px((kx1, ky1)), line, 1)
    for c in (LEFT_FT_CENTER, RIGHT_FT_CENTER):
        cv2.circle(img, to_px(c), rad(FT_CIRCLE_R), line, 1)
    for b in (LEFT_BASKET, RIGHT_BASKET):
        cv2.circle(img, to_px(b), max(2, rad(22.5)), (0, 140, 255), -1)
    cv2.ellipse(img, to_px(LEFT_BASKET), (rad(THREE_PT_R), rad(THREE_PT_R)), 0, -70, 70, line, 1)
    cv2.ellipse(img, to_px(RIGHT_BASKET), (rad(THREE_PT_R), rad(THREE_PT_R)), 0, 110, 250, line, 1)
    return img, to_px
