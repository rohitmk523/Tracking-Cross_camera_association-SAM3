"""Manual N-point homography: image pixels -> court cm (docs/03/06).

Calibration = 4+ correspondences {court_xy_cm <-> pixel_xy} per (court, camera),
saved as JSON. We map a track's foot point (bottom-centre of box) to court coords
for the top-down view and, later, the cross-camera fusion gate (court-distance).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def compute_homography(correspondences: list[dict]) -> np.ndarray:
    """correspondences: [{"court":[x,y], "pixel":[u,v]}, ...]. Returns 3x3 H (pixel->court)."""
    import cv2  # noqa: PLC0415

    if len(correspondences) < 4:
        raise ValueError("need >= 4 correspondences for a homography")
    src = np.array([c["pixel"] for c in correspondences], dtype=np.float64)
    dst = np.array([c["court"] for c in correspondences], dtype=np.float64)
    if len(correspondences) == 4:
        return cv2.getPerspectiveTransform(
            src.astype(np.float32), dst.astype(np.float32)).astype(np.float64)
    h, _ = cv2.findHomography(src, dst, method=0)
    return h


def project(points_xy, h: np.ndarray) -> np.ndarray:
    """Map pixel points (N,2) -> court points (N,2) through H."""
    import cv2  # noqa: PLC0415

    pts = np.array(points_xy, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, h).reshape(-1, 2)


def reprojection_error(correspondences: list[dict], h: np.ndarray) -> dict:
    """How well H maps the calibration pixels back onto their court coords (cm)."""
    proj = project([c["pixel"] for c in correspondences], h)
    dst = np.array([c["court"] for c in correspondences], dtype=np.float64)
    d = np.linalg.norm(proj - dst, axis=1)
    return {"mean_cm": round(float(d.mean()), 1), "max_cm": round(float(d.max()), 1),
            "per_point_cm": [round(float(x), 1) for x in d]}


def load_calib(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def calib_hull(d: dict) -> np.ndarray | None:
    """Convex hull (court cm) of the camera's calibration points, if present. Observations
    projected outside this region are EXTRAPOLATED (un-calibrated) and unreliable -- far
    cameras only calibrate their own half of the court (docs/06)."""
    pts = d.get("calib_hull")
    return np.array(pts, dtype=np.float32) if pts else None


def in_calib_region(court_xy, hull: np.ndarray | None, pad_cm: float = 250.0) -> bool:
    """True if a court point is inside the calibrated hull (or within pad_cm of it). With
    no hull, accept (back-compat). Gates each camera to where it is actually calibrated."""
    if hull is None:
        return True
    import cv2  # noqa: PLC0415

    return cv2.pointPolygonTest(hull, (float(court_xy[0]), float(court_xy[1])), True) >= -pad_cm


def homography_from_calib(d: dict) -> np.ndarray:
    """H (pixel->court) from a calib dict: a precomputed `homography_matrix`
    (DEMO_UBALL format) or 4+ `correspondences` (manual format)."""
    if "homography_matrix" in d:
        return np.array(d["homography_matrix"], dtype=np.float64)
    return compute_homography(d["correspondences"])
