#!/usr/bin/env python3
"""Refit the camera calibrations WITH lens undistortion (audit P2 — the measured binding
constraint on fusion precision).

The rig is fisheye but the calibrations were plain pinhole homographies on raw pixels, so
the model can't fit all clicked points at once (~50% 'outliers') and extrapolates badly
outside each camera's clicked region. Fix: a one-parameter DIVISION model per camera
    p' = c + (p - c) / (1 + lambda * r^2),   r = |p - c| / image_diagonal
with lambda grid-fit to minimise the ALL-points median reprojection error, then H refit on
the undistorted points (no outlier escape hatch) and the calib hull recomputed.

Measured on the clickpairs (median/max cm): NL 40.8/93 -> 23.0/52, NR 31.7/58 -> 10.7/33,
FR 12.1/27 -> 8.9/38, FL 10.1/31 -> 9.0/23.

  python scripts/refit_calibration.py            # rewrites configs/calib/{FL,FR,NL,NR}.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEMO_CAL = Path("/Users/rohitkale/Cellstrat/GitHub_Repositories/DEMO_UBALL/demo/calibration")
ANGLES = ("FL", "FR", "NL", "NR")


def fit_division(px: np.ndarray, ct: np.ndarray, size: tuple[int, int]):
    """Grid-fit lambda; return (lambda, H, med_cm, max_cm) with H fit on ALL points."""
    import cv2

    cx, cy = size[0] / 2.0, size[1] / 2.0
    diag = float(np.hypot(cx, cy))

    def undist(p, lam):
        q = p - [cx, cy]
        r2 = (q ** 2).sum(1) / diag ** 2
        return q / (1 + lam * r2)[:, None] + [cx, cy]

    def err(lam):
        u = undist(px, lam)
        h, _ = cv2.findHomography(u, ct, method=0)
        if h is None:
            return None, np.inf, np.inf
        d = np.linalg.norm(cv2.perspectiveTransform(u.reshape(-1, 1, 2), h).reshape(-1, 2) - ct,
                           axis=1)
        return h, float(np.median(d)), float(d.max())

    best = (0.0, *err(0.0))
    for lam in np.linspace(-0.9, 0.9, 361):
        h, med, mx = err(float(lam))
        if med < best[2]:
            best = (float(lam), h, med, mx)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", default=str(DEMO_CAL))
    ap.add_argument("--calib-dir", default="configs/calib")
    a = ap.parse_args()
    import cv2

    for ang in ANGLES:
        src = Path(a.pairs_dir) / f"{ang}_cal_clickpairs.json"
        pairs = json.loads(src.read_text())["pairs"]
        px = np.array([p[0] for p in pairs], np.float64)
        ct = np.array([p[1] for p in pairs], np.float64)
        out_path = REPO / a.calib_dir / f"{ang}.json"
        old = json.loads(out_path.read_text())
        size = old.get("image_size", [1920, 1080])

        lam, h, med, mx = fit_division(px, ct, size)
        # undistorted points -> hull/bbox in court space (the camera's trusted region)
        cx, cy = size[0] / 2.0, size[1] / 2.0
        diag = float(np.hypot(cx, cy))
        q = px - [cx, cy]
        u = q / (1 + lam * ((q ** 2).sum(1) / diag ** 2))[:, None] + [cx, cy]
        court_pts = cv2.perspectiveTransform(u.reshape(-1, 1, 2), h).reshape(-1, 2)
        hull = cv2.convexHull(court_pts.astype(np.float32)).reshape(-1, 2)

        out = {"homography_matrix": h.tolist(),
               "image_size": size,
               "division_lambda": round(lam, 4),
               "principal_point": [cx, cy],
               "num_points": len(pairs),
               "reproj_med_cm": round(med, 1), "reproj_max_cm": round(mx, 1),
               "calib_hull": [[round(float(x), 1), round(float(y), 1)] for x, y in hull],
               "calib_court_bbox": [round(float(v), 1) for v in
                                    (court_pts[:, 0].min(), court_pts[:, 1].min(),
                                     court_pts[:, 0].max(), court_pts[:, 1].max())],
               "source": str(src), "model": "division+homography (refit_calibration.py)"}
        out_path.write_text(json.dumps(out, indent=1))
        print(f"{ang}: lambda {lam:+.2f} | all-points reproj med {med:5.1f} / max {mx:6.1f} cm "
              f"(was ~{old.get('num_points', '?')} pts, RANSAC-subset fit) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
