"""Constant-velocity Kalman filter on the court plane (cm) — one per global player.

State = [px, py, vx, vy]. Measurements are court positions (cm). Multiple cameras
that see the player at one frame are fused into a single visibility-weighted update
(higher total weight => tighter measurement noise), so an occluded view simply
contributes nothing and the visible views carry the track (docs/06).
"""
from __future__ import annotations

import numpy as np


class CVKalman2D:
    def __init__(self, xy, dt: float = 1 / 30.0, q: float = 300.0, r: float = 40.0,
                 p0: float = 200.0):
        self.dt = dt
        self.x = np.array([xy[0], xy[1], 0.0, 0.0], dtype=float)
        self.P = np.diag([r * r, r * r, p0 * p0, p0 * p0]).astype(float)
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        # process noise (acceleration q cm/s^2 driving velocity, integrated to position)
        g = np.array([[0.5 * dt * dt], [0.5 * dt * dt], [dt], [dt]])
        self.Q = (g @ g.T) * (q * q)
        self.r = r

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, weight: float = 1.0) -> None:
        """Fused measurement z=(px,py); `weight` = total visibility weight (>0)."""
        r = (self.r * self.r) / max(weight, 1e-3)
        rmat = np.diag([r, r])
        y = np.asarray(z, float) - self.H @ self.x
        s = self.H @ self.P @ self.H.T + rmat
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.P = (np.eye(4) - k @ self.H) @ self.P

    @property
    def pos(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def vel(self) -> np.ndarray:
        return self.x[2:].copy()
