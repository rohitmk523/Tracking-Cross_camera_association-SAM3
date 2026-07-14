#!/usr/bin/env python3
"""Layer 1 (BALL_FIRST_SPEC): smoothed per-cam ball trajectories.

Raw ball detections -> teleport rejection -> constant-acceleration Kalman +
RTS smoothing -> bounded gap interpolation (<=12 frames bracketed by
conf>=0.30 detections). Parameters follow the fusion repo's track_clean
(vertical accel noise > horizontal: gravity-aware).

Writes runs/ball_traj/{game}_{ang}_{tag}.traj.npz:
  frame_idx, cx, cy, vx, vy, imputed(0/1), conf

  .venv/bin/python scripts/ball_traj.py --game c2a354fe
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from game_meta import GAME_CHUNKS

ANGLES = ("FL", "FR", "NL", "NR")
GAP_MAX = 12
CONF_BRACKET = 0.30
TELEPORT_PX = 125.0


def kalman_rts(fs, xs, ys):
    """Constant-acceleration KF + RTS smoother over (possibly gappy) frames."""
    n = fs[-1] - fs[0] + 1
    have = {f - fs[0]: i for i, f in enumerate(fs)}
    dim = 6                      # x vx ax y vy ay
    F = np.eye(dim)
    F[0, 1] = F[1, 2] = F[3, 4] = F[4, 5] = 1.0
    F[0, 2] = F[3, 5] = 0.5
    Hm = np.zeros((2, dim)); Hm[0, 0] = Hm[1, 3] = 1.0
    Q = np.diag([0.25, 0.5, 1.5**2, 0.25, 0.5, 4.0**2])
    R = np.eye(2) * 16.0
    x = np.zeros(dim); x[0], x[3] = xs[0], ys[0]
    P = np.eye(dim) * 100.0
    xp, Pp, xf, Pf = [], [], [], []
    for k in range(n):
        x = F @ x
        P = F @ P @ F.T + Q
        xp.append(x.copy()); Pp.append(P.copy())
        if k in have:
            z = np.array([xs[have[k]], ys[have[k]]])
            S = Hm @ P @ Hm.T + R
            K = P @ Hm.T @ np.linalg.inv(S)
            x = x + K @ (z - Hm @ x)
            P = (np.eye(dim) - K @ Hm) @ P
        xf.append(x.copy()); Pf.append(P.copy())
    xs_s = [None] * n
    xs_s[-1] = xf[-1]
    for k in range(n - 2, -1, -1):
        G = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs_s[k] = xf[k] + G @ (xs_s[k + 1] - xp[k + 1])
    return np.stack(xs_s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    a = ap.parse_args()
    outd = REPO / "runs/ball_traj"
    outd.mkdir(exist_ok=True)
    for tag in GAME_CHUNKS[a.game]:
        for ang in ANGLES:
            z = np.load(REPO / f"runs/ball_cache/{a.game}_{ang}_{tag}.ball.npz")
            det = {}
            for b, sc, f, c in zip(z["boxes"], z["scores"], z["frame_idx"], z["classes"]):
                if int(c) != 0:
                    continue
                f = int(f)
                if f not in det or sc > det[f][2]:
                    det[f] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, float(sc))
            fs = sorted(det)
            if len(fs) < 10:
                np.savez_compressed(outd / f"{a.game}_{ang}_{tag}.traj.npz",
                                    frame_idx=np.array([], int), cx=np.array([]),
                                    cy=np.array([]), vx=np.array([]), vy=np.array([]),
                                    imputed=np.array([], int), conf=np.array([]))
                continue
            # teleport rejection: drop isolated dets far from BOTH neighbors
            keep = []
            for i, f in enumerate(fs):
                x, y, s = det[f]
                ok = False
                for j in (i - 1, i + 1):
                    if 0 <= j < len(fs):
                        f2 = fs[j]
                        x2, y2, _ = det[f2]
                        if np.hypot(x - x2, y - y2) <= TELEPORT_PX * max(1, abs(f - f2)):
                            ok = True
                if ok:
                    keep.append(f)
            fs = keep
            # smooth per contiguous segment (gaps > GAP_MAX split segments,
            # unless bracketed by confident dets — those get bridged)
            F_, CX, CY, VX, VY, IMP, CF = [], [], [], [], [], [], []
            seg = [fs[0]]
            for f in fs[1:]:
                gap = f - seg[-1]
                bridge = (gap <= GAP_MAX and det[seg[-1]][2] >= CONF_BRACKET
                          and det[f][2] >= CONF_BRACKET)
                if gap <= 3 or bridge:
                    seg.append(f)
                else:
                    if len(seg) >= 5:
                        xs = [det[g][0] for g in seg]
                        ys = [det[g][1] for g in seg]
                        sm = kalman_rts(seg, xs, ys)
                        for k in range(seg[-1] - seg[0] + 1):
                            f0 = seg[0] + k
                            F_.append(f0)
                            CX.append(sm[k, 0]); CY.append(sm[k, 3])
                            VX.append(sm[k, 1]); VY.append(sm[k, 4])
                            IMP.append(0 if f0 in det else 1)
                            CF.append(det[f0][2] if f0 in det else 0.4)
                    seg = [f]
            if len(seg) >= 5:
                xs = [det[g][0] for g in seg]
                ys = [det[g][1] for g in seg]
                sm = kalman_rts(seg, xs, ys)
                for k in range(seg[-1] - seg[0] + 1):
                    f0 = seg[0] + k
                    F_.append(f0)
                    CX.append(sm[k, 0]); CY.append(sm[k, 3])
                    VX.append(sm[k, 1]); VY.append(sm[k, 4])
                    IMP.append(0 if f0 in det else 1)
                    CF.append(det[f0][2] if f0 in det else 0.4)
            np.savez_compressed(outd / f"{a.game}_{ang}_{tag}.traj.npz",
                                frame_idx=np.array(F_, int), cx=np.array(CX),
                                cy=np.array(CY), vx=np.array(VX), vy=np.array(VY),
                                imputed=np.array(IMP, int), conf=np.array(CF))
        n = len(np.load(outd / f"{a.game}_FL_{tag}.traj.npz")["frame_idx"])
        print(f"  [{tag}] FL traj frames: {n}", flush=True)
    print(f"-> {outd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
