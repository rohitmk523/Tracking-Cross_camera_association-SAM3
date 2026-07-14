#!/usr/bin/env python3
"""Camera sync offsets from BALL-CACHE velocity spikes (post-detection sync).

Passes, shots and rim hits produce ball-speed spikes at the same real instant
in every camera that sees the ball. Per camera we build a per-frame |ball
velocity| signal from the specialist ball cache (0 where absent), clip it,
and cross-correlate against FL per chunk. Chunks vote; the mode wins.

Validated against games with known offsets before use on new games.

  .venv/bin/python scripts/sync_ball_offsets.py --game 13e1ffad
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


def speed_signal(game: str, ang: str, tag: str) -> np.ndarray | None:
    p = REPO / f"runs/ball_cache/{game}_{ang}_{tag}.ball.npz"
    if not p.exists():
        return None
    z = np.load(p)
    cls = z["classes"] if "classes" in z else np.zeros(len(z["scores"]))
    best: dict[int, tuple[float, float, float]] = {}
    for b, s, f, c in zip(z["boxes"], z["scores"], z["frame_idx"], cls):
        if int(c) != 0:
            continue
        f = int(f)
        if f not in best or s > best[f][2]:
            best[f] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, float(s))
    if len(best) < 500:
        return None
    n = max(best) + 1
    sig = np.zeros(n)
    fs = sorted(best)
    for f0, f1 in zip(fs, fs[1:]):
        if f1 - f0 <= 3:
            x0, y0, _ = best[f0]
            x1, y1, _ = best[f1]
            v = float(np.hypot(x1 - x0, y1 - y0)) / (f1 - f0)
            sig[f1] = min(v, 60.0)          # clip teleports/misdetections
    s = sig - sig.mean()
    sd = s.std()
    return s / sd if sd > 0 else None


def xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 45):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    nfft = 1 << int(np.ceil(np.log2(2 * n - 1)))
    C = np.fft.irfft(np.conj(np.fft.rfft(a, nfft)) * np.fft.rfft(b, nfft), nfft)
    lags = np.arange(-max_lag, max_lag + 1)
    c = C[lags % nfft] / (n - np.abs(lags))
    j = int(np.argmax(c))
    away = np.abs(lags - lags[j]) > 5
    second = c[away].max() if away.any() else 0.0
    ratio = float(c[j] / abs(second)) if second else float("inf")
    return int(lags[j]), ratio


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--min-ratio", type=float, default=1.3)
    a = ap.parse_args()
    chunks = GAME_CHUNKS[a.game]

    sig = {}
    for ang in ANGLES:
        for tag in chunks:
            s = speed_signal(a.game, ang, tag)
            if s is not None:
                sig[(ang, tag)] = s

    print(f"{a.game}: offsets vs FL (frames, ball-velocity xcorr):")
    result = {"FL": 0}
    for ang in ("FR", "NL", "NR"):
        cands = []
        for tag in chunks:
            ref, oth = sig.get(("FL", tag)), sig.get((ang, tag))
            if ref is None or oth is None:
                continue
            k, ratio = xcorr_lag(ref, oth)
            keep = ratio >= a.min_ratio
            cands.append((k, ratio, keep))
            print(f"  {ang} {tag:>9}: {k:+4d} frames (ratio {ratio:4.2f})"
                  f"{'' if keep else '  [weak]'}")
        good = [c[0] for c in cands if c[2]]
        if len(good) >= 3:
            vals, counts = np.unique(good, return_counts=True)
            mode = int(vals[np.argmax(counts)])
            agree = int(counts.max())
            spread = int(np.max(good) - np.min(good))
            result[ang] = mode
            print(f"  {ang} -> {mode:+d} frames (mode {agree}/{len(good)}, "
                  f"spread {spread}) "
                  f"[{'OK' if agree >= 3 and spread <= 3 else 'CHECK'}]")
        else:
            result[ang] = None
            print(f"  {ang} -> UNRESOLVED")
    print("\nGAME_OFFS entry:", {ang: result.get(ang) for ang in ANGLES})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
