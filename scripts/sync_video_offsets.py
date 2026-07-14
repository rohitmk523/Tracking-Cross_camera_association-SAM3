#!/usr/bin/env python3
"""Camera sync offsets from VIDEO motion energy (audio-free fallback).

Game flow is globally synchronized across views: play starts/stops, fast
breaks, and dead balls modulate whole-frame motion in every camera at the
same instant. Per camera we build a per-frame motion-energy signal
(mean |frame diff| on a downscaled gray image) over several windows and
cross-correlate against FL — same confidence gating as the audio tool.

Runs on the local low-res fullsrc files (fetch them first).

  .venv/bin/python scripts/sync_video_offsets.py --game 13e1ffad
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")
FPS = 29.97


def motion_signal(path: Path, t0: float, dur: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(t0 * FPS))
    n = int(dur * FPS)
    sig = np.empty(n)
    prev = None
    got = 0
    for i in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(fr, (155, 112)), cv2.COLOR_BGR2GRAY)
        g = g.astype(np.float32)
        sig[i] = float(np.mean(np.abs(g - prev))) if prev is not None else 0.0
        prev = g
        got = i + 1
    cap.release()
    if got < n * 0.9:
        return None
    s = sig[1:got]
    s = s - s.mean()
    sd = s.std()
    return s / sd if sd > 0 else s


def xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 60):
    """Lag k (frames) maximizing correlation; positive k = b delayed vs a."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    nfft = 1 << int(np.ceil(np.log2(2 * n - 1)))
    C = np.fft.irfft(np.conj(np.fft.rfft(a, nfft)) * np.fft.rfft(b, nfft), nfft)
    lags = np.arange(-max_lag, max_lag + 1)
    c = C[lags % nfft] / (n - np.abs(lags))
    j = int(np.argmax(c))
    peak = c[j]
    away = np.abs(lags - lags[j]) > 8
    second = c[away].max() if away.any() else 0.0
    ratio = peak / abs(second) if second else float("inf")
    return int(lags[j]), float(ratio)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--windows", default="200,600,1000,1400,1800,2200,2600")
    ap.add_argument("--dur", type=float, default=60.0)
    ap.add_argument("--min-ratio", type=float, default=1.5,
                    help="motion xcorr peaks are broader than audio; gate lower "
                         "but require agreement across windows instead")
    ap.add_argument("--srcdir", default="runs/event_demo")
    a = ap.parse_args()
    wins = [float(w) for w in a.windows.split(",")]
    paths = {ang: REPO / a.srcdir / f"fullsrc_{a.game}_{ang}.mp4"
             for ang in ANGLES}

    sig = {}
    for ang in ANGLES:
        for w in wins:
            s = motion_signal(paths[ang], w, a.dur)
            if s is not None:
                sig[(ang, w)] = s

    print(f"{a.game}: offsets vs FL (frames, video motion xcorr):")
    result = {"FL": 0}
    for ang in ("FR", "NL", "NR"):
        cands = []
        for w in wins:
            ref, oth = sig.get(("FL", w)), sig.get((ang, w))
            if ref is None or oth is None:
                continue
            k, ratio = xcorr_lag(ref, oth)
            keep = ratio >= a.min_ratio
            cands.append((k, ratio, keep))
            print(f"  {ang} win{int(w):>5}s: {k:+4d} frames "
                  f"(peak-ratio {ratio:4.2f}){'' if keep else '  [weak]'}")
        good = [c[0] for c in cands if c[2]]
        if len(good) >= 3:
            vals, counts = np.unique(good, return_counts=True)
            mode = int(vals[np.argmax(counts)])
            agree = int(counts.max())
            med = int(round(float(np.median(good))))
            best = mode if agree >= 3 else med
            spread = int(np.max(good) - np.min(good))
            verdict = "OK" if (agree >= 3 or spread <= 2) else "LOOSE"
            result[ang] = best
            print(f"  {ang} -> {best:+d} frames ({len(good)} windows, mode-agree "
                  f"{agree}, spread {spread}) [{verdict}]")
        else:
            result[ang] = None
            print(f"  {ang} -> UNRESOLVED ({len(good)} confident windows)")
    print("\nGAME_OFFS entry:", {ang: result.get(ang) for ang in ANGLES})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
