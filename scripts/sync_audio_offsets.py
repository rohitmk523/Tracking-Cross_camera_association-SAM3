#!/usr/bin/env python3
"""Camera sync offsets from audio cross-correlation, straight off S3.

Downloads short mono audio windows from each angle via presigned URLs and
cross-correlates each angle against FL. Reports per-angle offsets in frames
(GAME_OFFS convention: cam_frame = ref_frame + offs[ang], i.e. positive =
the same real moment appears LATER in that camera's file).

  .venv/bin/python scripts/sync_audio_offsets.py --s3dir \
      court-a/2026-01-31/13e1ffad-e08b-4e1e-84bf --windows 300,900,1500
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ANGLES = ("FL", "FR", "NL", "NR")
SR = 8000
FPS = 29.97
BUCKET = "uball-videos-production"


def presign(key: str) -> str:
    return subprocess.run(
        ["aws", "s3", "presign", f"s3://{BUCKET}/{key}", "--expires-in", "3600"],
        check=True, capture_output=True, text=True).stdout.strip()


def grab(url: str, t0: float, dur: float, out: Path) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(t0),
         "-i", url, "-t", str(dur), "-vn", "-ac", "1", "-ar", str(SR),
         "-y", str(out)], capture_output=True)
    return r.returncode == 0 and out.exists() and out.stat().st_size > SR


def load(p: Path) -> np.ndarray:
    with wave.open(str(p)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    x = x.astype(np.float64)
    # bandpass 500-3000 Hz: whistles / buzzer / ball impacts give the sharp
    # transients that correlate; diffuse crowd rumble does not
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(len(x), 1 / SR)
    X[(fr < 500) | (fr > 3000)] = 0
    x = np.fft.irfft(X, len(x))
    x -= x.mean()
    s = x.std()
    return x / s if s > 0 else x


def xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag_s: float = 2.0):
    """Lag k (samples) maximizing correlation; b(t) ~ a(t - k/SR).
    Positive k = b is DELAYED vs a. Returns (k, peak_ratio). FFT-based."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    max_lag = int(max_lag_s * SR)
    nfft = 1 << int(np.ceil(np.log2(2 * n - 1)))
    # full cross-correlation c[k] = sum a[i] b[i+k] via FFT
    C = np.fft.irfft(np.conj(np.fft.rfft(a, nfft)) * np.fft.rfft(b, nfft), nfft)
    lags = np.arange(-max_lag, max_lag + 1)
    c = C[lags % nfft]
    # normalize by overlap length so long lags aren't penalized
    c = c / (n - np.abs(lags))
    j = int(np.argmax(c))
    peak = c[j]
    away = np.abs(lags - lags[j]) > int(0.25 * SR)
    second = c[away].max() if away.any() else 0.0
    ratio = peak / abs(second) if second else float("inf")
    return int(lags[j]), float(ratio)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3dir", required=True,
                    help="e.g. court-a/2026-01-31/13e1ffad-e08b-4e1e-84bf")
    ap.add_argument("--windows", default="200,500,800,1100,1400,1700,2000,2300",
                    help="comma window starts (s)")
    ap.add_argument("--min-ratio", type=float, default=3.0,
                    help="keep only windows whose corr peak beats the runner-up "
                         "by this factor")
    ap.add_argument("--dur", type=float, default=45.0)
    a = ap.parse_args()
    stem = a.s3dir.rstrip("/").split("/")[-1]
    date = a.s3dir.rstrip("/").split("/")[-2]
    wins = [float(w) for w in a.windows.split(",")]

    urls = {ang: presign(f"{a.s3dir.rstrip('/')}/{date}_{stem}_{ang}.mp4")
            for ang in ANGLES}
    tmp = Path(tempfile.mkdtemp(prefix="syncaud_"))
    audio: dict[tuple[str, float], np.ndarray] = {}
    for ang in ANGLES:
        for w in wins:
            p = tmp / f"{ang}_{int(w)}.wav"
            if grab(urls[ang], w, a.dur, p):
                audio[(ang, w)] = load(p)
            else:
                print(f"  WARN: no audio {ang}@{w}s", file=sys.stderr)

    print(f"{stem}: offsets vs FL (frames @ {FPS}):")
    result = {"FL": 0}
    for ang in ("FR", "NL", "NR"):
        cands = []
        for w in wins:
            ref, oth = audio.get(("FL", w)), audio.get((ang, w))
            if ref is None or oth is None:
                continue
            k, ratio = xcorr_lag(ref, oth)
            fr = k / SR * FPS
            keep = ratio >= a.min_ratio
            cands.append((fr, ratio, keep))
            print(f"  {ang} win{int(w):>5}s: {fr:+6.2f} frames "
                  f"({k / SR * 1000:+7.1f} ms, peak-ratio {ratio:4.1f})"
                  f"{'' if keep else '  [dropped: weak peak]'}")
        good = [c[0] for c in cands if c[2]]
        if len(good) >= 2:
            best = int(round(float(np.median(good))))
            result[ang] = best
            spread = float(np.max(good) - np.min(good))
            print(f"  {ang} -> {best:+d} frames "
                  f"({len(good)} confident windows, spread {spread:.1f}f)")
        else:
            result[ang] = None
            print(f"  {ang} -> UNRESOLVED ({len(good)} confident windows) — "
                  f"needs manual sync")
    print("\nGAME_OFFS entry:", {ang: result.get(ang) for ang in ANGLES})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
