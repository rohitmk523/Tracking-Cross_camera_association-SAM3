"""Audio sync: align camera clips by FFT cross-correlation of their audio.

Ported from uball_shot_detection_dual_fusion_v2/pipeline/clip_audio_sync.py — mono
16 kHz, cross-correlate the normalized first-difference (robust to level/music), pick
the peak lag. Used to put per-camera tracklets on a common timeline before fusion.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

SR = 16000
MAX_LAG_S = 2.5


def _wav(clip: Path, out: Path) -> bool:
    r = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(clip),
                        "-vn", "-ac", "1", "-ar", str(SR), str(out)],
                       capture_output=True, timeout=120)
    return r.returncode == 0 and out.exists() and out.stat().st_size > 1000


def _load(p: Path) -> np.ndarray:
    from scipy.io import wavfile  # noqa: PLC0415

    _, x = wavfile.read(p)
    x = np.diff(x.astype(np.float64))
    s = x.std()
    return x / s if s > 0 else x


def _xcorr(a: np.ndarray, b: np.ndarray, max_lag_s: float = MAX_LAG_S) -> tuple[float, float]:
    n = len(a) + len(b)
    nfft = 1 << (n - 1).bit_length()
    fa, fb = np.fft.rfft(a, nfft), np.fft.rfft(b, nfft)
    c = np.fft.irfft(fb * np.conj(fa), nfft)
    c = np.concatenate([c[-len(a) + 1:], c[:len(b)]])
    lags = np.arange(-len(a) + 1, len(b))
    keep = np.abs(lags) <= int(max_lag_s * SR)
    c, lags = c[keep], lags[keep]
    i = int(np.argmax(c))
    return lags[i] / SR, float(c[i] / (np.median(np.abs(c)) + 1e-9))


MIN_PEAK = 3.0        # xcorr peak below this = no confident alignment found


def audio_offset_seconds(ref_clip, other_clip, *, min_peak: float = MIN_PEAK,
                         strict: bool = True) -> tuple[float, float]:
    """Seconds to add to a REF time to get the matching OTHER time, + a peak score.

    other_time ≈ ref_time + offset. (Sign verified empirically per rig; flip if the
    fused dots don't overlap.) peak >> 1 means a confident match.

    strict (default): RAISES when a clip has no usable audio or the peak is below
    min_peak, instead of silently returning offset 0 — fusing unsynced cameras corrupts
    every downstream measurement (2026-07-02 audit: NL/NR were fused at silent 0-offset
    because the clips had no audio stream). Pass strict=False only to probe.
    """
    with tempfile.TemporaryDirectory() as td:
        wa, wb = Path(td) / "a.wav", Path(td) / "b.wav"
        ok_a, ok_b = _wav(Path(ref_clip), wa), _wav(Path(other_clip), wb)
        if not (ok_a and ok_b):
            missing = [str(c) for c, ok in ((ref_clip, ok_a), (other_clip, ok_b)) if not ok]
            if strict:
                raise RuntimeError(
                    f"audio-sync impossible — no usable audio stream in: {missing}. "
                    "Re-pull the clip WITH audio (scripts/pull_clip.py keeps it by default; "
                    "avoid --no-audio) or fuse with audio_sync disabled EXPLICITLY.")
            return 0.0, 0.0
        lag_s, peak = _xcorr(_load(wa), _load(wb))
        if strict and peak < min_peak:
            raise RuntimeError(
                f"audio-sync low confidence (peak {peak:.1f} < {min_peak}) between "
                f"{ref_clip} and {other_clip} — offset would be unreliable. "
                "Check the clips cover the same moment, or sync by content.")
        return lag_s, peak
