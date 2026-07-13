#!/usr/bin/env python3
"""ADAPTER: our ball+hoop cache (runs/ball_cache/*.ball.npz, yolo26s corrected
weights) -> the shot-detection repo's P1 tracks schema, so THEIR P2 features +
P3 make/miss model run unchanged on OUR detector (far_v16 retired).

Schema copied from their pipeline/extract_tracks.py (verified by code read):
one row per frame per (shot window x angle); ball/rim = per-frame PRIMARY
(best-conf) detection or NaN; x,y = box top-left; w,h sizes; pixels @1920x1080;
frame_idx = absolute video frame; windows = plays start/end +-2s buffer applied
to each angle's raw video time (per-angle sync offsets deliberately ignored,
matching their P1 convention).

Their detector emitted only conf >= 0.25 (measured from their e6 parquet);
ours caches at 0.15 — default floor 0.25 for distribution parity.

  python scripts/shotdet_p1_adapter.py --gid8 e6fba750 \
      --gt runs/shotdet_ab/gt_windows.json --out /tmp/p1tracks
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")
FPS = 29.97
BUFFER_S = 2.0                       # their TIMESTAMP_BUFFER_SECONDS
MAKE = {"3PT_MAKE", "FG_MAKE", "FREE_THROW_MAKE", "4PT_MAKE"}
MISS = {"3PT_MISS", "FG_MISS", "FREE_THROW_MISS", "4PT_MISS"}
CHUNK_RE = re.compile(r"_(\d+)_(\d+)\.ball\.npz$")


def load_angle(gid8: str, ang: str, floor: float):
    """Per-angle dicts abs_frame -> (x,y,w,h,conf) for ball and rim, plus the
    covered time ranges [(t0,t1)] of the chunks found on disk."""
    ball, rim, ranges = {}, {}, []
    for p in sorted((REPO / "runs/ball_cache").glob(f"{gid8}_{ang}_*.ball.npz")):
        m = CHUNK_RE.search(p.name)
        s, d = int(m.group(1)), int(m.group(2))
        ranges.append((float(s), float(s + d)))
        z = np.load(p)
        cls = z["classes"] if "classes" in z else np.zeros(len(z["scores"]))
        base = round(s * FPS)
        for b, sc, f, c in zip(z["boxes"], z["scores"], z["frame_idx"], cls):
            if sc < floor:
                continue
            af = base + int(f)
            rec = (float(b[0]), float(b[1]), float(b[2] - b[0]),
                   float(b[3] - b[1]), float(sc))
            tgt = rim if int(c) == 1 else ball
            if af not in tgt or sc > tgt[af][4]:
                tgt[af] = rec
        z.close()
    return ball, rim, ranges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid8", default="e6fba750")
    ap.add_argument("--gt", default="runs/shotdet_ab/gt_windows.json")
    ap.add_argument("--out", default="/tmp/p1tracks")
    ap.add_argument("--conf-floor", type=float, default=0.25)
    a = ap.parse_args()

    gt = json.loads((REPO / a.gt).read_text())["games"]
    uuid = next(g for g in gt if g.startswith(a.gid8))
    shots = [s for s in gt[uuid]
             if s.get("classification") in (MAKE | MISS)
             and s.get("start_timestamp") is not None
             and s.get("end_timestamp") is not None]
    shots.sort(key=lambda s: s["start_timestamp"])
    print(f"{uuid}: {len(shots)} shot windows in frozen GT")

    per_angle = {ang: load_angle(a.gid8, ang, a.conf_floor) for ang in ANGLES}
    for ang in ANGLES:
        b, r, rng = per_angle[ang]
        print(f"  {ang}: {len(b)} ball frames, {len(r)} rim frames, "
              f"coverage {[(int(x), int(y)) for x, y in rng]}")

    def covered(t0: float, t1: float, ranges) -> bool:
        # window fully inside the union of cached chunk ranges (chunks abut)
        need = t0
        for lo, hi in sorted(ranges):
            if lo <= need <= hi:
                need = hi
            if need >= t1:
                return True
        return need >= t1

    rows, skipped = [], []
    for s in shots:
        w0 = max(0.0, float(s["start_timestamp"]) - BUFFER_S)
        w1 = float(s["end_timestamp"]) + BUFFER_S
        if not all(covered(w0, w1, per_angle[ang][2]) for ang in ANGLES):
            skipped.append(s["play_id"])
            continue
        f0, f1 = int(np.ceil(w0 * FPS)), int(np.floor(w1 * FPS))
        for ang in ANGLES:
            ball, rim, _ = per_angle[ang]
            for af in range(f0, f1 + 1):
                b = ball.get(af)
                r = rim.get(af)
                rows.append({
                    "game_id": uuid, "play_id": s["play_id"],
                    "classification": s["classification"], "angle": ang,
                    "detector": "ours_yolo26s",
                    "frame_idx": af, "timestamp": round(af / FPS, 6),
                    "ball_x": b[0] if b else None, "ball_y": b[1] if b else None,
                    "ball_w": b[2] if b else None, "ball_h": b[3] if b else None,
                    "ball_conf": b[4] if b else None,
                    "rim_x": r[0] if r else None, "rim_y": r[1] if r else None,
                    "rim_w": r[2] if r else None, "rim_h": r[3] if r else None,
                    "rim_conf": r[4] if r else None,
                })

    df = pd.DataFrame(rows)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{uuid}.parquet"
    df.to_parquet(out, index=False)
    n_plays = df.play_id.nunique() if len(df) else 0
    print(f"wrote {out}: {len(df)} rows, {n_plays} plays "
          f"({len(skipped)} skipped outside cache coverage)")
    if len(df):
        print(f"ball frac {df.ball_conf.notna().mean():.3f} | "
              f"rim frac {df.rim_conf.notna().mean():.3f} "
              f"(theirs on e6: 0.328 / 0.844)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
