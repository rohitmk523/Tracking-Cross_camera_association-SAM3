#!/usr/bin/env python3
"""Full-court ball tracking: fuse MOTION ball candidates across the 4 cameras (docs/06/08).

Per camera: 3-frame motion detection (fusion/ball_motion) -> project to court (homography)
-> region-gate. Audio-sync each camera to a reference, fuse candidates by cross-camera
AGREEMENT (fusion/ball_fuse), drop fixed FPs (reject_stationary), and Kalman-track the ball
(fusion/ball). A single camera only sees its own half; the 4 fused give full-court coverage.

  python scripts/track_ball_4cam.py --game e6fba750 --suffix _fb --ref NR \
      --out runs/tracking/e6_ball_4cam_fb.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FPS = 29.97
ZONE = {"FL": 0.6, "FR": 0.6, "NL": 1.0, "NR": 1.0}     # near cams own the ends/floor; far cams downweighted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="e6fba750")
    ap.add_argument("--clip-dir", default="data/clips")
    ap.add_argument("--suffix", default="_fb")
    ap.add_argument("--calib-dir", default="configs/calib")
    ap.add_argument("--angles", nargs="+", default=["FL", "FR", "NL", "NR"])
    ap.add_argument("--ref", default="NR", help="sync-reference angle")
    ap.add_argument("--region-pad", type=float, default=500.0)
    ap.add_argument("--no-audio-sync", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import sys
    sys.path.insert(0, "src")
    from uball_cc.fusion.audiosync import audio_offset_seconds
    from uball_cc.fusion.ball import reject_stationary, track_ball
    from uball_cc.fusion.ball_fuse import fuse_ball_candidates
    from uball_cc.fusion.ball_motion import motion_candidates
    from uball_cc.fusion.homography import (calib_hull, homography_from_calib, in_calib_region,
                                            load_calib, project)
    from uball_cc.tracking import iter_video_frames

    def clip(ang):
        return Path(a.clip_dir) / f"{a.game}_{ang}{a.suffix}.mp4"

    ref_clip = clip(a.ref)
    per_cam: dict[str, dict[int, list[tuple]]] = {}
    for ang in a.angles:
        cp = clip(ang)
        if not cp.exists():
            print(f"  {ang}: clip missing {cp}", flush=True)
            continue
        calib = load_calib(Path(a.calib_dir) / f"{ang}.json")
        h, hull = homography_from_calib(calib), calib_hull(calib)
        frames = list(iter_video_frames(str(cp)))
        cand_img = motion_candidates(frames)
        # sync: cam frame fc corresponds to ref frame (fc - off_f)
        off_f = 0
        if not a.no_audio_sync and ang != a.ref:
            off_s, peak = audio_offset_seconds(str(ref_clip), str(cp))
            off_f = int(round(off_s * FPS))
            print(f"  {ang}: sync {off_f:+d} frames (peak {peak:.1f})", flush=True)
        zc = ZONE.get(ang, 1.0)
        court: dict[int, list[tuple]] = {}
        for fi, cands in cand_img.items():
            pts = project([(x, y) for x, y, _ in cands], h)
            for (cx, cy), (_x, _y, score) in zip(pts, cands):
                if in_calib_region((cx, cy), hull, a.region_pad):
                    court.setdefault(fi - off_f, []).append((float(cx), float(cy), float(score) * zc))
        per_cam[ang] = court
        print(f"  {ang}: motion candidates in {len(cand_img)} frames -> {len(court)} in-region", flush=True)

    fused = fuse_ball_candidates(per_cam)
    n_multi = sum(1 for v in fused.values() for c in v if c[2] > 0.3)  # rough agreement signal
    fused, banned = reject_stationary(fused)
    trace = track_ball(fused)
    print(f"\nfused candidate frames: {len(fused)} | banned {len(banned)} stationary | "
          f"tracked: {len(trace)} frames", flush=True)
    if trace:
        xy = np.array([trace[f] for f in sorted(trace)])
        print(f"  court x[{xy[:,0].min():.0f},{xy[:,0].max():.0f}] y[{xy[:,1].min():.0f},{xy[:,1].max():.0f}]"
              f" (court 2144x1426)", flush=True)
        print(f"  TRAVELED {xy[:,0].max()-xy[:,0].min():.0f}x{xy[:,1].max()-xy[:,1].min():.0f}cm | "
              f"net {np.linalg.norm(xy[-1]-xy[0]):.0f}cm | {n_multi} multi-cam-agreed frames", flush=True)

    dest = Path(a.out) if a.out else Path("runs/tracking") / f"{a.game}_ball_4cam.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(trace))
    Path(str(dest).replace(".json", "_fused.json")).write_text(
        json.dumps({str(f): c for f, c in fused.items()}))
    print(f"-> {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
