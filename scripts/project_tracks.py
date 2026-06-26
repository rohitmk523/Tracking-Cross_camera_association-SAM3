#!/usr/bin/env python3
"""Project tracklet foot points to the top-down court (manual homography) + radar.

  python scripts/project_tracks.py --tracks runs/tracking/e6fba750_FL_47_12_teams.json \
      --calib configs/calib_e6fba750_FL.json \
      --save-video runs/tracking/e6fba750_FL_47_12_radar.mp4

Reports reprojection error (cm) on the calibration points and the % of projected
player positions that land inside the court — a quick sanity signal for the calib.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


TEAM_COLOR = {"A": (0, 140, 255), "B": (255, 120, 40), "REF": (0, 255, 255)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--save-video", default=None)
    ap.add_argument("--out-fps", type=float, default=30.0)
    a = ap.parse_args()

    import cv2

    from uball_cc.fusion.court import X0, X1, Y0, Y1, draw_court
    from uball_cc.fusion.homography import homography_from_calib, load_calib, project, reprojection_error
    from uball_cc.tracking import Track

    calib = load_calib(a.calib)
    h = homography_from_calib(calib)
    if "correspondences" in calib:
        print("reprojection error (cm):", reprojection_error(calib["correspondences"], h))

    tracks = [Track.from_record(r) for r in json.loads(Path(a.tracks).read_text())["tracks"]]
    court = project([t.foot_xy for t in tracks], h)
    inb = ((court[:, 0] >= X0) & (court[:, 0] <= X1) & (court[:, 1] >= Y0) & (court[:, 1] <= Y1))
    print(f"projected {len(tracks)} observations; inside court: {inb.mean() * 100:.0f}%")

    if not a.save_video:
        return 0
    by_frame: dict[int, list] = collections.defaultdict(list)
    for t, c in zip(tracks, court):
        by_frame[t.frame].append((t, c))

    base, to_px = draw_court()
    for cpt in calib.get("correspondences", []):               # calib points = green crosses
        cv2.drawMarker(base, to_px(cpt["court"]), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 12, 1)

    out = Path(a.save_video)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for f in sorted(by_frame):
        img = base.copy()
        for t, c in by_frame[f]:
            col = TEAM_COLOR.get(t.team, (200, 200, 200))
            cv2.circle(img, to_px(c), 6, col, -1)
            cv2.circle(img, to_px(c), 6, (0, 0, 0), 1)
        if writer is None:
            hh, ww = img.shape[:2]
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), a.out_fps, (ww, hh))
        writer.write(img)
    if writer:
        writer.release()
    print(f"radar -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
