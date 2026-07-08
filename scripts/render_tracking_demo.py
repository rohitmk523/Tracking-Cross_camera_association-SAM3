#!/usr/bin/env python3
"""Tracking-progression demo: FUSE VIEW ONLY, full window (operator spec — no detection
stage; one versioned video per tracking improvement, same window every time so versions
diff honestly).

LEFT = 2x2 camera grid with FUSED team+jersey labels (cross-camera knowledge painted on
every panel), RIGHT = vertical court (EMA-smoothed dots). Hollow dot = briefly unseen.

  python scripts/render_tracking_demo.py --game e6fba750 --tag 44_60 \
      --worldstate runs/tracking/e6fba750_44_60_worldstate.json \
      --version v0_baseline
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from uball_cc.fusion.court import draw_court  # noqa: E402

ANGLES = ("FL", "FR", "NL", "NR")
W, H, FPS = 1920, 1080, 30
TEAM = {"A": (255, 210, 60), "B": (90, 220, 90), "REF": (0, 230, 230)}
FONT = cv2.FONT_HERSHEY_SIMPLEX
ALPHA = 0.35                      # court-dot EMA (display only)


def label_box(img, x1, y1, x2, y2, col, lbl, fs=2.0, th=5, bth=5):
    cv2.rectangle(img, (x1, y1), (x2, y2), col, bth)
    (tw, tth), _ = cv2.getTextSize(lbl, FONT, fs, th)
    yt = max(tth + 10, y1 - 8)
    cv2.rectangle(img, (x1, yt - tth - 8), (x1 + tw + 10, yt + 6), (0, 0, 0), -1)
    cv2.putText(img, lbl, (x1 + 5, yt), FONT, fs, col, th, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--worldstate", required=True)
    ap.add_argument("--version", required=True, help="e.g. v0_baseline, v1_bytetrack-tuned")
    ap.add_argument("--clip-dir", default="data/clips")
    ap.add_argument("--out-dir", default="runs/tracking")
    a = ap.parse_args()

    ws = json.loads(Path(a.worldstate).read_text())
    ws_frames = {fr["frame"]: fr["tracks"] for fr in ws["frames"]}
    roster = {p["global_id"]: p["jersey"] for p in ws["players"] if p.get("jersey") is not None}
    tracks_by = {}
    for ang in ANGLES:
        d = json.loads((REPO / f"runs/tracking/{a.game}_{ang}_{a.tag}_teams.json").read_text())
        m = defaultdict(list)
        for t in d["tracks"]:
            m[t["frame"]].append(t)
        tracks_by[ang] = m
    caps = {ang: cv2.VideoCapture(str(REPO / a.clip_dir / f"{a.game}_{ang}_{a.tag}.mp4"))
            for ang in ANGLES}
    n_frames = max(ws_frames) + 1

    base, to_px = draw_court(scale=0.40, margin=26)
    hb, wb = base.shape[:2]
    vcourt_base = np.rot90(base).copy()
    vh, vw_ = vcourt_base.shape[:2]

    def vpx(court_xy):
        x, y = to_px(court_xy)
        return (y, wb - 1 - x)

    gx0 = 0
    gy0 = 74 + (H - 74 - 720) // 2
    cx0 = 1280 + (640 - vw_) // 2
    cy0 = 74 + (H - 74 - vh) // 2
    raw_out = REPO / a.out_dir / f"demo_track_{a.version}_{a.game}_raw.mp4"
    vw = cv2.VideoWriter(str(raw_out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    smoothed: dict[int, tuple] = {}

    for f in range(n_frames):
        tracks = ws_frames.get(f, [])
        gmap = {}
        for t in tracks:
            if not t.get("coasting"):
                for cam, lid in (t.get("members") or {}).items():
                    gmap[(cam, lid)] = (t["global_id"], t.get("team"))
        canvas = np.zeros((H, W, 3), np.uint8)
        for k, ang in enumerate(ANGLES):
            ok, img = caps[ang].read()
            if not ok:
                img = np.zeros((1080, 1920, 3), np.uint8)
            for t in tracks_by[ang].get(f, []):
                x1, y1, x2, y2 = (int(v) for v in t["box_xyxy"])
                hit = gmap.get((ang, t["track_id"]))
                if hit:
                    gid, team = hit
                    num = roster.get(gid)
                    lbl = f"{team or ''}#{num}" if num is not None else f"G{gid}"
                    label_box(img, x1, y1, x2, y2, TEAM.get(team, (170, 170, 170)), lbl)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (105, 105, 105), 2)
            tile = cv2.resize(img, (640, 360))
            r, c = divmod(k, 2)
            y0 = gy0 + r * 360
            canvas[y0:y0 + 360, gx0 + c * 640:gx0 + (c + 1) * 640] = tile
            cv2.putText(canvas, ang, (gx0 + c * 640 + 10, y0 + 28), FONT, 0.85,
                        (255, 255, 255), 2, cv2.LINE_AA)
        court = vcourt_base.copy()
        for t in tracks:
            col = TEAM.get(t.get("team"), (170, 170, 170))
            gid = t["global_id"]
            xy = t["court_xy"]
            if gid in smoothed:
                sx, sy = smoothed[gid]
                xy = (sx + ALPHA * (xy[0] - sx), sy + ALPHA * (xy[1] - sy))
            smoothed[gid] = tuple(xy)
            p = vpx(xy)
            if t.get("coasting"):
                cv2.circle(court, p, 13, col, 3)
            else:
                cv2.circle(court, p, 13, col, -1)
                cv2.circle(court, p, 13, (0, 0, 0), 1)
            num = roster.get(gid)
            tag = f"#{num}" if num is not None else str(gid)
            cv2.putText(court, tag, (p[0] + 15, p[1] + 7), FONT, 0.85, col, 2, cv2.LINE_AA)
        canvas[cy0:cy0 + vh, cx0:cx0 + vw_] = court
        cv2.rectangle(canvas, (0, 0), (W, 74), (12, 12, 12), -1)
        cv2.putText(canvas, f"TRACKING {a.version}", (28, 32), FONT, 0.8,
                    (120, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{a.game} {a.tag} — fused identities, all four cameras",
                    (28, 62), FONT, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        mm, ss = divmod(f // FPS, 60)
        cv2.putText(canvas, f"{mm:02d}:{ss:02d}", (W - 120, 46), FONT, 0.9,
                    (170, 170, 170), 2, cv2.LINE_AA)
        vw.write(canvas)
    vw.release()
    for c in caps.values():
        c.release()

    final = REPO / a.out_dir / f"demo_track_{a.version}_{a.game}.mp4"
    subprocess.run(["ffmpeg", "-y", "-i", str(raw_out), "-c:v", "libx264", "-preset",
                    "fast", "-crf", "23", "-pix_fmt", "yuv420p", str(final)],
                   check=True, capture_output=True)
    raw_out.unlink()
    print(f"demo -> {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
