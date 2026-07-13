#!/usr/bin/env python3
"""SINGLE-PLAYER tracking audit (operator design): pick one player, follow ONLY him
for the whole window — his box highlighted in every camera that sees him, his dot and
trail on the court, and a visibility strip saying which cameras hold him each moment.
Any identity mistake is instantly visible: the highlight jumps to the wrong person.

Writes a per-player folder: video + trace.json (frame, per-cam boxes, court xy,
coasting, member local-ids) for ground-truth annotation and later scoring.

  python scripts/render_player_track.py --game e6fba750 --tag 44_60 \
      --worldstate runs/tracking/e6fba750_44_60_worldstate_v2_2.json \
      --version v2.2 --select "jersey:43"
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
HL = (0, 240, 255)               # highlight
DIM = (70, 70, 70)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--worldstate", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--select", required=True,
                    help='"jersey:43" (team optional: "jersey:B22") or "gid:10"')
    ap.add_argument("--clip-dir", default="data/clips")
    a = ap.parse_args()

    ws = json.loads(Path(a.worldstate).read_text())
    if a.select.startswith("gid:"):
        gid = int(a.select.split(":")[1])
        who = f"G{gid}"
    else:
        spec = a.select.split(":")[1]
        team = spec[0] if spec[0] in "AB" else None
        num = int(spec[1:] if team else spec)
        cands = [p for p in ws["players"] if p.get("jersey") == num
                 and (team is None or p.get("team") == team)]
        if not cands:
            print(f"no player with jersey {spec} in this worldstate; named: "
                  f"{[(p['team'], p['jersey']) for p in ws['players'] if p.get('jersey')]}")
            return 1
        gid = cands[0]["global_id"]
        who = f"{cands[0].get('team') or ''}#{num}"

    by_frame = {}
    for fr in ws["frames"]:
        for t in fr["tracks"]:
            if t["global_id"] == gid:
                by_frame[fr["frame"]] = t
    tracks_by = {}
    for ang in ANGLES:
        d = json.loads((REPO / f"runs/tracking/{a.game}_{ang}_{a.tag}_teams.json").read_text())
        m = defaultdict(dict)
        for t in d["tracks"]:
            m[t["frame"]][t["track_id"]] = t
        tracks_by[ang] = m
    caps = {ang: cv2.VideoCapture(str(REPO / a.clip_dir / f"{a.game}_{ang}_{a.tag}.mp4"))
            for ang in ANGLES}
    n_frames = max(f["frame"] for f in ws["frames"]) + 1

    base, to_px = draw_court(scale=0.40, margin=26)
    hb, wb = base.shape[:2]
    vcourt_base = np.rot90(base).copy()
    vh, vw_ = vcourt_base.shape[:2]

    def vpx(xy):
        x, y = to_px(xy)
        return (y, wb - 1 - x)

    pdir = REPO / f"runs/tracking/players/{a.game}_{a.tag}/{who.replace('#', 'n')}_{a.version}"
    pdir.mkdir(parents=True, exist_ok=True)
    raw = pdir / "player_raw.mp4"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    gx0, gy0 = 0, 74 + (H - 74 - 720) // 2
    cx0 = 1280 + (640 - vw_) // 2
    cy0 = 74 + (H - 74 - vh) // 2
    trail: list = []
    trace = []
    for f in range(n_frames):
        t = by_frame.get(f)
        members = (t or {}).get("members") or {}
        canvas = np.zeros((H, W, 3), np.uint8)
        rec = {"frame": f, "court_xy": (t or {}).get("court_xy"),
               "coasting": bool((t or {}).get("coasting")) if t else None, "cams": {}}
        for k, ang in enumerate(ANGLES):
            ok, img = caps[ang].read()
            if not ok:
                img = np.zeros((1080, 1920, 3), np.uint8)
            row = tracks_by[ang].get(f, {})
            for tid, tr in row.items():
                x1, y1, x2, y2 = (int(v) for v in tr["box_xyxy"])
                cv2.rectangle(img, (x1, y1), (x2, y2), DIM, 2)
            lid = members.get(ang)
            seen = lid is not None and lid in row
            if seen:
                tr = row[lid]
                x1, y1, x2, y2 = (int(v) for v in tr["box_xyxy"])
                cv2.rectangle(img, (x1, y1), (x2, y2), HL, 7)
                cv2.putText(img, who, (x1, max(40, y1 - 14)), FONT, 2.2, HL, 5, cv2.LINE_AA)
                rec["cams"][ang] = {"local_id": lid,
                                    "box": [round(v, 1) for v in tr["box_xyxy"]]}
            tile = cv2.resize(img, (640, 360))
            r, c = divmod(k, 2)
            y0 = gy0 + r * 360
            canvas[y0:y0 + 360, gx0 + c * 640:gx0 + (c + 1) * 640] = tile
            tag_col = HL if seen else (120, 120, 120)
            cv2.putText(canvas, f"{ang}{' *' if seen else ''}",
                        (gx0 + c * 640 + 10, y0 + 28), FONT, 0.85, tag_col, 2, cv2.LINE_AA)
        court = vcourt_base.copy()
        if t:
            trail.append(vpx(t["court_xy"]))
        for i in range(1, len(trail)):
            cv2.line(court, trail[i - 1], trail[i], (90, 200, 90), 2)
        if t:
            p = vpx(t["court_xy"])
            if t.get("coasting"):
                cv2.circle(court, p, 15, HL, 3)
            else:
                cv2.circle(court, p, 15, HL, -1)
        canvas[cy0:cy0 + vh, cx0:cx0 + vw_] = court
        cv2.rectangle(canvas, (0, 0), (W, 74), (12, 12, 12), -1)
        state = ("NOT TRACKED" if not t else
                 "COASTING (predicted)" if t.get("coasting") else
                 f"seen by {', '.join(sorted(rec['cams']))}")
        cv2.putText(canvas, f"PLAYER AUDIT {who}  [{a.version}]", (28, 32), FONT, 0.8,
                    HL, 2, cv2.LINE_AA)
        cv2.putText(canvas, state, (28, 62), FONT, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        mm, ss = divmod(f // FPS, 60)
        cv2.putText(canvas, f"{mm:02d}:{ss:02d}", (W - 120, 46), FONT, 0.9,
                    (170, 170, 170), 2, cv2.LINE_AA)
        vw.write(canvas)
        trace.append(rec)
    vw.release()
    for c in caps.values():
        c.release()

    (pdir / "trace.json").write_text(json.dumps(
        {"game": a.game, "tag": a.tag, "version": a.version, "who": who, "gid": gid,
         "frames": trace}, indent=0))
    final = pdir / f"player_{who.replace('#', 'n')}_{a.version}.mp4"
    subprocess.run(["ffmpeg", "-y", "-i", str(raw), "-c:v", "libx264", "-preset", "fast",
                    "-crf", "23", "-pix_fmt", "yuv420p", str(final)],
                   check=True, capture_output=True)
    raw.unlink()
    n_seen = sum(1 for r in trace if r["cams"])
    n_coast = sum(1 for r in trace if r["coasting"])
    print(f"{who}: tracked {n_seen}/{len(trace)} frames observed, {n_coast} coasting")
    print(f"video -> {final}\ntrace -> {pdir / 'trace.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
