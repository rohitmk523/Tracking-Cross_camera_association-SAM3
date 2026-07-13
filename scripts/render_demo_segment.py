#!/usr/bin/env python3
"""Client demo: a CONTINUOUS game segment with the pipeline's event feed and
the operator's GT feed side by side, events appearing at their timestamps.

Center: game video (one angle). Left panel: PIPELINE events (ours, source:cv).
Right panel: GROUND TRUTH plays. Each event slides in when it occurs and stays
in a rolling feed; the newest row flashes. Honest by construction — every CV
event in the window is shown, right or wrong.

  python scripts/render_demo_segment.py --game c2a354fe --t0 2100 --dur 480 \
      --angle FL --out runs/event_demo/c2a_demo_segment.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from collections import defaultdict
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from game_meta import GAME_CHUNKS

REPO = Path(__file__).resolve().parents[1]
W, H = 1600, 900
PANEL = 360
FPS = 29.97
FEED_ROWS = 9
FLASH_S = 2.5


def short(cls):
    return (cls.replace("FREE_THROW", "FT").replace("_", " "))


def draw_feed(img, x0, title, feed, t_now, color):
    cv2.rectangle(img, (x0, 0), (x0 + PANEL, H), (24, 24, 28), -1)
    cv2.putText(img, title, (x0 + 18, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.line(img, (x0 + 18, 62), (x0 + PANEL - 18, 62), color, 1)
    y = 108
    for e in feed[-FEED_ROWS:][::-1]:
        age = t_now - e["t"]
        hot = age <= FLASH_S
        col = (90, 230, 255) if hot else (225, 225, 230)
        cv2.putText(img, f"{int(e['t'])//60:02d}:{int(e['t'])%60:02d}  {short(e['classification'])}",
                    (x0 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2 if hot else 1)
        y += 26
        nm = e.get("player_a") or "?"
        cv2.putText(img, f"   {nm[:26]}", (x0 + 18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 155, 160) if not hot else col, 1)
        y += 34
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--dur", type=float, default=480.0)
    ap.add_argument("--angle", default="FL")
    ap.add_argument("--plays", default=None)
    ap.add_argument("--src", default=None, help="local video/clip; else presigned S3 cut")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-shift", type=float, default=0.0,
                    help="seconds ADDED to GT t for display alignment")
    a = ap.parse_args()

    plays = json.loads((REPO / (a.plays or f"data/plays/{a.game}_full.json")).read_text())["plays"]
    ev = json.loads((REPO / f"runs/tracking/ledger/events_v2_{a.game}.json").read_text())["events"]
    pos = REPO / f"runs/tracking/ledger/possession_events_{a.game}.json"
    if pos.exists():
        ev = ev + json.loads(pos.read_text())["events"]
    ev.sort(key=lambda e: e["t"])
    # identity tracks for the rendered angle: predicted-player boxing
    roster = json.loads((REPO / f"data/rosters/{a.game}.json").read_text())
    by_num = defaultdict(list)
    for pr in roster["players"]:
        by_num[pr["num"]].append(pr)
    def stream_of(name):
        recs = [p for p in roster["players"] if p["name"] == name]
        if not recs:
            return None
        r = recs[0]
        dual = len(by_num[r["num"]]) > 1
        return f"n{r['num']}" + (("B" if r["team"] == 1 else "W") if dual else "")
    tracks = {}
    for tag in GAME_CHUNKS[a.game]:
        t0c = float(tag.split("_")[0])
        tdir = REPO / (f"runs/events_fg_{tag}" if a.game == "e6fba750"
                       else f"runs/events_fg_{a.game[:3]}_{tag}")
        for p in tdir.glob(f"{a.game}_{tag}__n*__{a.angle}.json"):
            sid = p.stem.split("__")[1]
            d = json.loads(p.read_text())["frames"]
            tracks.setdefault(sid, {}).update(
                {round(t0c * FPS) + int(fr): r["box"]
                 for fr, r in d.items() if r.get("present")})
    gt = [{"t": p["t"] + a.gt_shift, "classification": p["cls"], "player_a": p["a"]}
          for p in plays]

    # source segment
    src = a.src
    if src is None:
        gj = json.loads((REPO / "configs/games.json").read_text())
        g = next(x for x in gj["working_games"] if x["gid8"] == a.game)
        pfx = g["s3_prefix"].rstrip("/")
        date = pfx.split("/")[1]
        base = pfx.split("/")[2]
        key = f"{pfx}/{date}_{base}_{a.angle}.mp4"
        url = subprocess.run(["aws", "s3", "presign", f"s3://uball-videos-production/{key}",
                              "--expires-in", "7200"], capture_output=True, text=True).stdout.strip()
        src = str(REPO / f"runs/event_demo/_seg_{a.game}_{a.angle}_{int(a.t0)}.mp4")
        if not Path(src).exists():
            print("cutting segment from S3...", flush=True)
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error",
                            "-ss", str(a.t0), "-i", url, "-t", str(a.dur),
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                            "-pix_fmt", "yuv420p", "-an", "-y", src], check=True)

    cap = cv2.VideoCapture(src)
    vw_w = W - 2 * PANEL
    outp = REPO / a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(outp) + ".raw.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok or fi > a.dur * FPS:
            break
        t_now = a.t0 + fi / FPS
        canvas = np.zeros((H, W, 3), np.uint8)
        vid = cv2.resize(frame, (vw_w, int(vw_w * frame.shape[0] / frame.shape[1])))
        y0 = (H - vid.shape[0]) // 2
        canvas[y0:y0 + vid.shape[0], PANEL:PANEL + vw_w] = vid
        # box the predicted player around each firing CV event
        for e in ev:
            if not (e["t"] - 0.8 <= t_now <= e["t"] + 3.0) or not e.get("player_a"):
                continue
            sid = stream_of(e["player_a"])
            box = tracks.get(sid, {}).get(round(t_now * FPS)) if sid else None
            if box is None:
                continue
            fh, fw = frame.shape[:2]
            sx = vw_w / fw
            sy = vid.shape[0] / fh
            x1, y1 = PANEL + int(box[0] * sx), y0 + int(box[1] * sy)
            x2, y2 = PANEL + int(box[2] * sx), y0 + int(box[3] * sy)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (90, 230, 255), 3)
            cv2.putText(canvas, e["player_a"].split()[-1], (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 230, 255), 2)
        cv_feed = [e for e in ev if a.t0 - 1 <= e["t"] <= t_now]
        gt_feed = [e for e in gt if a.t0 - 1 <= e["t"] <= t_now]
        draw_feed(canvas, 0, "PIPELINE (ours)", cv_feed, t_now, (60, 180, 230))
        draw_feed(canvas, W - PANEL, "GROUND TRUTH", gt_feed, t_now, (200, 200, 210))
        clock = f"{int(t_now)//60:02d}:{int(t_now)%60:02d}"
        cv2.putText(canvas, clock, (PANEL + 16, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (240, 240, 245), 2)
        cv2.putText(canvas, f"{a.game[:8]}  cam {a.angle}", (PANEL + 16, H - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 155, 160), 1)
        vw.write(canvas)
        fi += 1
        if fi % 3000 == 0:
            print(f"  {fi/FPS:.0f}s rendered", flush=True)
    cap.release()
    vw.release()
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", tmp,
                    "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p",
                    "-y", str(outp)], check=True)
    Path(tmp).unlink()
    print(f"-> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
