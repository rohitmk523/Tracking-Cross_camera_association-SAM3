#!/usr/bin/env python3
"""Event demo reel: for each selected play, a short clip with the GROUND-TRUTH
event (right panel) and the PIPELINE prediction (left panel), plus a per-frame
box on the predicted player when his jersey number is read in-clip.

  python scripts/render_event_demo.py --plays /tmp/demo_plays.json \
      --clips-dir runs/event_demo/clips --out runs/event_demo/event_demo.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO / "src"))

W, H = 1280, 720
PANEL = 300


def wrap(txt, n=22):
    words, lines, cur = txt.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= n:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def panel(img, x0, title, lines, color):
    cv2.rectangle(img, (x0, 0), (x0 + PANEL, H), (28, 28, 32), -1)
    cv2.putText(img, title, (x0 + 16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.line(img, (x0 + 16, 58), (x0 + PANEL - 16, 58), color, 1)
    y = 100
    for label, val in lines:
        cv2.putText(img, label, (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 155), 1)
        y += 26
        for ln in wrap(val, 24):
            cv2.putText(img, ln, (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 240), 2)
            y += 30
        y += 14


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plays", default="/tmp/demo_plays.json")
    ap.add_argument("--angles", default="/tmp/demo_angles.json")
    ap.add_argument("--clips-dir", default="runs/event_demo/clips")
    ap.add_argument("--out", default="runs/event_demo/event_demo.mp4")
    ap.add_argument("--box", action="store_true", help="detect+read jersey to box the predicted player")
    a = ap.parse_args()
    plays = json.load(open(a.plays))
    angles = {float(k): v for k, v in json.load(open(a.angles)).items()}
    roster = json.loads((REPO / "data/rosters/e6fba750.json").read_text())
    num_of = {}
    for pr in roster["players"]:
        num_of.setdefault(pr["name"], pr["num"])

    stack = model = None
    if a.box:
        from ultralytics import YOLO
        from uball_cc.tracking.jersey_stack import JerseyStack
        model = YOLO(str(REPO / "runs/yolo26s-1280-ourdata-v1_fetch/runs/detect/runs/"
                         "yolo26s-1280-ourdata-v1/weights/best.pt"))
        stack = JerseyStack()

    tmpdir = REPO / "runs/event_demo/_frames"
    tmpdir.mkdir(parents=True, exist_ok=True)
    seg_paths = []
    for i, e in enumerate(plays):
        t = e["t"]
        ang = angles.get(t, "FL")
        clip = REPO / a.clips_dir / f"play_{t}_{ang}.mp4"
        if not clip.exists():
            print(f"missing clip {clip}"); continue
        pred_num = num_of.get((e["pred_player"] or "").rstrip("?"))
        gt_lines = [("EVENT", e["label"].replace("_", " ")),
                    ("PLAYER", e["gt_player"] or "-")]
        if e["gt_zone"]:
            gt_lines.append(("POINTS", e["gt_zone"]))
        pz = e["pred_zone"] or ("—" if e["gt_zone"] else "n/a")
        pred_lines = [("PLAYER", (e["pred_player"] or "no read").rstrip("?"))]
        if e["gt_zone"]:
            pred_lines.append(("POINTS (from court zone)", pz))
        who_mark = "OK" if e["who_ok"] else "MISS"
        who_col = (60, 200, 90) if e["who_ok"] else (60, 90, 230)

        cap = cv2.VideoCapture(str(clip))
        seg = tmpdir / f"seg_{i:02d}.mp4"
        vw = cv2.VideoWriter(str(seg), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vid = cv2.resize(frame, (W - 2 * PANEL, H))
            canvas = np.zeros((H, W, 3), np.uint8)
            canvas[:, PANEL:W - PANEL] = vid
            # box predicted player when his number is read in this frame
            if a.box and pred_num is not None and fi % 3 == 0:
                r = model.predict(frame, imgsz=1280, conf=0.25, classes=[0],
                                  device="mps", verbose=False)[0]
                fh, fw = frame.shape[:2]
                for b in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = b
                    if (y2 - y1) < 90:
                        continue
                    num, cf = stack.read_crop(frame[int(y1):int(y2), int(x1):int(x2)])
                    if num is not None and int(num) == pred_num:
                        sx = (W - 2 * PANEL) / fw
                        bx1, by1 = PANEL + int(x1 * sx), int(y1 * H / fh)
                        bx2, by2 = PANEL + int(x2 * sx), int(y2 * H / fh)
                        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), who_col, 3)
                        cv2.putText(canvas, f"#{pred_num}", (bx1, by1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, who_col, 2)
                        break
            panel(canvas, 0, "PIPELINE (ours)", pred_lines, (230, 180, 60))
            cv2.putText(canvas, f"WHO: {who_mark}", (16, H - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, who_col, 2)
            panel(canvas, W - PANEL, "GROUND TRUTH", gt_lines, (200, 200, 210))
            cv2.putText(canvas, f"play {i+1}/{len(plays)}", (W - PANEL + 16, H - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 155), 1)
            vw.write(canvas)
            fi += 1
        cap.release(); vw.release()
        seg_paths.append(seg)
        print(f"play {i+1}/{len(plays)} t={t} {e['label']} rendered", flush=True)

    lst = tmpdir / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in seg_paths))
    out = REPO / a.out
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
                   capture_output=True)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
