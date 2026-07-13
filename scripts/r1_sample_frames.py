#!/usr/bin/env python3
"""R1-full stage 0 — sample frames from S3 (1 per SAMPLE_S seconds, per cam)
and cut per-stream crops, giving every identity stream a dense appearance
timeline for tracklet split/merge analysis.

  .venv/bin/python scripts/r1_sample_frames.py --game c2a354fe
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from game_meta import GAME_CHUNKS

FPS = 29.97
SAMPLE_S = 2.0
ANGLES = ("FL", "FR", "NL", "NR")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--min-h", type=int, default=85)
    a = ap.parse_args()
    game = a.game
    gj = json.loads((REPO / "configs/games.json").read_text())
    g = next(x for x in gj["working_games"] if x["gid8"] == game)
    pfx = g["s3_prefix"].rstrip("/")
    date, base = pfx.split("/")[1], pfx.split("/")[2]

    out = REPO / f"runs/r1_samples_{game[:3]}"
    (out / "crops").mkdir(parents=True, exist_ok=True)

    # stream boxes per (ang, global frame)
    streams = defaultdict(lambda: defaultdict(dict))
    for tag in GAME_CHUNKS[game]:
        t0c = float(tag.split("_")[0])
        tdir = REPO / (f"runs/events_fg_{tag}" if game == "e6fba750"
                       else f"runs/events_fg_{game[:3]}_{tag}")
        for p in tdir.glob(f"{game}_{tag}__n*__*.json"):
            parts = p.stem.split("__")
            sid, ang = parts[1], parts[2]
            d = json.loads(p.read_text())["frames"]
            for fr, r in d.items():
                if r.get("present"):
                    streams[sid][ang][round(t0c * FPS) + int(fr)] = r["box"]

    index = []
    for ang in ANGLES:
        key = f"{pfx}/{date}_{base}_{ang}.mp4"
        u = subprocess.run(["aws", "s3", "presign",
                            f"s3://uball-videos-production/{key}",
                            "--expires-in", "14400"],
                           capture_output=True, text=True).stdout.strip()
        vdir = out / f"_frames_{ang}"
        vdir.mkdir(exist_ok=True)
        # one pass: dump a frame every SAMPLE_S seconds
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-i", u, "-vf", f"fps=1/{SAMPLE_S}", "-q:v", "4",
                        "-y", str(vdir / "f_%05d.jpg")], check=True)
        n_img = len(list(vdir.glob("*.jpg")))
        print(f"{ang}: {n_img} sampled frames", flush=True)
        for i in range(1, n_img + 1):
            t = (i - 1) * SAMPLE_S
            gf = round(t * FPS)
            img = None
            for sid, angs in streams.items():
                box = None
                for df in (0, -3, 3, -6, 6):
                    box = angs.get(ang, {}).get(gf + df)
                    if box is not None:
                        break
                if box is None or (box[3] - box[1]) < a.min_h:
                    continue
                if img is None:
                    img = cv2.imread(str(vdir / f"f_{i:05d}.jpg"))
                    if img is None:
                        break
                x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
                x2, y2 = min(img.shape[1], int(box[2])), min(img.shape[0], int(box[3]))
                if x2 - x1 < 20 or y2 - y1 < 50:
                    continue
                name = f"{sid}_{ang}_{gf}.jpg"
                cv2.imwrite(str(out / "crops" / name), img[y1:y2, x1:x2])
                index.append({"sid": sid, "ang": ang, "gf": gf, "path": f"crops/{name}"})
        for p in vdir.glob("*.jpg"):
            p.unlink()
        vdir.rmdir()
    (out / "index.json").write_text(json.dumps(index))
    print(f"total stream crops: {len(index)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
