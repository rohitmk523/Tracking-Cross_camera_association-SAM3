#!/usr/bin/env python3
"""Fetch grid-cell-resolution full-game sources straight off S3 (presigned
ffmpeg transcode on the fly — no full-res copy ever touches disk).

Produces runs/event_demo/fullsrc_{gid8}_{ANG}.mp4 at exactly the renderer's
cell size, source fps. ~250-350 MB per angle for a ~52-min game.

  .venv/bin/python scripts/fetch_fullgame_lowres.py --game 2c490f1a
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")
BUCKET = "uball-videos-production"
CELL_W, CELL_H = 620, 450


def game_prefix(gid8: str) -> str:
    games = json.loads((REPO / "configs/games.json").read_text())
    for g in games.get("working_games", []):
        if g.get("gid8") == gid8 and g.get("s3_prefix"):
            return g["s3_prefix"].rstrip("/")
    raise SystemExit(f"no s3_prefix for {gid8} in configs/games.json")


def presign(key: str) -> str:
    return subprocess.run(
        ["aws", "s3", "presign", f"s3://{BUCKET}/{key}", "--expires-in", "21600"],
        check=True, capture_output=True, text=True).stdout.strip()


def probe_dur(path_or_url: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path_or_url], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def fetch(ang: str, url: str, out: Path, crf: int) -> tuple[str, bool, float]:
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", url,
         "-vf", f"scale={CELL_W}:{CELL_H}", "-r", "30000/1001", "-an",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
         "-movflags", "+faststart", "-y", str(out)], capture_output=True)
    return ang, r.returncode == 0, probe_dur(str(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--crf", type=int, default=26)
    ap.add_argument("--outdir", default="runs/event_demo")
    a = ap.parse_args()
    pfx = game_prefix(a.game)
    date, stem = pfx.split("/")[-2], pfx.split("/")[-1]
    outdir = REPO / a.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    urls = {ang: presign(f"{pfx}/{date}_{stem}_{ang}.mp4") for ang in ANGLES}
    src_dur = probe_dur(urls["FL"])
    print(f"{a.game}: source FL duration {src_dur:.0f}s -> "
          f"{CELL_W}x{CELL_H} crf{a.crf}", flush=True)
    jobs = {}
    with ThreadPoolExecutor(4) as ex:
        for ang in ANGLES:
            out = outdir / f"fullsrc_{a.game}_{ang}.mp4"
            jobs[ang] = ex.submit(fetch, ang, urls[ang], out, a.crf)
    ok = True
    for ang in ANGLES:
        _, rc_ok, dur = jobs[ang].result()
        out = outdir / f"fullsrc_{a.game}_{ang}.mp4"
        sz = out.stat().st_size / 1e6 if out.exists() else 0
        # partial-MP4 trap: a source shorter than expected means the
        # transcode died mid-stream — do not let the renderer consume it
        good = rc_ok and dur >= src_dur - 10
        ok &= good
        print(f"  {ang}: {dur:7.0f}s {sz:6.0f} MB {'OK' if good else 'BAD'}",
              flush=True)
    if not ok:
        print("FETCH INCOMPLETE — do not render", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
