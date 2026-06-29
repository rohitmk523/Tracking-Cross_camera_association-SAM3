#!/usr/bin/env python3
"""Event-anchored frame sampler for ball annotation (docs/08).

Reads a plays export (gid8, angle LEFT/RIGHT, classification, start/end/ts seconds) and,
for each shot, pulls the NEAR camera on that basket (LEFT->NL, RIGHT->NR) and grabs a few
frames across the shot window from S3. At a shot the action is at the camera's own basket,
so the ball is big and clearly annotatable -- exactly the data the ball detector lacks.

  python scripts/sample_event_frames.py --plays data/event_plays_batch1.json \
      --pool data/annotate_pool_events --frames-per-play 4

Then: scripts/prelabel.py --pool data/annotate_pool_events ; scripts/annotate.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUCKET = "uball-videos-production"
REGION = "us-east-1"
NEAR = {"LEFT": "NL", "RIGHT": "NR"}


def _games_index() -> dict[str, tuple[str, str, str]]:
    gj = json.loads((REPO / "configs" / "games.json").read_text())
    idx = {}
    for g in gj["working_games"]:
        pfx = g["s3_prefix"].rstrip("/")                 # court-a/<date>/<full_gid>
        _, date, full = pfx.split("/")
        idx[g["gid8"]] = (pfx, date, full)
    return idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plays", default=str(REPO / "data" / "event_plays_batch1.json"))
    ap.add_argument("--pool", default=str(REPO / "data" / "annotate_pool_events"))
    ap.add_argument("--frames-per-play", type=int, default=4)
    ap.add_argument("--pad", type=float, default=0.3, help="seconds added either side of the window")
    ap.add_argument("--timeout", type=float, default=75.0, help="per-play ffmpeg timeout (skip non-faststart)")
    ap.add_argument("--vf", default="field=top,scale=1920:1080")
    a = ap.parse_args()

    import boto3
    s3 = boto3.client("s3", region_name=REGION)
    gidx = _games_index()
    out_dir = Path(a.pool) / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    plays = json.loads(Path(a.plays).read_text())

    url_cache: dict[tuple[str, str], str] = {}
    slow_games: set[str] = set()                          # non-faststart -> skip over HTTP, use EC2
    total, n_ok = 0, 0
    for i, p in enumerate(plays):
        gid8, cam = p["gid8"], NEAR.get(p["angle"])
        if cam is None or gid8 not in gidx:
            continue
        if gid8 in slow_games:
            print(f"[{i+1}/{len(plays)}] {gid8}/{cam}: skip (game is non-faststart over HTTP)", flush=True)
            continue
        pfx, date, full = gidx[gid8]
        key = (gid8, cam)
        if key not in url_cache:
            s3key = f"{pfx}/{date}_{full}_{cam}.mp4"
            url_cache[key] = s3.generate_presigned_url(
                "get_object", Params={"Bucket": BUCKET, "Key": s3key}, ExpiresIn=7200)
        url = url_cache[key]
        start = max(0.0, float(p["start"]) - a.pad)
        dur = max(1.0, float(p["end"]) - float(p["start"]) + 2 * a.pad)
        n = a.frames_per_play
        stem = f"{gid8}_{cam}_{p['classification']}_t{int(round(float(p['ts'])))}"
        pat = str(out_dir / f"{stem}_f%02d.jpg")
        cmd = ["ffmpeg", "-nostdin", "-y",
               "-reconnect", "1", "-reconnect_streamed", "1",
               "-reconnect_on_network_error", "1", "-reconnect_on_http_error", "5xx",
               "-reconnect_delay_max", "10",
               "-ss", f"{start:.2f}", "-i", url, "-t", f"{dur:.2f}",
               "-vf", f"{a.vf},fps={n}/{dur:.2f}", "-frames:v", str(n), "-q:v", "2", pat]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
        except subprocess.TimeoutExpired:
            slow_games.add(gid8)                          # non-faststart mp4 over HTTP -> skip rest of game
        got = len(list(out_dir.glob(f"{stem}_f*.jpg")))
        total += got
        n_ok += 1 if got else 0
        flag = "" if got else "  (timeout/skip -- likely non-faststart)"
        print(f"[{i+1}/{len(plays)}] {gid8}/{cam} {p['classification']} @t{p['ts']}: {got} frames{flag}", flush=True)

    print(f"\nEVENT-ANCHORED: {total} frames from {n_ok}/{len(plays)} plays -> {out_dir}", flush=True)
    print("next: scripts/prelabel.py --pool", a.pool, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
