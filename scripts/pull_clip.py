#!/usr/bin/env python3
"""Pull a short camera clip from S3 (ranged slice) for local tracking/eval.

  python scripts/pull_clip.py --gid8 e6fba750 --angle FL --start 47 --dur 12

Resolves the s3_key from configs/games.json, presigns a GET URL, and ffmpeg-slices
[start, start+dur] via HTTP range (downloads only the window, not the multi-GB file).
Deinterlaces (field=top) + scales to 1920x1080 to match the detector's training frames
(docs/14 demo_clip_window vf). Needs valid (rotated) AWS creds in the env / ~/.aws.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUCKET = "uball-videos-production"
REGION = "us-east-1"


def s3_key(gid8: str, angle: str) -> str:
    gj = json.loads((REPO / "configs" / "games.json").read_text())
    g = next((x for x in gj["working_games"] if x["gid8"] == gid8), None)
    if not g:
        raise SystemExit(f"unknown gid8 {gid8} (not in configs/games.json working_games)")
    pfx = g["s3_prefix"].rstrip("/")                 # court-a/<date>/<full_gid>
    _, date, full = pfx.split("/")
    return f"{pfx}/{date}_{full}_{angle}.mp4"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid8", required=True)
    ap.add_argument("--angle", required=True, choices=("FL", "FR", "NL", "NR"))
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--dur", type=float, default=12.0)
    ap.add_argument("--vf", default="field=top,scale=1920:1080")
    ap.add_argument("--no-audio", action="store_true", help="drop audio (smaller; breaks audio-sync)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import boto3

    key = s3_key(a.gid8, a.angle)
    url = boto3.client("s3", region_name=REGION).generate_presigned_url(
        "get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=3600)
    out = Path(a.out) if a.out else (
        REPO / "data" / "clips" / f"{a.gid8}_{a.angle}_{int(a.start)}_{int(a.dur)}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-nostdin", "-y",
           "-reconnect", "1", "-reconnect_streamed", "1",
           "-reconnect_on_network_error", "1", "-reconnect_on_http_error", "5xx",
           "-reconnect_delay_max", "10",
           "-ss", str(a.start), "-i", url, "-t", str(a.dur)]
    if a.vf:
        cmd += ["-vf", a.vf]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    cmd += ["-an"] if a.no_audio else ["-c:a", "aac", "-ac", "1"]   # keep mono audio for sync
    cmd += [str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if p.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        raise SystemExit(f"ffmpeg failed:\n{p.stderr[-1800:]}")
    print(f"clip -> {out}  ({out.stat().st_size / 1e6:.1f} MB)  key={key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
