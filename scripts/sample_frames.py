#!/usr/bin/env python3
"""Sample frames from S3 game videos into the annotation pool (Model A: player/
referee/ball). Gap-weighted across NEW games x 4 angles; frames stream-extracted
from S3 via presigned URL + ffmpeg seek (no full downloads). OUR FOOTAGE ONLY.

  python scripts/sample_frames.py --test          # tiny batch to eyeball
  python scripts/sample_frames.py                  # full ~1500-frame pool

Needs AWS creds (set -a; source .env). Output: data/annotate_pool/images/<stem>.jpg
with stem = <gid8>_<ANGLE>_t<sec> (provenance-parseable).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
BUCKET = "uball-videos-production"


def _games_index(games_json: Path) -> dict[str, dict]:
    """gid8 -> {date, full_gid, s3_prefix} from configs/games.json."""
    data = json.loads(games_json.read_text())
    out = {}
    for g in data.get("working_games", []):
        pfx = g["s3_prefix"].rstrip("/")              # court-a/<date>/<full_gid>
        parts = pfx.split("/")
        out[g["gid8"]] = {"date": parts[1], "full_gid": parts[2], "s3_prefix": g["s3_prefix"]}
    return out


def _presign(key: str, expires=7200) -> str:
    r = subprocess.run(["aws", "s3", "presign", f"s3://{BUCKET}/{key}",
                        "--expires-in", str(expires)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"presign failed for {key}: {r.stderr.strip()}")
    return r.stdout.strip()


def _extract_window(url: str, window: int, n: int, out_dir: Path,
                    gid8: str, ang: str) -> int:
    """Extract n frames spread across the first `window` seconds in ONE pass.

    Byte-copy the window (fast, no decode, no per-frame remote seek) to a temp
    file, then decode it locally to n evenly-spaced JPGs. ~1 remote op per video
    instead of n -- orders of magnitude faster than per-frame seeking.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "w.mp4"
        grab = subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", url, "-t", str(window),
             "-an", "-c", "copy", str(tmp)],
            capture_output=True, text=True)
        if grab.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            return 0
        pat = str(out_dir / f"{gid8}_{ang}_f%03d.jpg")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", str(tmp),
             "-vf", f"fps={n}/{window}", "-frames:v", str(n), "-q:v", "2", pat],
            capture_output=True, text=True)
    return len(list(out_dir.glob(f"{gid8}_{ang}_f*.jpg")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "annotation_sampling.yaml"))
    ap.add_argument("--games", default=str(REPO / "configs" / "games.json"))
    ap.add_argument("--test", action="store_true",
                    help="tiny batch: 2 games x 4 angles x 3 frames")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    gidx = _games_index(Path(a.games))
    out_dir = REPO / cfg["out_dir"] / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    games = list(cfg["games"])
    n = int(cfg["n_per_game_angle"])
    window = int(cfg.get("window_sec", 480))
    if a.test:
        games, n = games[:2], 6

    ok = 0
    for gid8 in games:
        gi = gidx.get(gid8)
        if not gi:
            print(f"SKIP {gid8}: not in games.json")
            continue
        for ang in cfg["angles"]:
            key = f"{gi['s3_prefix']}{gi['date']}_{gi['full_gid']}_{ang}.mp4"
            try:
                url = _presign(key)
            except RuntimeError as e:
                print(f"  {gid8}/{ang}: {e}")
                continue
            got = _extract_window(url, window, n, out_dir, gid8, ang)
            ok += got
            print(f"  {gid8}/{ang}: {got} frames")
    print(f"\nSAMPLED {ok} frames -> {out_dir}")
    print("next: scripts/prelabel.py to pre-fill player/referee/ball boxes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
