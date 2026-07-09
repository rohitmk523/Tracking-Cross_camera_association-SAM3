#!/usr/bin/env python3
"""FULL-GAME SAM3 single-object tracking for ONE player — one instance per camera.

Per camera: the instance downloads the full game video straight from S3 (in-region),
transcodes with the SAME vf as pull_clip.py (field=top,scale=1920:1080 — coordinates
match calib + seeds), then propagates the player's mask from his earliest confident
jersey read to the end of the game. 4 instances run in parallel: ~56 min of video
each, ~7-8h wall-clock, self-terminating.

Seed frames come from the window jerseyseeds (clip-relative) mapped onto the full
video timeline: full_frame = round(clip_start_s * fps) + clip_seed_frame.

  export UBALL_AWS_CREDS_ROTATED=1
  python scripts/aws_sam3_fullgame_job.py --game e6fba750 --player "#11" \
      --seeds-tag 44_60 --clip-start 44 --i-rotated-creds
  python scripts/aws_sam3_fullgame_job.py --fetch --game e6fba750 --player "#11"
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import aws_sam3_job as J   # noqa: E402

ANGLES = ("FL", "FR", "NL", "NR")
FPS = 29.97
BUCKET_VIDEOS = "uball-videos-production"


def video_key(game: str, angle: str) -> str:
    gj = json.loads((REPO / "configs/games.json").read_text())
    g = next(x for x in gj["working_games"] if x["gid8"] == game)
    pfx = g["s3_prefix"].rstrip("/")
    _, date, full = pfx.split("/")
    return f"{pfx}/{date}_{full}_{angle}.mp4"


def userdata(video_url, weights_url, script_url, results_url, log_url, cmd: str) -> str:
    return f"""#!/bin/bash
exec > /var/log/sam3.log 2>&1
export HOME=/root PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive
LOG_URL="{log_url}"
(while true; do sleep 30; curl -s -T /var/log/sam3.log "$LOG_URL" >/dev/null 2>&1 || true; done) &
PYBIN=""
for P in /opt/pytorch/bin/python /usr/bin/python3; do
  if [ -x "$P" ] && $P -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then PYBIN=$P; break; fi
done
[ -z "$PYBIN" ] && PYBIN=/usr/bin/python3
$PYBIN -m pip install -q -U ultralytics opencv-python-headless
which ffmpeg || apt-get update -qq && apt-get install -y -qq ffmpeg
mkdir -p /work && cd /work
echo "[boot] downloading full video (in-region)..."
curl -s -L "{video_url}" -o raw.mp4 && ls -la raw.mp4
curl -s -L "{script_url}" -o sam3_track_player.py
curl -s -L "{weights_url}" -o sam3.pt && ls -la sam3.pt
echo "[boot] transcoding (deinterlace+scale, matches pull_clip vf)..."
ffmpeg -nostdin -y -i raw.mp4 -vf "field=top,scale=1920:1080" \\
  -c:v libx264 -preset veryfast -crf 23 -an full.mp4 && rm raw.mp4
ls -la full.mp4
mkdir -p out
RC=0
{cmd}
tar czf results.tar.gz -C out .
CODE=$(curl -sS --max-time 1800 -w '%{{http_code}}' -o /dev/null -T results.tar.gz "{results_url}")
echo "[boot] results upload http=$CODE rc=$RC"
curl -s -T /var/log/sam3.log "$LOG_URL" >/dev/null 2>&1 || true
sleep 5; shutdown -h now
"""


def launch(a) -> None:
    import boto3
    aws = J._aws_cfg()
    J._guard_rotation(aws, a)
    region, bucket = aws.get("region", "us-east-1"), aws.get("s3_bucket")
    s3 = boto3.client("s3", region_name=region)
    J._ensure_weights(s3, bucket)

    seeds_path = REPO / f"runs/anchors/{a.game}_{a.seeds_tag}.jerseyseeds.json"
    cams = json.loads(seeds_path.read_text())["seeds"][a.player]
    safe = a.player.replace("#", "n").replace(" ", "")
    script_key = f"{J.PREFIX}/sam3_track_player.py"
    s3.upload_file(str(REPO / "scripts/sam3_track_player.py"), bucket, script_key)

    def presign(op, key, exp, buck=None):
        return s3.generate_presigned_url("get_object" if op == "get" else "put_object",
                                         Params={"Bucket": buck or bucket, "Key": key},
                                         ExpiresIn=exp)

    ec2 = boto3.client("ec2", region_name=region)
    base = round(a.clip_start * FPS)
    for cam in ANGLES:
        if cam not in cams:
            print(f"{cam}: no seed for {a.player} — skipped")
            continue
        d = cams[cam]
        full_seed = base + d["seed_frame"]
        box = ",".join(str(v) for v in d["seed_box"])
        tag = f"{safe}full_{cam}"
        outp = f"out/{a.game}_full__{safe}__{cam}.json"
        cmd = (f'echo "=== {a.player} {cam} FULL GAME (seed frame {full_seed}) ==="; '
               f'$PYBIN sam3_track_player.py --video full.mp4 --seed-frame {full_seed} '
               f'--seed-box "{box}" --player "{a.player}" --cam {cam} --out "{outp}" '
               f'--weights /work/sam3.pt || RC=1')
        ud = userdata(
            presign("get", video_key(a.game, cam), 86400, BUCKET_VIDEOS),
            presign("get", J.WEIGHTS_KEY, 86400),
            presign("get", script_key, 86400),
            presign("put", J.results_key(tag), 172800),
            presign("put", J.log_key(tag), 172800), cmd)
        r = ec2.run_instances(
            ImageId=aws.get("ami"), InstanceType=aws.get("instance_type", "g5.2xlarge"),
            MinCount=1, MaxCount=1, InstanceInitiatedShutdownBehavior="terminate",
            BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                                  "Ebs": {"VolumeSize": 120, "VolumeType": "gp3",
                                          "DeleteOnTermination": True}}],
            UserData=ud,
            TagSpecifications=[{"ResourceType": "instance",
                                "Tags": [{"Key": "Name", "Value": f"uball-sam3-full-{cam}"}]}])
        iid = r["Instances"][0]["InstanceId"]
        print(f"{cam}: launched {iid} (seed frame {full_seed}) -> s3 tag {tag}")
    print(f"logs/results under s3://{bucket}/{J.PREFIX}/sam3_{safe}full_<CAM>[.log|_results.tar.gz]")


def fetch(a) -> None:
    import boto3
    aws = J._aws_cfg()
    region, bucket = aws.get("region", "us-east-1"), aws.get("s3_bucket")
    s3 = boto3.client("s3", region_name=region)
    safe = a.player.replace("#", "n").replace(" ", "")
    out = REPO / "runs/sam3_players_fullgame"
    out.mkdir(parents=True, exist_ok=True)
    got = []
    for cam in ANGLES:
        tag = f"{safe}full_{cam}"
        tarp = out / f"results_{cam}.tar.gz"
        try:
            s3.download_file(bucket, J.results_key(tag), str(tarp))
        except Exception:
            print(f"{cam}: not ready")
            continue
        with tarfile.open(tarp) as t:
            t.extractall(out)
        got.append(cam)
    print(f"fetched cams: {got} -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="e6fba750")
    ap.add_argument("--player", default="#11")
    ap.add_argument("--seeds-tag", default="44_60", help="window whose jerseyseeds seed the run")
    ap.add_argument("--clip-start", type=float, default=44.0, help="window start second on the game video")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--i-rotated-creds", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch(a)
    else:
        launch(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
