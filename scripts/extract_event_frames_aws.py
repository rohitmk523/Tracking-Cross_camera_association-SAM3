#!/usr/bin/env python3
"""Event-anchored frame extraction IN-REGION on EC2 (S3->EC2 is fast + faststart-agnostic,
so it doesn't hang the way laptop ffmpeg-over-HTTP does on non-faststart mp4s).

For each shot in a plays export, grabs frames across the shot window from the NEAR camera on
that basket (LEFT->NL, RIGHT->NR) -- where the ball is big and clearly annotatable. Reuses the
sample_frames_aws safety pattern: presigned manifest, 12h session creds on the box, per-video
upload (a late hang can't lose work), and a hard self-terminate.

  python scripts/extract_event_frames_aws.py --plays data/event_plays_all.json --dry-run
  python scripts/extract_event_frames_aws.py --plays data/event_plays_all.json --accept-unrotated-creds
  python scripts/extract_event_frames_aws.py --fetch --pool data/annotate_pool_events
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import boto3

REPO = Path(__file__).resolve().parents[1]
BUCKET = "uball-videos-production"
REGION = "us-east-1"
PREFIX = "_tmp_event_extract"
AMI = "ami-012ba162b9cd2729c"          # DL AMI (Ubuntu 22.04; curl + apt ffmpeg)
FLAGGED = "840102831548"
NEAR = {"LEFT": "NL", "RIGHT": "NR"}


def _games_index() -> dict[str, tuple[str, str, str]]:
    gj = json.loads((REPO / "configs" / "games.json").read_text())
    idx = {}
    for g in gj["working_games"]:
        pfx = g["s3_prefix"].rstrip("/")
        _, date, full = pfx.split("/")
        idx[g["gid8"]] = (pfx, date, full)
    return idx


def _manifest(plays: list[dict], n_frames: int, pad: float) -> tuple[str, str]:
    """Build (manifest, videos) text. manifest line: gid|cam|key|ss|dur|n|cls|ts."""
    gidx = _games_index()
    lines, vids = [], {}
    for p in plays:
        cam = NEAR.get(p["angle"])
        if cam is None or p["gid8"] not in gidx:
            continue
        pfx, date, full = gidx[p["gid8"]]
        key = f"{pfx}/{date}_{full}_{cam}.mp4"
        ss = max(0.0, float(p["start"]) - pad)
        dur = max(1.0, float(p["end"]) - float(p["start"]) + 2 * pad)
        ts = int(round(float(p["ts"])))
        lines.append(f"{p['gid8']}|{cam}|{key}|{ss:.2f}|{dur:.2f}|{n_frames}|{p['classification']}|{ts}")
        vids[(p["gid8"], cam)] = key
    lines.sort()
    vid_lines = [f"{g}|{c}|{k}" for (g, c), k in sorted(vids.items())]
    return "\n".join(lines), "\n".join(vid_lines)


def _userdata(creds: dict, par: int, hard_limit: int) -> str:
    return f"""#!/bin/bash
exec > /var/log/extract.log 2>&1
export AWS_ACCESS_KEY_ID="{creds['AccessKeyId']}"
export AWS_SECRET_ACCESS_KEY="{creds['SecretAccessKey']}"
export AWS_SESSION_TOKEN="{creds['SessionToken']}"
export AWS_DEFAULT_REGION="{REGION}"
B={BUCKET}; P={PREFIX}
(sleep {hard_limit}; shutdown -h now) &                       # hard safety net
(while true; do sleep 20; aws s3 cp /var/log/extract.log s3://$B/$P/extract.log >/dev/null 2>&1 || true; done) &
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get install -y ffmpeg
mkdir -p /work/frames && cd /work
aws s3 cp s3://$B/$P/manifest.txt manifest.txt
aws s3 cp s3://$B/$P/videos.txt videos.txt
echo "[extract] $(wc -l < videos.txt) videos, $(wc -l < manifest.txt) shots, PAR={par}"
T0=$(date +%s)
one() {{
  local gid="$1" cam="$2" key="$3" v="/work/$1_$2.mp4"
  timeout 600 aws s3 cp "s3://$B/$key" "$v" --only-show-errors || {{ echo "[extract] dl FAIL $gid/$cam"; rm -f "$v"; return; }}
  grep "^${{gid}}|${{cam}}|" manifest.txt | while IFS='|' read -r g c k ss dur n cls ts; do
    timeout 30 ffmpeg -nostdin -y -ss "$ss" -i "$v" -t "$dur" \
      -vf "field=top,scale=1920:1080,fps=${{n}}/${{dur}}" -frames:v "$n" -q:v 2 \
      "/work/frames/${{g}}_${{c}}_${{cls}}_t${{ts}}_f%02d.jpg" 2>/dev/null || true
  done
  rm -f "$v"
  aws s3 cp /work/frames/ s3://$B/$P/pool/ --recursive --exclude "*" --include "${{gid}}_${{cam}}_*.jpg" --only-show-errors
  echo "[extract] $gid/$cam done -> $(ls /work/frames/${{gid}}_${{cam}}_*.jpg 2>/dev/null|wc -l) frames (t+$(($(date +%s)-T0))s)"
}}
while IFS='|' read -r gid cam key; do
  [ -z "$gid" ] && continue
  one "$gid" "$cam" "$key" &
  while [ $(jobs -r | wc -l) -ge {par} ]; do sleep 0.5; done
done < videos.txt
wait
echo "[extract] DONE total $(ls /work/frames/*.jpg 2>/dev/null|wc -l) frames in $(($(date +%s)-T0))s"
aws s3 cp /var/log/extract.log s3://$B/$P/extract.log >/dev/null 2>&1 || true
shutdown -h now
"""


def launch(a, plays: list[dict]) -> None:
    flagged_ok = a.accept_unrotated_creds or os.environ.get("UBALL_AWS_CREDS_ROTATED") == "1"
    acct = boto3.client("sts", region_name=REGION).get_caller_identity().get("Account")
    print(f"AWS account: {acct}")
    if acct == FLAGGED and not flagged_ok:
        sys.exit("flagged account -- pass --accept-unrotated-creds (rotation still owed).")

    manifest, videos = _manifest(plays, a.frames_per_play, a.pad)
    n_shots, n_vids = len(manifest.splitlines()), len(videos.splitlines())
    s3 = boto3.client("s3", region_name=REGION)
    s3.put_object(Bucket=BUCKET, Key=f"{PREFIX}/manifest.txt", Body=manifest.encode())
    s3.put_object(Bucket=BUCKET, Key=f"{PREFIX}/videos.txt", Body=videos.encode())
    creds = boto3.client("sts", region_name=REGION).get_session_token(
        DurationSeconds=43200)["Credentials"]
    ud = _userdata(creds, a.par, a.hard_limit)

    ec2 = boto3.client("ec2", region_name=REGION)
    r = ec2.run_instances(
        ImageId=AMI, InstanceType=a.instance_type, MinCount=1, MaxCount=1,
        InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                              "Ebs": {"VolumeSize": 120, "VolumeType": "gp3",
                                      "Throughput": 750, "Iops": 6000,
                                      "DeleteOnTermination": True}}],
        UserData=ud,
        TagSpecifications=[{"ResourceType": "instance",
                            "Tags": [{"Key": "Name", "Value": "uball-event-extract"}]}])
    iid = r["Instances"][0]["InstanceId"]
    print(f"launched {iid} ({a.instance_type}): {n_shots} shots across {n_vids} videos "
          f"x {a.frames_per_play} frames = ~{n_shots * a.frames_per_play} frames")
    print(f"watch:  aws s3 cp s3://{BUCKET}/{PREFIX}/extract.log -")
    print(f"frames land at: s3://{BUCKET}/{PREFIX}/pool/   (then: --fetch)")


def dry_run(a, plays: list[dict]) -> None:
    manifest, videos = _manifest(plays, a.frames_per_play, a.pad)
    n_shots, n_vids = len(manifest.splitlines()), len(videos.splitlines())
    print(f"DRY RUN -- {n_shots} shots across {n_vids} (game,cam) videos "
          f"x {a.frames_per_play} frames = ~{n_shots * a.frames_per_play} frames")
    print(f"  instance={a.instance_type}  par={a.par}  hard_limit={a.hard_limit}s  EBS=120GB")
    for line in manifest.splitlines()[:5]:
        print("   ", line)
    print(f"   ... ({n_shots} shots)")


def fetch(a) -> None:
    out = Path(a.pool) / "images"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", f"s3://{BUCKET}/{PREFIX}/pool/", str(out),
                    "--only-show-errors"], check=False)
    n = len(list(out.glob("*.jpg")))
    print(f"synced {n} frames -> {out}")
    print("next: scripts/prelabel.py --pool", a.pool)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plays", default=str(REPO / "data" / "event_plays_all.json"))
    ap.add_argument("--pool", default=str(REPO / "data" / "annotate_pool_events"))
    ap.add_argument("--frames-per-play", type=int, default=4)
    ap.add_argument("--pad", type=float, default=0.3)
    ap.add_argument("--instance-type", default="c5n.2xlarge")
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--hard-limit", type=int, default=2700, help="seconds; instance self-terminates")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--accept-unrotated-creds", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch(a)
        return 0
    plays = json.loads(Path(a.plays).read_text())
    if a.dry_run:
        dry_run(a, plays)
    else:
        launch(a, plays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
