#!/usr/bin/env python3
"""Extract the annotation-pool frames on an IN-REGION AWS instance (S3->EC2 is
fast/free), then download only the ~450MB of JPGs -- NOT 144GB of video. Laptop
S3 download is too slow + the mp4s are non-faststart (ffmpeg-over-HTTP hangs).

Creds-free: the instance only uses PRESIGNED URLs (GET videos + manifest, PUT the
result tarball) -- no AWS keys ever land on the box.

  python scripts/sample_frames_aws.py --dry-run            # show plan, no AWS
  python scripts/sample_frames_aws.py --accept-unrotated-creds   # launch
  python scripts/sample_frames_aws.py --fetch              # pull + unpack frames
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import boto3
import yaml

REPO = Path(__file__).resolve().parents[1]
BUCKET = "uball-videos-production"
REGION = "us-east-1"
PREFIX = "_tmp_annotate_extract"
AMI = "ami-012ba162b9cd2729c"          # DL AMI (Ubuntu 22.04; has curl + apt ffmpeg)
FLAGGED = "840102831548"


def _video_jobs(cfg: dict) -> list[tuple[str, str, str]]:
    """(gid8, angle, s3_key) for every game x angle in the sampling config."""
    import json
    gj = json.loads((REPO / "configs" / "games.json").read_text())
    gidx = {}
    for g in gj["working_games"]:
        pfx = g["s3_prefix"]
        parts = pfx.rstrip("/").split("/")          # court-a/<date>/<full_gid>
        gidx[g["gid8"]] = (pfx, parts[1], parts[2])
    jobs = []
    for gid8 in cfg["games"]:
        if gid8 not in gidx:
            continue
        pfx, date, full = gidx[gid8]
        for ang in cfg["angles"]:
            jobs.append((gid8, ang, f"{pfx}{date}_{full}_{ang}.mp4"))
    return jobs


def _s3():
    return boto3.client("s3", region_name=REGION)


def _presign(s3, method, key, exp=28800):
    op = "get_object" if method == "get" else "put_object"
    return s3.generate_presigned_url(op, Params={"Bucket": BUCKET, "Key": key},
                                     ExpiresIn=exp)


def _userdata(manifest_url, result_put_url, log_put_url, n, window, par=8) -> str:
    step = max(1, window // n)
    return f"""#!/bin/bash
exec > /var/log/extract.log 2>&1
LOG_URL="{log_put_url}"
(while true; do sleep 15; curl -s -T /var/log/extract.log "$LOG_URL" >/dev/null 2>&1 || true; done) &
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get install -y ffmpeg
mkdir -p /work/frames && cd /work
curl -s -L "{manifest_url}" -o manifest.txt
echo "[extract] $(wc -l < manifest.txt) videos; n={n} step={step}s; PARALLEL={par}"
T0=$(date +%s)
one() {{
  local game="$1" angle="$2" url="$3" v="/work/$1_$2.mp4"
  curl -s -L --connect-timeout 20 --max-time 240 --retry 1 --retry-max-time 300 \
    "$url" -o "$v" || {{ echo "[extract] dl FAIL $game/$angle (skipped)"; rm -f "$v"; return; }}
  for k in $(seq 0 $(({n}-1))); do
    ffmpeg -nostdin -y -ss $((k*{step})) -i "$v" -frames:v 1 -q:v 2 \
      "/work/frames/$1_$2_f$(printf %03d $k).jpg" 2>/dev/null
  done
  rm -f "$v"
  echo "[extract] $game/$angle done -> $(ls /work/frames/$1_$2_f*.jpg 2>/dev/null|wc -l) (t+$(($(date +%s)-T0))s)"
}}
while IFS='|' read -r game angle url; do
  [ -z "$game" ] && continue
  one "$game" "$angle" "$url" &
  while [ $(jobs -r | wc -l) -ge {par} ]; do sleep 0.5; done
done < manifest.txt
wait
echo "[extract] DONE total $(ls /work/frames/*.jpg 2>/dev/null|wc -l) frames in $(($(date +%s)-T0))s"
tar czf frames.tar.gz -C frames .
for try in 1 2 3; do
  CODE=$(curl -sS --max-time 1800 -w '%{{http_code}}' -o /dev/null -T frames.tar.gz "{result_put_url}")
  echo "[extract] upload try $try http=$CODE"; [ "$CODE" = "200" ] && break; sleep 10
done
curl -s -T /var/log/extract.log "$LOG_URL" >/dev/null 2>&1 || true
sleep 5; shutdown -h now
"""


def launch(a, cfg) -> None:
    flagged_ok = a.accept_unrotated_creds or os.environ.get("UBALL_AWS_CREDS_ROTATED") == "1"
    s3 = _s3()
    acct = boto3.client("sts", region_name=REGION).get_caller_identity().get("Account")
    print(f"AWS account: {acct}")
    if acct == FLAGGED and not flagged_ok:
        sys.exit("flagged account -- pass --accept-unrotated-creds (rotation still owed).")

    jobs = _video_jobs(cfg)
    n, window = int(cfg["n_per_game_angle"]), int(cfg.get("window_sec", 480))
    # manifest: game|angle|presigned_video_url  (one line per video)
    lines = [f"{g}|{ang}|{_presign(s3, 'get', key)}" for g, ang, key in jobs]
    man_key = f"{PREFIX}/manifest.txt"
    s3.put_object(Bucket=BUCKET, Key=man_key, Body="\n".join(lines).encode())
    result_key = f"{PREFIX}/frames.tar.gz"
    log_key = f"{PREFIX}/extract.log"
    ud = _userdata(_presign(s3, "get", man_key), _presign(s3, "put", result_key),
                   _presign(s3, "put", log_key), n, window)

    ec2 = boto3.client("ec2", region_name=REGION)
    r = ec2.run_instances(
        ImageId=AMI, InstanceType=a.instance_type, MinCount=1, MaxCount=1,
        InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                              "Ebs": {"VolumeSize": 60, "VolumeType": "gp3",
                                      "Throughput": 750, "Iops": 6000,
                                      "DeleteOnTermination": True}}],
        UserData=ud,
        TagSpecifications=[{"ResourceType": "instance",
                            "Tags": [{"Key": "Name", "Value": "uball-frame-extract"}]}])
    iid = r["Instances"][0]["InstanceId"]
    print(f"launched {iid} ({a.instance_type})  {len(jobs)} videos x {n} frames "
          f"= ~{len(jobs)*n} frames")
    print(f"watch:  aws s3 cp s3://{BUCKET}/{log_key} -")
    print(f"result lands: s3://{BUCKET}/{result_key}  (then run --fetch)")


def dry_run(cfg) -> None:
    jobs = _video_jobs(cfg)
    n, window = int(cfg["n_per_game_angle"]), int(cfg.get("window_sec", 480))
    print(f"DRY RUN -- {len(jobs)} videos x {n} frames = ~{len(jobs)*n} frames")
    print(f"  window={window}s  instance=c5.2xlarge  creds-free (presigned URLs)")
    for g, ang, key in jobs[:6]:
        print(f"    {g}/{ang}: {key}")
    print(f"    ... ({len(jobs)} total)")


def fetch(cfg) -> None:
    s3 = _s3()
    out_imgs = REPO / cfg["out_dir"] / "images"
    out_imgs.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp()) / "frames.tar.gz"
    s3.download_file(BUCKET, f"{PREFIX}/frames.tar.gz", str(tmp))
    with tarfile.open(tmp) as t:
        t.extractall(out_imgs)                       # noqa: S202 (our own tarball)
    n = len(list(out_imgs.glob("*.jpg")))
    print(f"unpacked {n} frames -> {out_imgs}")
    print("next: scripts/prelabel.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "annotation_sampling.yaml"))
    ap.add_argument("--instance-type", default="c5n.2xlarge")   # 25 Gbps network
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--accept-unrotated-creds", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.dry_run:
        dry_run(cfg)
    elif a.fetch:
        fetch(cfg)
    else:
        launch(a, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
