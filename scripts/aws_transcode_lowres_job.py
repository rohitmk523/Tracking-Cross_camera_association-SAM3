#!/usr/bin/env python3
"""In-region low-res transcode of full-game videos for the review renderer.

One CPU instance reads each game/angle straight off S3 (presigned, in-region
= fast), transcodes to the renderer's cell size (620x450, source fps), and
PUTs each fullsrc file back to S3. Local download then uses `aws s3 cp`
(multipart) instead of long single HTTPS streams that get throttled to
~1 MB/min off-region. ~16 files, c6i.2xlarge ~1-1.5h ~= $0.5.

  export UBALL_AWS_CREDS_ROTATED=1
  python scripts/aws_transcode_lowres_job.py --games e6fba750,c2a354fe,2c490f1a,13e1ffad --i-rotated-creds
  python scripts/aws_transcode_lowres_job.py --fetch --games e6fba750,c2a354fe,2c490f1a,13e1ffad
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import aws_sam3_job as J   # noqa: E402

ANGLES = ("FL", "FR", "NL", "NR")
CELL_W, CELL_H = 620, 450


def game_prefix(gid8: str) -> str:
    games = json.loads((REPO / "configs/games.json").read_text())
    for g in games.get("working_games", []):
        if g.get("gid8") == gid8 and g.get("s3_prefix"):
            return g["s3_prefix"].rstrip("/")
    raise SystemExit(f"no s3_prefix for {gid8}")


def out_key(gid8: str, ang: str) -> str:
    return f"{J.PREFIX}/fullsrc/{gid8}_{ang}.mp4"


def userdata(jobs, log_url) -> str:
    # groups of 3 concurrent transcodes; NEVER a bare `wait` (the infinite
    # log-uploader loop would hang it — the documented $3.4 trap)
    lines = []
    for i, (src_url, put_url, name) in enumerate(jobs):
        lines.append(
            f'(ffmpeg -hide_banner -loglevel error -i "{src_url}" '
            f'-vf scale={CELL_W}:{CELL_H} -r 30000/1001 -an -c:v libx264 '
            f'-preset veryfast -crf 26 -movflags +faststart -y /work/{name} '
            f'&& curl -sS -o /dev/null -T /work/{name} "{put_url}" '
            f'&& echo "[done] {name}" && rm -f /work/{name}'
            f' || echo "[FAIL] {name}") &\nG="$G $!"\n'
            + ('wait $G\nG=""\n' if i % 3 == 2 else ""))
    body = "".join(lines)
    return f"""#!/bin/bash
exec > /var/log/txc.log 2>&1
export HOME=/root
LOG_URL="{log_url}"
(while true; do sleep 30; curl -s -T /var/log/txc.log "$LOG_URL" >/dev/null 2>&1 || true; done) &
(sleep 10800; echo "[boot] 3h failsafe"; shutdown -h now) &
which ffmpeg || (apt-get update -qq && apt-get install -y -qq ffmpeg)
mkdir -p /work
G=""
{body}
[ -n "$G" ] && wait $G
echo "[boot] all transcodes done"
curl -s -T /var/log/txc.log "$LOG_URL" >/dev/null 2>&1 || true
sleep 5; shutdown -h now
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True, help="comma gid8 list")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--instance-type", default="c6i.2xlarge")
    ap.add_argument("--i-rotated-creds", action="store_true")
    a = ap.parse_args()
    gids = [g.strip() for g in a.games.split(",") if g.strip()]
    import boto3
    aws = J._aws_cfg()
    region, bucket = aws.get("region", "us-east-1"), aws.get("s3_bucket")
    s3 = boto3.client("s3", region_name=region)

    if a.fetch:
        outdir = REPO / "runs/event_demo"
        ok = True
        for gid in gids:
            for ang in ANGLES:
                dst = outdir / f"fullsrc_{gid}_{ang}.mp4"
                r = subprocess.run(["aws", "s3", "cp",
                                    f"s3://{bucket}/{out_key(gid, ang)}", str(dst)],
                                   capture_output=True, text=True)
                sz = dst.stat().st_size / 1e6 if dst.exists() else 0
                print(f"  {gid} {ang}: {'OK' if r.returncode == 0 else 'MISSING'} "
                      f"{sz:5.0f} MB")
                ok &= r.returncode == 0
        return 0 if ok else 1

    J._guard_rotation(aws, a)

    def presign(op, key, exp=43200):
        return s3.generate_presigned_url(
            "get_object" if op == "get" else "put_object",
            Params={"Bucket": bucket, "Key": key}, ExpiresIn=exp)

    jobs = []
    for gid in gids:
        pfx = game_prefix(gid)
        date, stem = pfx.split("/")[-2], pfx.split("/")[-1]
        for ang in ANGLES:
            src = presign("get", f"{pfx}/{date}_{stem}_{ang}.mp4")
            put = presign("put", out_key(gid, ang))
            jobs.append((src, put, f"{gid}_{ang}.mp4"))
    log_url = presign("put", J.log_key("txc_lowres"), 86400)
    ud = userdata(jobs, log_url)
    ec2 = boto3.client("ec2", region_name=region)
    r = ec2.run_instances(
        ImageId=aws.get("ami"), InstanceType=a.instance_type,
        MinCount=1, MaxCount=1, InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                              "Ebs": {"VolumeSize": 60, "VolumeType": "gp3",
                                      "DeleteOnTermination": True}}],
        UserData=ud,
        TagSpecifications=[{"ResourceType": "instance",
                            "Tags": [{"Key": "Name", "Value": "uball-txc-lowres"}]}])
    iid = r["Instances"][0]["InstanceId"]
    print(f"instance {iid}: {len(jobs)} transcodes "
          f"({len(gids)} games), 3h failsafe")
    print(f"log: s3://{bucket}/{J.log_key('txc_lowres')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
