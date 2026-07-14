#!/usr/bin/env python3
"""In-region audio-track extraction (stream copy, no re-encode) so the proven
uball_cc audio sync can run on FULL-GAME audio locally. ~$0.05.

  export UBALL_AWS_CREDS_ROTATED=1
  python scripts/aws_audiograb_job.py --games c2a354fe,13e1ffad --i-rotated-creds
  python scripts/aws_audiograb_job.py --fetch --games c2a354fe,13e1ffad
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


def game_prefix(gid8: str) -> str:
    games = json.loads((REPO / "configs/games.json").read_text())
    for g in games.get("working_games", []):
        if g.get("gid8") == gid8 and g.get("s3_prefix"):
            return g["s3_prefix"].rstrip("/")
    raise SystemExit(f"no s3_prefix for {gid8}")


def out_key(gid8: str, ang: str) -> str:
    return f"{J.PREFIX}/audio/{gid8}_{ang}.m4a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--i-rotated-creds", action="store_true")
    a = ap.parse_args()
    gids = [g.strip() for g in a.games.split(",") if g.strip()]
    import boto3
    aws = J._aws_cfg()
    region, bucket = aws.get("region", "us-east-1"), aws.get("s3_bucket")
    s3 = boto3.client("s3", region_name=region)

    if a.fetch:
        outdir = REPO / "runs/audio_full"
        outdir.mkdir(parents=True, exist_ok=True)
        ok = True
        for gid in gids:
            for ang in ANGLES:
                dst = outdir / f"{gid}_{ang}.m4a"
                r = subprocess.run(["aws", "s3", "cp",
                                    f"s3://{bucket}/{out_key(gid, ang)}",
                                    str(dst)], capture_output=True, text=True)
                sz = dst.stat().st_size / 1e6 if dst.exists() else 0
                print(f"  {gid} {ang}: {'OK' if r.returncode == 0 else 'MISSING'} {sz:5.0f} MB")
                ok &= r.returncode == 0
        return 0 if ok else 1

    J._guard_rotation(aws, a)

    def presign(op, key, exp=14400):
        return s3.generate_presigned_url(
            "get_object" if op == "get" else "put_object",
            Params={"Bucket": bucket, "Key": key}, ExpiresIn=exp)

    lines = []
    for gid in gids:
        pfx = game_prefix(gid)
        date, stem = pfx.split("/")[-2], pfx.split("/")[-1]
        for ang in ANGLES:
            src = presign("get", f"{pfx}/{date}_{stem}_{ang}.mp4")
            put = presign("put", out_key(gid, ang))
            name = f"{gid}_{ang}.m4a"
            lines.append(
                f'(ffmpeg -hide_banner -loglevel error -i "{src}" -vn -c:a copy '
                f'-y /work/{name} && curl -sS -o /dev/null -T /work/{name} "{put}" '
                f'&& echo "[done] {name}" || echo "[FAIL] {name}") &\nG="$G $!"\n'
                + ('wait $G\nG=""\n' if len(lines) % 4 == 3 else ""))
    body = "".join(lines)
    log_url = presign("put", J.log_key("audiograb"), 14400)
    ud = f"""#!/bin/bash
exec > /var/log/ag.log 2>&1
export HOME=/root
(while true; do sleep 20; curl -s -T /var/log/ag.log "{log_url}" >/dev/null 2>&1 || true; done) &
(sleep 3600; echo "[boot] 1h failsafe"; shutdown -h now) &
which ffmpeg || (apt-get update -qq && apt-get install -y -qq ffmpeg)
mkdir -p /work
G=""
{body}
[ -n "$G" ] && wait $G
echo "[boot] all audio grabs done"
curl -s -T /var/log/ag.log "{log_url}" >/dev/null 2>&1 || true
sleep 5; shutdown -h now
"""
    ec2 = boto3.client("ec2", region_name=region)
    r = ec2.run_instances(
        ImageId=aws.get("ami"), InstanceType="c6i.large",
        MinCount=1, MaxCount=1, InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                              "Ebs": {"VolumeSize": 40, "VolumeType": "gp3",
                                      "DeleteOnTermination": True}}],
        UserData=ud,
        TagSpecifications=[{"ResourceType": "instance",
                            "Tags": [{"Key": "Name", "Value": "uball-audiograb"}]}])
    print(f"instance {r['Instances'][0]['InstanceId']}: {len(lines)} audio grabs")
    print(f"log: s3://{bucket}/{J.log_key('audiograb')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
