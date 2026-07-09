#!/usr/bin/env python3
"""Launch MULTI-OBJECT SAM3 tracking on AWS: ALL roster players in ONE pass per camera
(forward + backward from a common identity-confirmed seed frame). Replaces the 59-pass
single-object roster job (~$10) at ~$1-2. Hard 3h shutdown failsafe caps worst-case
spend at ~$3.60 (budget rule: runs stay under $5).

  export UBALL_AWS_CREDS_ROTATED=1
  python scripts/aws_sam3_multi_job.py --game e6fba750 --tag 44_60 --i-rotated-creds
  python scripts/aws_sam3_multi_job.py --fetch                       # -> runs/sam3_players_multi/
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import aws_sam3_job as J   # noqa: E402

ANGLES = ("FL", "FR", "NL", "NR")


def userdata(bundle_url, weights_url, results_url, log_url, cmds: list[str]) -> str:
    body = "\n".join(cmds)
    return f"""#!/bin/bash
exec > /var/log/sam3.log 2>&1
export HOME=/root PYTHONUNBUFFERED=1
LOG_URL="{log_url}"
(while true; do sleep 30; curl -s -T /var/log/sam3.log "$LOG_URL" >/dev/null 2>&1 || true; done) &
(sleep 10800; echo "[boot] 3h failsafe shutdown"; shutdown -h now) &
PYBIN=""
for P in /opt/pytorch/bin/python /usr/bin/python3; do
  if [ -x "$P" ] && $P -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then PYBIN=$P; break; fi
done
[ -z "$PYBIN" ] && PYBIN=/usr/bin/python3
$PYBIN -m pip install -q -U ultralytics opencv-python-headless
mkdir -p /work && cd /work
curl -s -L "{bundle_url}" -o b.tgz && tar xzf b.tgz
curl -s -L "{weights_url}" -o sam3.pt
ls -la sam3.pt
mkdir -p out
RC=0
{body}
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

    key = f"{a.game}_{a.tag}"
    ms = json.loads((REPO / f"runs/anchors/{key}.multiseeds.json").read_text())["cams"]
    cmds = []
    clips = []
    for cam in ANGLES:
        if cam not in ms or not ms[cam]["players"]:
            continue
        clip = f"{a.game}_{cam}_{a.tag}.mp4"
        clips.append(REPO / "data/clips" / clip)
        seeds_json = json.dumps(ms[cam]["players"]).replace('"', '\\"')
        cmds.append(
            f'echo "=== {cam} multi ({len(ms[cam]["players"])} players, seed {ms[cam]["seed_frame"]}) ==="; '
            f'$PYBIN sam3_track_multi.py --video "clips/{clip}" '
            f'--seed-frame {ms[cam]["seed_frame"]} --seeds "{seeds_json}" '
            f'--cam {cam} --tag "{key}" --out-dir out --weights /work/sam3.pt || RC=1')

    tmp = Path(tempfile.mkdtemp()) / "sam3multi_bundle.tar.gz"
    with tarfile.open(tmp, "w:gz") as t:
        t.add(REPO / "scripts/sam3_track_multi.py", arcname="sam3_track_multi.py")
        for c in clips:
            t.add(c, arcname=f"clips/{c.name}")
    tag = a.tag2
    bundle_key = f"{J.PREFIX}/sam3multi_bundle_{tag}.tar.gz"
    print(f"uploading bundle ({tmp.stat().st_size // 1_000_000} MB, {len(clips)} clips, "
          f"{len(cmds)} multi-object passes)...")
    s3.upload_file(str(tmp), bucket, bundle_key)

    def presign(op, key_, exp):
        return s3.generate_presigned_url("get_object" if op == "get" else "put_object",
                                         Params={"Bucket": bucket, "Key": key_}, ExpiresIn=exp)

    ud = userdata(presign("get", bundle_key, 28800), presign("get", J.WEIGHTS_KEY, 28800),
                  presign("put", J.results_key(tag), 86400), presign("put", J.log_key(tag), 86400),
                  cmds)
    ec2 = boto3.client("ec2", region_name=region)
    r = ec2.run_instances(
        ImageId=aws.get("ami"), InstanceType=aws.get("instance_type", "g5.2xlarge"),
        MinCount=1, MaxCount=1, InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                              "Ebs": {"VolumeSize": 100, "VolumeType": "gp3",
                                      "DeleteOnTermination": True}}],
        UserData=ud,
        TagSpecifications=[{"ResourceType": "instance",
                            "Tags": [{"Key": "Name", "Value": "uball-sam3-multi"}]}])
    iid = r["Instances"][0]["InstanceId"]
    print(f"launched {iid} ({aws.get('instance_type')}) — {len(cmds)} multi-object passes, "
          f"3h failsafe, self-terminates")
    print(f"log:     s3://{bucket}/{J.log_key(tag)}")
    print(f"results: s3://{bucket}/{J.results_key(tag)}   (then: --fetch --tag2 {tag})")


def fetch(a) -> None:
    import boto3
    aws = J._aws_cfg()
    region, bucket = aws.get("region", "us-east-1"), aws.get("s3_bucket")
    s3 = boto3.client("s3", region_name=region)
    out = REPO / "runs/sam3_players_multi"
    out.mkdir(parents=True, exist_ok=True)
    tarp = out / "results.tar.gz"
    s3.download_file(bucket, J.results_key(a.tag2), str(tarp))
    with tarfile.open(tarp) as t:
        t.extractall(out)
    names = sorted(p.name for p in out.glob("*.json"))
    print(f"fetched {len(names)} masklets -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="e6fba750")
    ap.add_argument("--tag", default="44_60")
    ap.add_argument("--tag2", default="e6multi", help="job tag (S3 scoping)")
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
