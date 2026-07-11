#!/usr/bin/env python3
"""Run the detector race (Level-3) on one A10G: RF-DETR-FP16 vs YOLO lanes on
identical blind-game frames. ~15-25 min ≈ $0.4; 1.5h failsafe caps $1.85.

  export UBALL_AWS_CREDS_ROTATED=1
  python scripts/aws_race_job.py --i-rotated-creds [--yolo2 path/to/yolo11m_best.pt]
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import aws_sam3_job as J   # noqa: E402

ANGLES = ("FL", "FR", "NL", "NR")


def userdata(bundle_url, results_url, log_url, yolo2) -> str:
    y2 = " --yolo2 weights/yolo11m_best.pt" if yolo2 else ""
    return f"""#!/bin/bash
exec > /var/log/race.log 2>&1
export HOME=/root PYTHONUNBUFFERED=1 YOLO_CONFIG_DIR=/tmp/Ultralytics
LOG_URL="{log_url}"
(while true; do sleep 30; curl -s -T /var/log/race.log "$LOG_URL" >/dev/null 2>&1 || true; done) &
(sleep 5400; echo "[boot] 1.5h failsafe"; shutdown -h now) &
PYBIN=""
for P in /opt/pytorch/bin/python /usr/bin/python3; do
  if [ -x "$P" ] && $P -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then PYBIN=$P; break; fi
done
TV=$($PYBIN -c "import torch;print(torch.__version__.split('+')[0])")
$PYBIN -m pip install -q "torch==$TV" rfdetr ultralytics supervision opencv-python-headless timm 2>&1 | tail -1
$PYBIN -m pip uninstall -q -y torchaudio 2>/dev/null || true
mkdir -p /work && cd /work
curl -s -L "{bundle_url}" -o b.tgz && tar xzf b.tgz
export PYTHONPATH=/work/src
RC=0
$PYBIN scripts/race_detectors.py --game f66eb3b2 --tag race_60 \\
  --rfdetr weights/best.pth --yolo weights/yolo11s_best.pt{y2} | tee race_out.txt || RC=1
curl -sS -o /dev/null -T race_out.txt "{results_url}" || true
echo "[boot] rc=$RC"
curl -s -T /var/log/race.log "$LOG_URL" >/dev/null 2>&1 || true
sleep 5; shutdown -h now
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo2", default=None, help="optional yolo11m best.pt path")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--i-rotated-creds", action="store_true")
    a = ap.parse_args()
    import boto3
    aws = J._aws_cfg()
    region, bucket = aws.get("region", "us-east-1"), aws.get("s3_bucket")
    s3 = boto3.client("s3", region_name=region)
    if a.fetch:
        s3.download_file(bucket, J.results_key("race"), "/tmp/race_out.txt")
        print(Path("/tmp/race_out.txt").read_text())
        return 0
    J._guard_rotation(aws, a)

    yolo_best = next((REPO / "runs/yolo11s-1280-ourdata-v1_fetch/runs/detect").rglob("best.pt"))
    tmp = Path(tempfile.mkdtemp()) / "race_bundle.tar.gz"
    print("bundling race clips + weights + sources...")
    with tarfile.open(tmp, "w:gz") as t:
        for ang in ANGLES:
            t.add(REPO / f"data/clips/f66eb3b2_{ang}_race_60.mp4",
                  arcname=f"data/clips/f66eb3b2_{ang}_race_60.mp4")
        t.add(REPO / "scripts/race_detectors.py", arcname="scripts/race_detectors.py")
        t.add(REPO / "runs/rfdetr-s-1280-ourdata-v1/best.pth", arcname="weights/best.pth")
        t.add(yolo_best, arcname="weights/yolo11s_best.pt")
        if a.yolo2:
            t.add(a.yolo2, arcname="weights/yolo11m_best.pt")
        def flt(ti):
            return None if "__pycache__" in ti.name else ti
        t.add(REPO / "src", arcname="src", filter=flt)
    key = f"{J.PREFIX}/race_bundle.tar.gz"
    print(f"uploading bundle ({tmp.stat().st_size // 1_000_000} MB)...")
    s3.upload_file(str(tmp), bucket, key)

    def presign(op, key_, exp):
        return s3.generate_presigned_url("get_object" if op == "get" else "put_object",
                                         Params={"Bucket": bucket, "Key": key_}, ExpiresIn=exp)

    ud = userdata(presign("get", key, 28800), presign("put", J.results_key("race"), 86400),
                  presign("put", J.log_key("race"), 86400), a.yolo2)
    ec2 = boto3.client("ec2", region_name=region)
    r = ec2.run_instances(
        ImageId=aws.get("ami"), InstanceType=aws.get("instance_type", "g5.2xlarge"),
        MinCount=1, MaxCount=1, InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                              "Ebs": {"VolumeSize": 100, "VolumeType": "gp3",
                                      "DeleteOnTermination": True}}],
        UserData=ud,
        TagSpecifications=[{"ResourceType": "instance",
                            "Tags": [{"Key": "Name", "Value": "uball-race"}]}])
    print(f"launched {r['Instances'][0]['InstanceId']} — the race, 1.5h failsafe")
    print(f"log: s3://{bucket}/{J.log_key('race')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
