#!/usr/bin/env python3
"""Train YOLO11 on the consolidated detection dataset (player/referee/ball) on AWS.

Per YOLO_TRAINING_PLAN.md: imgsz 1280, 100 epochs, mosaic 0.5, cross-game splits.
Dataset bundle uploaded once (shared by both model runs). ~1.5-2.5h ≈ $2-3 per model;
3.5h failsafe caps $4.25. best.pt uploaded incrementally every 10 min.

  export UBALL_AWS_CREDS_ROTATED=1
  python scripts/aws_yolo_train_job.py --model yolo11s --i-rotated-creds
  python scripts/aws_yolo_train_job.py --model yolo11m --i-rotated-creds
  python scripts/aws_yolo_train_job.py --fetch --model yolo11s
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

DATASET = REPO / "data/detect_consolidated"
DATASET_KEY = f"{J.PREFIX}/detect_consolidated.tar.gz"


def userdata(dataset_url, results_url, log_url, model, batch, epochs) -> str:
    return f"""#!/bin/bash
exec > /var/log/yolo.log 2>&1
export HOME=/root PYTHONUNBUFFERED=1 YOLO_CONFIG_DIR=/tmp/Ultralytics
LOG_URL="{log_url}"
(while true; do sleep 30; curl -s -T /var/log/yolo.log "$LOG_URL" >/dev/null 2>&1 || true; done) &
(sleep 12600; echo "[boot] 3.5h failsafe shutdown"; shutdown -h now) &
PYBIN=""
for P in /opt/pytorch/bin/python /usr/bin/python3; do
  if [ -x "$P" ] && $P -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then PYBIN=$P; break; fi
done
[ -z "$PYBIN" ] && PYBIN=/usr/bin/python3
TV=$($PYBIN -c "import torch;print(torch.__version__.split('+')[0])")
$PYBIN -m pip install -q "torch==$TV" ultralytics opencv-python-headless 2>&1 | tail -1
$PYBIN -m pip uninstall -q -y torchaudio 2>/dev/null || true
$PYBIN -c "import torch; assert torch.cuda.is_available(); print('torch OK')"
mkdir -p /work && cd /work
echo "[boot] downloading dataset..."
curl -s -L "{dataset_url}" -o d.tgz && tar xzf d.tgz && rm d.tgz
sed -i "s|^path:.*|path: /work/detect_consolidated|" detect_consolidated/data.yaml
# incremental best.pt uploader
(while true; do sleep 600
   B=$(find runs -name best.pt 2>/dev/null | head -1)
   if [ -n "$B" ]; then tar czf results.tar.gz runs; curl -sS -o /dev/null -T results.tar.gz "{results_url}" || true; fi
done) &
$PYBIN - <<'PY'; RC=$?
from ultralytics import YOLO
m = YOLO("{model}.pt")
m.train(data="/work/detect_consolidated/data.yaml", imgsz=1280, epochs={epochs}, batch={batch},
        cos_lr=True, patience=25, cache="disk", mosaic=0.5, close_mosaic=15,
        copy_paste=0.0, project="runs", name="{model}-1280-ourdata-v1", exist_ok=True)
best = YOLO("runs/{model}-1280-ourdata-v1/weights/best.pt")
best.val(data="/work/detect_consolidated/data.yaml", imgsz=1280, split="test")
PY
tar czf results.tar.gz runs
CODE=$(curl -sS --max-time 1800 -w '%{{http_code}}' -o /dev/null -T results.tar.gz "{results_url}")
echo "[boot] results upload http=$CODE rc=$RC"
curl -s -T /var/log/yolo.log "$LOG_URL" >/dev/null 2>&1 || true
sleep 5; shutdown -h now
"""


def ensure_dataset(s3, bucket) -> None:
    try:
        s3.head_object(Bucket=bucket, Key=DATASET_KEY)
        print("dataset already staged in S3")
        return
    except Exception:
        pass
    tmp = Path(tempfile.mkdtemp()) / "detect_consolidated.tar.gz"
    print("packing dataset (1.7GB)...")
    with tarfile.open(tmp, "w:gz") as t:
        t.add(DATASET, arcname="detect_consolidated")
    print(f"uploading dataset ({tmp.stat().st_size // 1_000_000} MB)...")
    s3.upload_file(str(tmp), bucket, DATASET_KEY)


def launch(a) -> None:
    import boto3
    aws = J._aws_cfg()
    J._guard_rotation(aws, a)
    region, bucket = aws.get("region", "us-east-1"), aws.get("s3_bucket")
    s3 = boto3.client("s3", region_name=region)
    ensure_dataset(s3, bucket)

    def presign(op, key_, exp):
        return s3.generate_presigned_url("get_object" if op == "get" else "put_object",
                                         Params={"Bucket": bucket, "Key": key_}, ExpiresIn=exp)

    tag = f"yolo_{a.model}"
    ud = userdata(presign("get", DATASET_KEY, 28800),
                  presign("put", J.results_key(tag), 172800),
                  presign("put", J.log_key(tag), 172800), a.model, a.batch, a.epochs)
    ec2 = boto3.client("ec2", region_name=region)
    r = ec2.run_instances(
        ImageId=aws.get("ami"), InstanceType=aws.get("instance_type", "g5.2xlarge"),
        MinCount=1, MaxCount=1, InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                              "Ebs": {"VolumeSize": 120, "VolumeType": "gp3",
                                      "DeleteOnTermination": True}}],
        UserData=ud,
        TagSpecifications=[{"ResourceType": "instance",
                            "Tags": [{"Key": "Name", "Value": f"uball-yolo-{a.model}"}]}])
    iid = r["Instances"][0]["InstanceId"]
    print(f"launched {iid} — {a.model} @1280, 100 epochs, incremental best.pt, 3.5h failsafe")
    print(f"log:     s3://{bucket}/{J.log_key(tag)}")
    print(f"results: s3://{bucket}/{J.results_key(tag)}")


def fetch(a) -> None:
    import boto3
    aws = J._aws_cfg()
    region, bucket = aws.get("region", "us-east-1"), aws.get("s3_bucket")
    s3 = boto3.client("s3", region_name=region)
    out = REPO / f"runs/{a.model}-1280-ourdata-v1_fetch"
    out.mkdir(parents=True, exist_ok=True)
    tarp = out / "results.tar.gz"
    s3.download_file(bucket, J.results_key(f"yolo_{a.model}"), str(tarp))
    with tarfile.open(tarp) as t:
        t.extractall(out)
    for p in sorted(out.rglob("best.pt")):
        print(" ", p)
    for p in sorted(out.rglob("results.csv")):
        print(" ", p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["yolo11s", "yolo11m"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
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
