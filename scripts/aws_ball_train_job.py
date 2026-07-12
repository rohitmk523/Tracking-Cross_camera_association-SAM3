#!/usr/bin/env python3
"""Train the POOLED all-angle ball+hoop specialist (events Phase 1, no annotation).
Dataset: data/ball_pooled (7.1k imgs, ~7.5k Basketball boxes, far+near+consolidated,
e6 excluded). ~1.5-2h ≈ $2-3.

  export UBALL_AWS_CREDS_ROTATED=1
  python scripts/aws_ball_train_job.py --model yolo11s --i-rotated-creds
  python scripts/aws_ball_train_job.py --fetch --model yolo11s
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

DATASET = REPO / "data/ball_pooled"
DATASET_KEY = f"{J.PREFIX}/ball_pooled.tar.gz"


def userdata(dataset_url, results_url, log_url, model, batch, epochs) -> str:
    return f"""#!/bin/bash
exec > /var/log/ball.log 2>&1
export HOME=/root PYTHONUNBUFFERED=1 YOLO_CONFIG_DIR=/tmp/Ultralytics
LOG_URL="{log_url}"
(while true; do sleep 30; curl -s -T /var/log/ball.log "$LOG_URL" >/dev/null 2>&1 || true; done) &
(sleep 10800; echo "[boot] 3h failsafe"; shutdown -h now) &
PYBIN=""
for P in /opt/pytorch/bin/python /usr/bin/python3; do
  if [ -x "$P" ] && $P -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then PYBIN=$P; break; fi
done
[ -z "$PYBIN" ] && PYBIN=/usr/bin/python3
TV=$($PYBIN -c "import torch;print(torch.__version__.split('+')[0])")
$PYBIN -m pip install -q "torch==$TV" ultralytics opencv-python-headless 2>&1 | tail -1
$PYBIN -m pip uninstall -q -y torchaudio 2>/dev/null || true
mkdir -p /work && cd /work
echo "[boot] downloading pooled ball dataset..."
curl -s -L "{dataset_url}" -o d.tgz && tar xzf d.tgz && rm d.tgz
sed -i "s|^path:.*|path: /work/ball_pooled|" ball_pooled/data.yaml
(while true; do sleep 480
   B=$(find runs -name best.pt 2>/dev/null | head -1)
   if [ -n "$B" ]; then tar czf results.tar.gz runs; curl -sS -o /dev/null -T results.tar.gz "{results_url}" || true; fi
done) &
$PYBIN - <<'PY'; RC=$?
from ultralytics import YOLO
m = YOLO("{model}.pt")
m.train(data="/work/ball_pooled/data.yaml", imgsz=1280, epochs={epochs}, batch={batch},
        cos_lr=True, patience=25, cache="disk", mosaic=0.5, close_mosaic=15,
        copy_paste=0.0, project="runs", name="ball-{model}-1280-v1", exist_ok=True)
best = YOLO("runs/ball-{model}-1280-v1/weights/best.pt")
r = best.val(data="/work/ball_pooled/data.yaml", imgsz=1280, split="test")
for i, cls in r.names.items():
    print(f"BALLVAL {{cls}}: mAP50 {{r.box.ap50[i]:.3f}} mAP50-95 {{r.box.ap[i]:.3f}}")
PY
tar czf results.tar.gz runs
CODE=$(curl -sS --max-time 1800 -w '%{{http_code}}' -o /dev/null -T results.tar.gz "{results_url}")
echo "[boot] results upload http=$CODE rc=$RC"
curl -s -T /var/log/ball.log "$LOG_URL" >/dev/null 2>&1 || true
sleep 5; shutdown -h now
"""


def pack_deref(tmp: Path) -> None:
    """Tar the dataset, resolving image symlinks to real files."""
    with tarfile.open(tmp, "w:gz") as t:
        for p in sorted(DATASET.rglob("*")):
            if p.is_dir():
                continue
            arc = "ball_pooled/" + str(p.relative_to(DATASET))
            real = p.resolve() if p.is_symlink() else p
            t.add(real, arcname=arc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s", choices=["yolo11n", "yolo11s", "yolo11m", "yolo26s"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--i-rotated-creds", action="store_true")
    a = ap.parse_args()
    import boto3
    aws = J._aws_cfg()
    region, bucket = aws.get("region", "us-east-1"), aws.get("s3_bucket")
    s3 = boto3.client("s3", region_name=region)
    tag = f"ball_{a.model}"
    if a.fetch:
        tmp = Path(tempfile.mkdtemp()) / "r.tar.gz"
        s3.download_file(bucket, J.results_key(tag), str(tmp))
        dst = REPO / f"runs/ball_{a.model}_fetch"
        dst.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tmp) as t:
            t.extractall(dst)
        best = next(dst.rglob("best.pt"), None)
        print(f"fetched -> {dst}\nbest.pt: {best}")
        return 0
    J._guard_rotation(aws, a)

    try:
        s3.head_object(Bucket=bucket, Key=DATASET_KEY)
        print("pooled ball dataset already staged")
    except Exception:
        tmp = Path(tempfile.mkdtemp()) / "ball_pooled.tar.gz"
        print("packing pooled ball dataset (deref symlinks)...")
        pack_deref(tmp)
        print(f"uploading ({tmp.stat().st_size // 1_000_000} MB)...")
        s3.upload_file(str(tmp), bucket, DATASET_KEY)

    def presign(op, key_, exp):
        return s3.generate_presigned_url("get_object" if op == "get" else "put_object",
                                         Params={"Bucket": bucket, "Key": key_}, ExpiresIn=exp)

    ud = userdata(presign("get", DATASET_KEY, 28800),
                  presign("put", J.results_key(tag), 172800),
                  presign("put", J.log_key(tag), 172800), a.model, a.batch, a.epochs)
    ec2 = boto3.client("ec2", region_name=region)
    r = ec2.run_instances(
        ImageId=aws.get("ami"), InstanceType=aws.get("instance_type", "g5.2xlarge"),
        MinCount=1, MaxCount=1, InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                              "Ebs": {"VolumeSize": 100, "VolumeType": "gp3",
                                      "DeleteOnTermination": True}}],
        UserData=ud,
        TagSpecifications=[{"ResourceType": "instance",
                            "Tags": [{"Key": "Name", "Value": f"uball-ball-{a.model}"}]}])
    print(f"launched {r['Instances'][0]['InstanceId']} — pooled ball {a.model} @1280, "
          f"{a.epochs}ep, 3h failsafe")
    print(f"log: s3://{bucket}/{J.log_key(tag)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
