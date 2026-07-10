#!/usr/bin/env python3
"""Fine-tune KPR on AWS (one g5.2xlarge, ~30-60 min, ~$1-2; 3h failsafe caps $3.65).

Bundles the patched /tmp/kpr checkout (uball dataset class + config + trainer),
the reid dataset, and the pretrained checkpoint. Returns the fine-tuned weights.

  export UBALL_AWS_CREDS_ROTATED=1
  python scripts/aws_kpr_train_job.py --i-rotated-creds
  python scripts/aws_kpr_train_job.py --fetch          # -> runs/kpr_ft/
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

KPR = Path("/tmp/kpr")


def userdata(bundle_url, results_url, log_url) -> str:
    return f"""#!/bin/bash
exec > /var/log/kpr.log 2>&1
export HOME=/root PYTHONUNBUFFERED=1
LOG_URL="{log_url}"
(while true; do sleep 30; curl -s -T /var/log/kpr.log "$LOG_URL" >/dev/null 2>&1 || true; done) &
(sleep 10800; echo "[boot] 3h failsafe shutdown"; shutdown -h now) &
PYBIN=""
for P in /opt/pytorch/bin/python /usr/bin/python3; do
  if [ -x "$P" ] && $P -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then PYBIN=$P; break; fi
done
[ -z "$PYBIN" ] && PYBIN=/usr/bin/python3
$PYBIN -m pip install -q numpy Cython h5py six scipy opencv-python-headless "matplotlib<3.9" future yacs gdown albumentations==1.3.1 pandas tabulate deepdiff wandb monai torchmetrics==1.3.0 timm scikit-image tqdm omegaconf scikit-learn tensorboard segment-anything 2>&1 | tail -1
SP=$($PYBIN -c "import site;print(site.getsitepackages()[0])")
mkdir -p "$SP/cosine_annealing_warmup"
printf 'class CosineAnnealingWarmupRestarts:\\n    def __init__(self,*a,**k): raise NotImplementedError()\\n' > "$SP/cosine_annealing_warmup/__init__.py"
mkdir -p /work && cd /work
curl -s -L "{bundle_url}" -o b.tgz && tar xzf b.tgz && cd kpr
export WANDB_MODE=disabled
$PYBIN train_uball.py; RC=$?
tar czf /work/results.tar.gz logs
CODE=$(curl -sS --max-time 1800 -w '%{{http_code}}' -o /dev/null -T /work/results.tar.gz "{results_url}")
echo "[boot] results upload http=$CODE rc=$RC"
curl -s -T /var/log/kpr.log "$LOG_URL" >/dev/null 2>&1 || true
sleep 5; shutdown -h now
"""


def launch(a) -> None:
    import boto3
    aws = J._aws_cfg()
    J._guard_rotation(aws, a)
    region, bucket = aws.get("region", "us-east-1"), aws.get("s3_bucket")
    s3 = boto3.client("s3", region_name=region)

    tmp = Path(tempfile.mkdtemp()) / "kpr_train_bundle.tar.gz"
    print("bundling /tmp/kpr (repo + dataset + pretrained weights)...")
    with tarfile.open(tmp, "w:gz") as t:
        def flt(ti):
            n = ti.name
            if any(x in n for x in (".venv", ".git", "__pycache__", "/logs/", "assets/demo")):
                return None
            return ti
        t.add(KPR, arcname="kpr", filter=flt)
    key = f"{J.PREFIX}/kpr_train_bundle.tar.gz"
    print(f"uploading bundle ({tmp.stat().st_size // 1_000_000} MB)...")
    s3.upload_file(str(tmp), bucket, key)

    def presign(op, key_, exp):
        return s3.generate_presigned_url("get_object" if op == "get" else "put_object",
                                         Params={"Bucket": bucket, "Key": key_}, ExpiresIn=exp)

    ud = userdata(presign("get", key, 28800),
                  presign("put", J.results_key("kprft"), 86400),
                  presign("put", J.log_key("kprft"), 86400))
    ec2 = boto3.client("ec2", region_name=region)
    r = ec2.run_instances(
        ImageId=aws.get("ami"), InstanceType=aws.get("instance_type", "g5.2xlarge"),
        MinCount=1, MaxCount=1, InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                              "Ebs": {"VolumeSize": 100, "VolumeType": "gp3",
                                      "DeleteOnTermination": True}}],
        UserData=ud,
        TagSpecifications=[{"ResourceType": "instance",
                            "Tags": [{"Key": "Name", "Value": "uball-kpr-train"}]}])
    iid = r["Instances"][0]["InstanceId"]
    print(f"launched {iid} — KPR fine-tune, 3h failsafe, self-terminates")
    print(f"log:     s3://{bucket}/{J.log_key('kprft')}")
    print(f"results: s3://{bucket}/{J.results_key('kprft')}")


def fetch(a) -> None:
    import boto3
    aws = J._aws_cfg()
    region, bucket = aws.get("region", "us-east-1"), aws.get("s3_bucket")
    s3 = boto3.client("s3", region_name=region)
    out = REPO / "runs/kpr_ft"
    out.mkdir(parents=True, exist_ok=True)
    tarp = out / "results.tar.gz"
    s3.download_file(bucket, J.results_key("kprft"), str(tarp))
    with tarfile.open(tarp) as t:
        t.extractall(out)
    print(f"fetched -> {out}")
    for p in sorted(out.rglob("*.pth.tar")):
        print(" ", p)


def main() -> int:
    ap = argparse.ArgumentParser()
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
