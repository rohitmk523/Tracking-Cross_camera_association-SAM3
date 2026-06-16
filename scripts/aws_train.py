#!/usr/bin/env python3
"""Launch an RF-DETR-Small @1280 training run on AWS GPU (CUDA-only).

Ported from the proven Training_frameworks/Uball E6 Demo/src/aws_train.py pattern:
bundle the consolidated YOLO dataset + train script -> S3 (presigned), launch a
g5/g6, install rfdetr[train]+sahi, train, upload best weights -> S3, self-terminate.

  python scripts/aws_train.py --dry-run     # verify bundle + plan, NO AWS calls
  python scripts/aws_train.py               # launch (guarded by creds rotation)
  python scripts/aws_train.py --fetch       # download trained weights when done

CREDENTIALS: read from ~/.aws / env (boto3 default chain). NEVER embedded here.
The prior-project keys are flagged for ROTATION (docs/11): launch is blocked on
account 840102831548 unless you pass --i-rotated-creds or set
UBALL_AWS_CREDS_ROTATED=1 after rotating.
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((REPO / "configs" / "train_rfdetr.yaml").read_text())
AWS = CFG.get("aws", {})
REGION = AWS.get("region", "us-east-1")
BUCKET = AWS.get("s3_bucket", "uball-videos-production")
PREFIX = AWS.get("s3_prefix", "_tmp_rfdetr_train")
DATASET = REPO / CFG.get("dataset", "data/detect_consolidated")


def _bundle_files() -> list[tuple[Path, str]]:
    """(*src*, *arcname*) pairs placing everything under the instance's /work."""
    pairs = [
        (REPO / "scripts" / "train_rfdetr.py", "scripts/train_rfdetr.py"),
        (REPO / "configs" / "train_rfdetr.yaml", "configs/train_rfdetr.yaml"),
        (DATASET / "data.yaml", f"{DATASET.name}-as-data/data.yaml"),
    ]
    base = f"{DATASET.name}-as-data"
    for split in ("train", "valid"):                # test recreated on instance
        for sub in ("images", "labels"):
            d = DATASET / split / sub
            if d.is_dir():
                for p in sorted(d.iterdir()):
                    if p.is_file() and not p.is_symlink():
                        pairs.append((p, f"{base}/{split}/{sub}/{p.name}"))
    return pairs


def bundle(pairs: list[tuple[Path, str]]) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "bundle.tar.gz"
    with tarfile.open(tmp, "w:gz") as t:
        for src, arc in pairs:
            t.add(src, arcname=arc)
    return tmp


def build_userdata(bundle_url: str, weights_url: str, log_url: str) -> str:
    res, model = CFG.get("resolution", 1280), CFG.get("model", "small")
    run = CFG.get("run_name", "rfdetr-s-1280-ourdata-v1")
    return f"""#!/bin/bash
exec > /var/log/train.log 2>&1
export HOME=/root PYTHONUNBUFFERED=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4
LOG_URL="{log_url}"
(while true; do sleep 30; curl -s -T /var/log/train.log "$LOG_URL" >/dev/null 2>&1 || true; done) &
PYBIN=""
for P in /opt/pytorch/bin/python /usr/bin/python3; do
  if [ -x "$P" ] && $P -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then PYBIN=$P; break; fi
done
[ -z "$PYBIN" ] && PYBIN=/usr/bin/python3
$PYBIN -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
$PYBIN -m pip install -q "rfdetr[train,loggers]" "sahi>=0.11.18"
$PYBIN -m pip uninstall -y transformer_engine >/dev/null 2>&1 || true
mkdir -p /work && cd /work
curl -s -L "{bundle_url}" -o b.tgz && tar xzf b.tgz
# Restore the dataset dir name the config expects + recreate the test symlink.
mv {DATASET.name}-as-data data/{DATASET.name} 2>/dev/null || (mkdir -p data && mv {DATASET.name}-as-data data/{DATASET.name})
mkdir -p data/{DATASET.name}/test
ln -sf ../valid/images data/{DATASET.name}/test/images
ln -sf ../valid/labels data/{DATASET.name}/test/labels
nvidia-smi || true
echo "[boot] train RF-DETR-{model} @{res}  run={run}"
$PYBIN scripts/train_rfdetr.py --config configs/train_rfdetr.yaml
echo "[boot] train exit=$?"; sync
BEST=$(ls -t /work/runs/*/checkpoint_best_ema.pth /work/runs/*/checkpoint_best*.pth /work/runs/*/*.pth 2>/dev/null | head -1)
echo "[boot] best=$BEST"
if [ -n "$BEST" ]; then
  for try in 1 2 3; do
    CODE=$(curl -sS --max-time 1800 -w '%{{http_code}}' -o /dev/null -T "$BEST" "{weights_url}")
    echo "[boot] weights upload try $try http=$CODE"; [ "$CODE" = "200" ] && {{ echo "[boot] WEIGHTS DONE"; break; }}; sleep 10
  done
else
  echo "[boot] NO WEIGHTS FOUND"
fi
curl -s -T /var/log/train.log "$LOG_URL" >/dev/null 2>&1 || true
sleep 5; shutdown -h now
"""


def _guard_rotation(a) -> None:
    """Refuse to spend AWS on un-rotated, flagged credentials (docs/11)."""
    import boto3
    flagged = str(AWS.get("flagged_account", ""))
    confirmed = a.i_rotated_creds or os.environ.get("UBALL_AWS_CREDS_ROTATED") == "1"
    try:
        ident = boto3.client("sts", region_name=REGION).get_caller_identity()
    except Exception as e:                      # noqa: BLE001
        sys.exit(f"AWS credentials not usable ({e}). Configure ~/.aws or .env first.")
    account = ident.get("Account", "?")
    print(f"AWS account: {account}  (arn={ident.get('Arn','?')})")
    if account == flagged and not confirmed:
        sys.exit(
            f"\nREFUSING TO LAUNCH: account {account} uses keys flagged for ROTATION "
            "(docs/11).\nRotate them, then re-run with --i-rotated-creds (or set "
            "UBALL_AWS_CREDS_ROTATED=1). Never commit the new keys.")


def launch(a) -> None:
    import boto3
    _guard_rotation(a)
    pairs = _bundle_files()
    b = bundle(pairs)
    s3 = boto3.client("s3", region_name=REGION)
    weights_key = f"{PREFIX}/{a.weights_key}"
    bundle_key = f"{PREFIX}/{a.weights_key.replace('.pth', '_bundle.tar.gz')}"
    log_key = f"{PREFIX}/{a.weights_key.replace('.pth', '_train.log')}"
    print(f"uploading bundle ({b.stat().st_size // 1_000_000} MB, {len(pairs)} files)...")
    s3.upload_file(str(b), BUCKET, bundle_key)

    def _presign(method, key, exp=28800):
        op = "get_object" if method == "get" else "put_object"
        return s3.generate_presigned_url(op, Params={"Bucket": BUCKET, "Key": key},
                                         ExpiresIn=exp)
    ud = build_userdata(_presign("get", bundle_key), _presign("put", weights_key),
                        _presign("put", log_key))
    ec2 = boto3.client("ec2", region_name=REGION)
    r = ec2.run_instances(
        ImageId=AWS.get("ami"), InstanceType=AWS.get("instance_type", "g5.2xlarge"),
        MinCount=1, MaxCount=1, InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                              "Ebs": {"VolumeSize": AWS.get("volume_gb", 120),
                                      "VolumeType": "gp3", "DeleteOnTermination": True}}],
        UserData=ud,
        TagSpecifications=[{"ResourceType": "instance",
                            "Tags": [{"Key": "Name", "Value": "uball-rfdetr-train"}]}])
    iid = r["Instances"][0]["InstanceId"]
    print(f"launched {iid}  ({AWS.get('instance_type')}, RF-DETR-{CFG.get('model')})")
    print(f"watch:  aws s3 cp s3://{BUCKET}/{log_key} -")
    print(f"weights land: s3://{BUCKET}/{weights_key}  (fetch with --fetch)")


def dry_run() -> None:
    pairs = _bundle_files()
    n_img = sum(1 for _, a in pairs if "/images/" in a)
    n_lab = sum(1 for _, a in pairs if "/labels/" in a)
    size_mb = sum(s.stat().st_size for s, _ in pairs) // 1_000_000
    print("DRY RUN -- no AWS calls.")
    print(f"  dataset:       {DATASET}")
    print(f"  bundle files:  {len(pairs)}  ({n_img} images, {n_lab} labels)")
    print(f"  bundle size:   ~{size_mb} MB (uncompressed)")
    print(f"  model:         RF-DETR-{CFG.get('model')} @ {CFG.get('resolution')}  "
          f"epochs={CFG.get('epochs')} batch={CFG.get('batch_size')}")
    print(f"  instance:      {AWS.get('instance_type')}  ami={AWS.get('ami')}  "
          f"region={REGION}")
    print(f"  s3:            s3://{BUCKET}/{PREFIX}/")
    print(f"  rotation guard: blocks launch on account {AWS.get('flagged_account')} "
          "unless --i-rotated-creds / UBALL_AWS_CREDS_ROTATED=1")
    if not (DATASET / "data.yaml").exists():
        print("  WARNING: dataset not built -- run scripts/build_detection_dataset.py")


def fetch(a) -> None:
    import boto3
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3", region_name=REGION).download_file(
        BUCKET, f"{PREFIX}/{a.weights_key}", str(out))
    print(f"downloaded -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="verify bundle+plan, no AWS")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--i-rotated-creds", action="store_true",
                    help="confirm the flagged AWS keys were rotated (docs/11)")
    ap.add_argument("--weights-key", default="rfdetr_s_1280_ourdata_v1_best.pth")
    ap.add_argument("--out", default=str(REPO / "runs" / "rfdetr_s_1280_ourdata_v1"
                                         / "best.pth"))
    a = ap.parse_args()
    if a.dry_run:
        dry_run()
    elif a.fetch:
        fetch(a)
    else:
        launch(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
