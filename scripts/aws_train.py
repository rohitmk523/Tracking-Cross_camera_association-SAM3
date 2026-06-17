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


class Ctx:
    """Everything derived from the chosen training config (one per run task)."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config_name = config_path.name
        self.cfg = yaml.safe_load(config_path.read_text())
        self.aws = self.cfg.get("aws", {})
        self.region = self.aws.get("region", "us-east-1")
        self.bucket = self.aws.get("s3_bucket", "uball-videos-production")
        self.prefix = self.aws.get("s3_prefix", "_tmp_rfdetr_train")
        ds = self.cfg.get("dataset", "data/detect_consolidated")
        self.dataset = Path(ds) if Path(ds).is_absolute() else REPO / ds


def _bundle_files(ctx: Ctx) -> list[tuple[Path, str]]:
    """(*src*, *arcname*) pairs placing everything under the instance's /work."""
    ds = ctx.dataset
    pairs = [
        (REPO / "scripts" / "train_rfdetr.py", "scripts/train_rfdetr.py"),
        (ctx.config_path, f"configs/{ctx.config_name}"),
        (ds / "data.yaml", f"{ds.name}-as-data/data.yaml"),
    ]
    base = f"{ds.name}-as-data"
    for split in ("train", "valid"):                # test recreated on instance
        for sub in ("images", "labels"):
            d = ds / split / sub
            if d.is_dir():
                for p in sorted(d.iterdir()):
                    # Resolve symlinks so the REAL file content is bundled -- a bare
                    # is_symlink skip would silently ship an EMPTY dataset when the
                    # dataset dir uses symlinks (review #9).
                    if p.is_file():            # is_file() follows symlinks
                        pairs.append((p.resolve(), f"{base}/{split}/{sub}/{p.name}"))
    return pairs


def bundle(pairs: list[tuple[Path, str]]) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "bundle.tar.gz"
    with tarfile.open(tmp, "w:gz") as t:
        for src, arc in pairs:
            t.add(src, arcname=arc)
    return tmp


def build_userdata(ctx: Ctx, bundle_url: str, weights_url: str, log_url: str) -> str:
    res, model = ctx.cfg.get("resolution", 1280), ctx.cfg.get("model", "small")
    run = ctx.cfg.get("run_name", "rfdetr-s-1280-ourdata-v1")
    dsname = ctx.dataset.name
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
mv {dsname}-as-data data/{dsname} 2>/dev/null || (mkdir -p data && mv {dsname}-as-data data/{dsname})
mkdir -p data/{dsname}/test
ln -sf ../valid/images data/{dsname}/test/images
ln -sf ../valid/labels data/{dsname}/test/labels
nvidia-smi || true
echo "[boot] train RF-DETR-{model} @{res}  run={run}"
$PYBIN scripts/train_rfdetr.py --config configs/{ctx.config_name}
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


def _guard_rotation(ctx: Ctx, a) -> None:
    """Refuse to spend AWS on flagged credentials unless explicitly authorized (docs/11).

    Two honest overrides:
      --i-rotated-creds        : the keys WERE rotated (clean).
      --accept-unrotated-creds : operator accepts spending on the CURRENT (un-rotated)
                                 flagged keys now; rotation is still OWED afterwards.
    """
    import boto3
    flagged = str(ctx.aws.get("flagged_account", "")).strip()
    rotated = a.i_rotated_creds or os.environ.get("UBALL_AWS_CREDS_ROTATED") == "1"
    accept_unrotated = getattr(a, "accept_unrotated_creds", False)
    if not flagged:                              # fail CLOSED (review #16)
        sys.exit("config aws.flagged_account is empty -- refusing to launch without "
                 "a rotation guard. Set it (docs/11) before launching.")
    try:
        ident = boto3.client("sts", region_name=ctx.region).get_caller_identity()
    except Exception as e:                      # noqa: BLE001
        sys.exit(f"AWS credentials not usable ({e}). Configure ~/.aws or .env first.")
    account = ident.get("Account", "?")
    print(f"AWS account: {account}  (arn={ident.get('Arn','?')})")
    if account == flagged and not (rotated or accept_unrotated):
        sys.exit(
            f"\nREFUSING TO LAUNCH: account {account} uses keys flagged for ROTATION "
            "(docs/11).\nEither rotate them and pass --i-rotated-creds, or pass "
            "--accept-unrotated-creds to launch on the current keys (rotation still owed).")
    if account == flagged and accept_unrotated and not rotated:
        print("\n  ⚠️  LAUNCHING ON UN-ROTATED FLAGGED KEYS (operator-authorized).")
        print("  ⚠️  These admin keys are exposed -- ROTATION IS STILL OWED after this run.\n")


def launch(ctx: Ctx, a) -> None:
    import boto3
    _guard_rotation(ctx, a)
    weights_key_name = a.weights_key or f"{ctx.cfg.get('run_name', 'rfdetr')}_best.pth"
    pairs = _bundle_files(ctx)
    b = bundle(pairs)
    s3 = boto3.client("s3", region_name=ctx.region)
    weights_key = f"{ctx.prefix}/{weights_key_name}"
    bundle_key = f"{ctx.prefix}/{weights_key_name.replace('.pth', '_bundle.tar.gz')}"
    log_key = f"{ctx.prefix}/{weights_key_name.replace('.pth', '_train.log')}"
    print(f"uploading bundle ({b.stat().st_size // 1_000_000} MB, {len(pairs)} files)...")
    s3.upload_file(str(b), ctx.bucket, bundle_key)

    def _presign(method, key, exp):
        op = "get_object" if method == "get" else "put_object"
        return s3.generate_presigned_url(op, Params={"Bucket": ctx.bucket, "Key": key},
                                         ExpiresIn=exp)
    # Weights/log PUTs are consumed at the END of training -- give them 24h so a
    # long run can't lose its output to an expired URL (review #6). Bundle GET is
    # fetched at boot, so 8h is plenty.
    ud = build_userdata(ctx, _presign("get", bundle_key, 28800),
                        _presign("put", weights_key, 86400),
                        _presign("put", log_key, 86400))
    ec2 = boto3.client("ec2", region_name=ctx.region)
    r = ec2.run_instances(
        ImageId=ctx.aws.get("ami"), InstanceType=ctx.aws.get("instance_type", "g5.2xlarge"),
        MinCount=1, MaxCount=1, InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                              "Ebs": {"VolumeSize": ctx.aws.get("volume_gb", 120),
                                      "VolumeType": "gp3", "DeleteOnTermination": True}}],
        UserData=ud,
        TagSpecifications=[{"ResourceType": "instance",
                            "Tags": [{"Key": "Name",
                                      "Value": f"uball-rfdetr-{ctx.cfg.get('run_name','train')}"}]}])
    iid = r["Instances"][0]["InstanceId"]
    print(f"launched {iid}  ({ctx.aws.get('instance_type')}, RF-DETR-{ctx.cfg.get('model')}, "
          f"run={ctx.cfg.get('run_name')})")
    print(f"watch:  aws s3 cp s3://{ctx.bucket}/{log_key} -")
    print(f"weights land: s3://{ctx.bucket}/{weights_key}  (fetch with "
          f"--config {ctx.config_name} --fetch)")


def dry_run(ctx: Ctx) -> None:
    pairs = _bundle_files(ctx)
    n_img = sum(1 for _, a in pairs if "/images/" in a)
    n_lab = sum(1 for _, a in pairs if "/labels/" in a)
    size_mb = sum(s.stat().st_size for s, _ in pairs) // 1_000_000
    print(f"DRY RUN ({ctx.config_name}) -- no AWS calls.")
    print(f"  dataset:       {ctx.dataset}")
    print(f"  bundle files:  {len(pairs)}  ({n_img} images, {n_lab} labels)")
    print(f"  bundle size:   ~{size_mb} MB (uncompressed)")
    print(f"  model:         RF-DETR-{ctx.cfg.get('model')} @ {ctx.cfg.get('resolution')}  "
          f"epochs={ctx.cfg.get('epochs')} batch={ctx.cfg.get('batch_size')}  "
          f"classes={ctx.cfg.get('class_names')}")
    print(f"  instance:      {ctx.aws.get('instance_type')}  ami={ctx.aws.get('ami')}  "
          f"region={ctx.region}")
    print(f"  s3:            s3://{ctx.bucket}/{ctx.prefix}/")
    print(f"  rotation guard: blocks launch on account {ctx.aws.get('flagged_account')} "
          "unless --i-rotated-creds / UBALL_AWS_CREDS_ROTATED=1")
    if not (ctx.dataset / "data.yaml").exists():
        print(f"  WARNING: dataset not built at {ctx.dataset}")


def fetch(ctx: Ctx, a) -> None:
    import boto3
    weights_key_name = a.weights_key or f"{ctx.cfg.get('run_name', 'rfdetr')}_best.pth"
    out = Path(a.out) if a.out else (REPO / "runs" / ctx.cfg.get("run_name", "rfdetr")
                                     / "best.pth")
    out.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3", region_name=ctx.region).download_file(
        ctx.bucket, f"{ctx.prefix}/{weights_key_name}", str(out))
    print(f"downloaded -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "train_rfdetr.yaml"),
                    help="training config (selects dataset/model/classes/run-name)")
    ap.add_argument("--dry-run", action="store_true", help="verify bundle+plan, no AWS")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--i-rotated-creds", action="store_true",
                    help="confirm the flagged AWS keys were rotated (docs/11)")
    ap.add_argument("--accept-unrotated-creds", action="store_true",
                    help="launch on the CURRENT un-rotated flagged keys; rotation owed after")
    ap.add_argument("--weights-key", default=None,
                    help="S3 weights filename (default <run_name>_best.pth)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ctx = Ctx(Path(a.config))
    if a.dry_run:
        dry_run(ctx)
    elif a.fetch:
        fetch(ctx, a)
    else:
        launch(ctx, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
