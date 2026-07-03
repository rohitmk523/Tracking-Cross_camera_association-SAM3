#!/usr/bin/env python3
"""Run the SAM3 reference pass on an AWS GPU (adapted from the proven aws_train.py scaffold):
stage the LOCAL sam3.pt (Ultralytics checkpoint, DEMO_UBALL/demo/sam3.pt) to S3 once,
bundle sam3_reference.py + local clips, launch a g5, run SAM3 per clip via presigned URLs,
upload results tar + live log, self-terminate. NO Hugging Face dependency — the .pt is
everything (the HF gate only matters for downloading weights we already have).

  python scripts/aws_sam3_job.py --dry-run
  UBALL_AWS_CREDS_ROTATED=1 python scripts/aws_sam3_job.py         # launch
  python scripts/aws_sam3_job.py --fetch                           # results -> runs/sam3_ref/

CREDENTIALS: boto3 default chain; rotation guard identical to aws_train.py (docs/11).
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
DEFAULT_CLIPS = [f"data/clips/{gid}_{ang}_{tag}.mp4"
                 for gid, tag in (("e6fba750", "47_12"), ("c2a354fe", "333_14"))
                 for ang in ("FL", "FR", "NL", "NR")]
PREFIX = "_tmp_sam3_ref"
RESULTS_KEY = f"{PREFIX}/sam3_ref_results.tar.gz"
LOG_KEY = f"{PREFIX}/sam3_ref.log"
WEIGHTS_KEY = f"{PREFIX}/sam3.pt"
WEIGHTS_LOCAL = Path("/Users/rohitkale/Cellstrat/GitHub_Repositories/DEMO_UBALL/demo/sam3.pt")


def _aws_cfg() -> dict:
    return yaml.safe_load((REPO / "configs" / "train_rfdetr.yaml").read_text()).get("aws", {})


def _ensure_weights(s3, bucket: str) -> None:
    """Stage sam3.pt to S3 once (3.45 GB multipart); skip if already there."""
    try:
        head = s3.head_object(Bucket=bucket, Key=WEIGHTS_KEY)
        print(f"sam3.pt already staged in s3 ({head['ContentLength'] / 1e9:.2f} GB)")
        return
    except Exception:
        pass
    if not WEIGHTS_LOCAL.exists():
        sys.exit(f"sam3.pt not found at {WEIGHTS_LOCAL} and not staged in S3 — aborting.")
    size = WEIGHTS_LOCAL.stat().st_size
    print(f"uploading sam3.pt ({size / 1e9:.2f} GB) to s3://{bucket}/{WEIGHTS_KEY} "
          "(one-time; multipart)...")
    s3.upload_file(str(WEIGHTS_LOCAL), bucket, WEIGHTS_KEY)
    print("sam3.pt staged.")


def _guard_rotation(aws: dict, a) -> None:
    """Same fail-closed rotation guard as aws_train.py (docs/11)."""
    import boto3
    flagged = str(aws.get("flagged_account", "")).strip()
    rotated = a.i_rotated_creds or os.environ.get("UBALL_AWS_CREDS_ROTATED") == "1"
    if not flagged:
        sys.exit("configs/train_rfdetr.yaml aws.flagged_account is empty -- refusing to launch.")
    ident = boto3.client("sts", region_name=aws.get("region", "us-east-1")).get_caller_identity()
    account = ident.get("Account", "?")
    print(f"AWS account: {account}")
    if account == flagged and not rotated:
        sys.exit(f"REFUSING TO LAUNCH: account {account} keys are flagged for rotation "
                 "(docs/11). Rotate them, then pass --i-rotated-creds or set "
                 "UBALL_AWS_CREDS_ROTATED=1.")


def bundle(clips: list[str]) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "sam3_bundle.tar.gz"
    with tarfile.open(tmp, "w:gz") as t:
        t.add(REPO / "scripts" / "sam3_reference.py", arcname="sam3_reference.py")
        for c in clips:
            p = REPO / c
            if not p.exists():
                sys.exit(f"missing clip: {p}")
            t.add(p, arcname=f"clips/{p.name}")
    return tmp


def userdata(bundle_url: str, weights_url: str, results_url: str, log_url: str) -> str:
    return f"""#!/bin/bash
exec > /var/log/sam3.log 2>&1
export HOME=/root PYTHONUNBUFFERED=1
LOG_URL="{log_url}"
(while true; do sleep 30; curl -s -T /var/log/sam3.log "$LOG_URL" >/dev/null 2>&1 || true; done) &
PYBIN=""
for P in /opt/pytorch/bin/python /usr/bin/python3; do
  if [ -x "$P" ] && $P -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then PYBIN=$P; break; fi
done
[ -z "$PYBIN" ] && PYBIN=/usr/bin/python3
$PYBIN -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
$PYBIN -m pip install -q -U ultralytics opencv-python-headless
$PYBIN -c "import ultralytics;print('ultralytics',ultralytics.__version__)"
mkdir -p /work && cd /work
echo "[boot] fetch bundle + sam3.pt"
curl -s -L "{bundle_url}" -o b.tgz && tar xzf b.tgz
curl -s -L "{weights_url}" -o sam3.pt
ls -la sam3.pt
mkdir -p out
RC_ALL=0
for C in clips/*.mp4; do
  B=$(basename "$C" .mp4)
  echo "=== SAM3 on $B ==="
  $PYBIN sam3_reference.py --video "$C" --out "out/$B.sam3.json" --weights /work/sam3.pt || RC_ALL=1
done
tar czf results.tar.gz -C out .
CODE=$(curl -sS --max-time 1800 -w '%{{http_code}}' -o /dev/null -T results.tar.gz "{results_url}")
echo "[boot] results upload http=$CODE rc_all=$RC_ALL"
curl -s -T /var/log/sam3.log "$LOG_URL" >/dev/null 2>&1 || true
sleep 5; shutdown -h now
"""


def launch(a) -> None:
    import boto3
    aws = _aws_cfg()
    _guard_rotation(aws, a)
    region, bucket = aws.get("region", "us-east-1"), aws.get("s3_bucket")
    s3 = boto3.client("s3", region_name=region)
    _ensure_weights(s3, bucket)
    b = bundle(a.clips)
    bundle_key = f"{PREFIX}/sam3_bundle.tar.gz"
    print(f"uploading bundle ({b.stat().st_size // 1_000_000} MB, {len(a.clips)} clips)...")
    s3.upload_file(str(b), bucket, bundle_key)

    def presign(op, key, exp):
        return s3.generate_presigned_url("get_object" if op == "get" else "put_object",
                                         Params={"Bucket": bucket, "Key": key}, ExpiresIn=exp)

    ud = userdata(presign("get", bundle_key, 28800), presign("get", WEIGHTS_KEY, 28800),
                  presign("put", RESULTS_KEY, 86400), presign("put", LOG_KEY, 86400))
    ec2 = boto3.client("ec2", region_name=region)
    r = ec2.run_instances(
        ImageId=aws.get("ami"), InstanceType=aws.get("instance_type", "g5.2xlarge"),
        MinCount=1, MaxCount=1, InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                              "Ebs": {"VolumeSize": 100, "VolumeType": "gp3",
                                      "DeleteOnTermination": True}}],
        UserData=ud,
        TagSpecifications=[{"ResourceType": "instance",
                            "Tags": [{"Key": "Name", "Value": "uball-sam3-reference"}]}])
    iid = r["Instances"][0]["InstanceId"]
    print(f"launched {iid} ({aws.get('instance_type')}) — self-terminates when done")
    print(f"log:     s3://{bucket}/{LOG_KEY}")
    print(f"results: s3://{bucket}/{RESULTS_KEY}   (then: --fetch)")


def fetch(a) -> None:
    import boto3
    aws = _aws_cfg()
    s3 = boto3.client("s3", region_name=aws.get("region", "us-east-1"))
    out = REPO / "runs" / "sam3_ref"
    out.mkdir(parents=True, exist_ok=True)
    tar_path = out / "results.tar.gz"
    s3.download_file(aws.get("s3_bucket"), RESULTS_KEY, str(tar_path))
    with tarfile.open(tar_path) as t:
        t.extractall(out, filter="data")
    print(f"results -> {out}: {sorted(p.name for p in out.glob('*.sam3.json'))}")


def log(a) -> None:
    import boto3
    aws = _aws_cfg()
    s3 = boto3.client("s3", region_name=aws.get("region", "us-east-1"))
    body = s3.get_object(Bucket=aws.get("s3_bucket"), Key=LOG_KEY)["Body"].read().decode()
    print(body[-int(a.tail):])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=DEFAULT_CLIPS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--tail", type=int, default=4000)
    ap.add_argument("--i-rotated-creds", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        aws = _aws_cfg()
        b = bundle(a.clips)
        w = (f"local {WEIGHTS_LOCAL.stat().st_size / 1e9:.2f} GB" if WEIGHTS_LOCAL.exists()
             else "LOCAL COPY MISSING (ok if already staged in S3)")
        print(f"DRY RUN — bundle {b.stat().st_size // 1_000_000} MB ({len(a.clips)} clips), "
              f"instance {aws.get('instance_type')} in {aws.get('region')}, "
              f"s3://{aws.get('s3_bucket')}/{PREFIX}/ | sam3.pt: {w}")
    elif a.fetch:
        fetch(a)
    elif a.log:
        log(a)
    else:
        launch(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
