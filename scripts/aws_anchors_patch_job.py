#!/usr/bin/env python3
"""Anchors-only patch job: the full-game prep's detection+pose succeeded but the
OCR-anchors stage crashed on an offsets-quoting bug. This job re-runs ONLY the
batched anchors for all chunks on one instance: bundles the already-fetched dets
and pose caches, slices chunk clips from the S3 full-game videos, extracts
stride-1 anchors with inline shades. ~30 min ≈ $0.6.

  export UBALL_AWS_CREDS_ROTATED=1
  python scripts/aws_anchors_patch_job.py --gid8 e6fba750 --i-rotated-creds
  python scripts/aws_anchors_patch_job.py --gid8 e6fba750 --fetch
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
CHUNKS = ("0_600", "600_600", "1200_600", "1800_600", "2400_600", "3000_345")
OFFSETS = {"e6fba750": {"FL": 0, "FR": -11, "NL": -1, "NR": -1}}
FG_DIR = "runs/fullgame_{gid8}/runs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid8", default="e6fba750")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--i-rotated-creds", action="store_true")
    a = ap.parse_args()
    import boto3
    aws = J._aws_cfg()
    region, bucket = aws.get("region", "us-east-1"), aws.get("s3_bucket")
    s3 = boto3.client("s3", region_name=region)
    tag2 = f"anch{a.gid8[:3]}"
    if a.fetch:
        tmp = Path(tempfile.mkdtemp()) / "r.tar.gz"
        s3.download_file(bucket, J.results_key(tag2), str(tmp))
        dst = REPO / f"runs/fullgame_{a.gid8}"
        with tarfile.open(tmp) as t:
            t.extractall(dst)
        print(f"fetched -> {dst}/runs/anchors")
        return 0
    J._guard_rotation(aws, a)

    gj = json.loads((REPO / "configs/games.json").read_text())
    g = next(x for x in gj["working_games"] if x["gid8"] == a.gid8)
    pfx = g["s3_prefix"].rstrip("/")
    _, date, full = pfx.split("/")
    offsets = json.dumps(OFFSETS[a.gid8])

    def presign(op, key_, exp):
        return s3.generate_presigned_url("get_object" if op == "get" else "put_object",
                                         Params={"Bucket": bucket, "Key": key_}, ExpiresIn=exp)

    video_urls = {ang: presign("get", f"{pfx}/{date}_{full}_{ang}.mp4", 28800) for ang in ANGLES}
    fg = REPO / FG_DIR.format(gid8=a.gid8)
    tmp = Path(tempfile.mkdtemp()) / "anch_bundle.tar.gz"
    print("bundling dets+pose caches + jersey stack + scripts + src...")
    with tarfile.open(tmp, "w:gz") as t:
        for ch in CHUNKS:
            for ang in ANGLES:
                t.add(fg / "dets_cache" / f"{a.gid8}_{ang}_{ch}_small_1280_t0.25.dets.npz",
                      arcname=f"runs/dets_cache/{a.gid8}_{ang}_{ch}_small_1280_t0.25.dets.npz")
                t.add(fg / "pose_cache" / f"{a.gid8}_{ang}_{ch}.pose.npz",
                      arcname=f"runs/pose_cache/{a.gid8}_{ang}_{ch}.pose.npz")
        for w in (REPO / "runs/jersey").glob("*.pt"):
            t.add(w, arcname=f"jersey/{w.name}")
        t.add(REPO / "scripts/extract_jersey_anchors.py",
              arcname="scripts/extract_jersey_anchors.py")
        def flt(ti):
            return None if "__pycache__" in ti.name else ti
        t.add(REPO / "src", arcname="src", filter=flt)
    key = f"{J.PREFIX}/anch_bundle.tar.gz"
    print(f"uploading bundle ({tmp.stat().st_size // 1_000_000} MB)...")
    s3.upload_file(str(tmp), bucket, key)

    dls = "\n".join(f'curl -s -L "{u}" -o "videos/{a.gid8}_{ang}.mp4" & DLPIDS="$DLPIDS $!"'
                    for ang, u in video_urls.items())
    chunk_lines = " ".join(CHUNKS)
    ud = f"""#!/bin/bash
exec > /var/log/anch.log 2>&1
export HOME=/root PYTHONUNBUFFERED=1 YOLO_CONFIG_DIR=/tmp/Ultralytics
LOG_URL="{presign('put', J.log_key(tag2), 86400)}"
(while true; do sleep 30; curl -s -T /var/log/anch.log "$LOG_URL" >/dev/null 2>&1 || true; done) &
(sleep 7200; echo "[boot] 2h failsafe"; shutdown -h now) &
PYBIN=""
for P in /opt/pytorch/bin/python /usr/bin/python3; do
  if [ -x "$P" ] && $P -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then PYBIN=$P; break; fi
done
[ -z "$PYBIN" ] && PYBIN=/usr/bin/python3
TV=$($PYBIN -c "import torch;print(torch.__version__.split('+')[0])")
$PYBIN -m pip install -q "torch==$TV" ultralytics pytorch-lightning nltk opencv-python-headless timm 2>&1 | tail -1
$PYBIN -m pip uninstall -q -y torchaudio 2>/dev/null || true
which ffmpeg || (apt-get update -qq && apt-get install -y -qq ffmpeg)
mkdir -p /work && cd /work
curl -s -L "{presign('get', key, 28800)}" -o b.tgz && tar xzf b.tgz
export PYTHONPATH=/work/src
mkdir -p videos data/clips runs/anchors runs/jersey
cp jersey/*.pt runs/jersey/
DLPIDS=""
{dls}
wait $DLPIDS
ls -la videos/
RC=0
for CH in {chunk_lines}; do
  S="${{CH%_*}}"; D="${{CH#*_}}"
  echo "===== ANCHORS $CH ====="
  for ANG in FL FR NL NR; do
    ffmpeg -hide_banner -loglevel error -ss "$S" -i "videos/{a.gid8}_${{ANG}}.mp4" \\
      -t "$D" -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -an \\
      -y "data/clips/{a.gid8}_${{ANG}}_${{CH}}.mp4" || RC=1
  done
  $PYBIN scripts/extract_jersey_anchors.py --game {a.gid8} --tag "$CH" --stride 1 \\
    --offsets '{offsets}' || RC=1
  rm -f data/clips/{a.gid8}_*_"$CH".mp4
  tar czf results.tar.gz runs/anchors
  curl -sS -o /dev/null -T results.tar.gz "{presign('put', J.results_key(tag2), 86400)}" || true
done
echo "[boot] rc=$RC ANCHORS ALL DONE"
curl -s -T /var/log/anch.log "$LOG_URL" >/dev/null 2>&1 || true
sleep 5; shutdown -h now
"""
    ec2 = boto3.client("ec2", region_name=region)
    r = ec2.run_instances(
        ImageId=aws.get("ami"), InstanceType=aws.get("instance_type", "g5.2xlarge"),
        MinCount=1, MaxCount=1, InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                              "Ebs": {"VolumeSize": 120, "VolumeType": "gp3",
                                      "DeleteOnTermination": True}}],
        UserData=ud,
        TagSpecifications=[{"ResourceType": "instance",
                            "Tags": [{"Key": "Name", "Value": f"uball-anch-{a.gid8[:3]}"}]}])
    print(f"launched {r['Instances'][0]['InstanceId']} — anchors patch, 2h failsafe")
    print(f"log: s3://{bucket}/{J.log_key(tag2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
