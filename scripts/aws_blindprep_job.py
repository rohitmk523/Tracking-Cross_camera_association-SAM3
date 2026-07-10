#!/usr/bin/env python3
"""One-shot AWS prep for a blind game window: DETECTION cache + JERSEY anchors.

Bundles the detector (track.py + rfdetr weights), the jersey stack (legibility +
localizer + PARSeq weights), uball_cc sources and the 4 clips. The instance builds
the dets cache per camera, then runs the dense anchor extraction with offsets we
computed locally (audio sync). ~20-30 min on an A10G ≈ $0.6; 2h failsafe caps $2.45.

  export UBALL_AWS_CREDS_ROTATED=1
  python scripts/aws_blindprep_job.py --game f66eb3b2 --tag 60_300 \
      --offsets '{"FL":0,"FR":1,"NL":2,"NR":-1}' --i-rotated-creds
  python scripts/aws_blindprep_job.py --fetch --game f66eb3b2 --tag 60_300
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


def userdata(bundle_url, results_url, log_url, game, tag, offsets_json, anchors_only="0") -> str:
    cams = " ".join(ANGLES)
    return f"""#!/bin/bash
exec > /var/log/prep.log 2>&1
export HOME=/root PYTHONUNBUFFERED=1
LOG_URL="{log_url}"
(while true; do sleep 30; curl -s -T /var/log/prep.log "$LOG_URL" >/dev/null 2>&1 || true; done) &
(sleep 7200; echo "[boot] 2h failsafe shutdown"; shutdown -h now) &
PYBIN=""
for P in /opt/pytorch/bin/python /usr/bin/python3; do
  if [ -x "$P" ] && $P -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then PYBIN=$P; break; fi
done
[ -z "$PYBIN" ] && PYBIN=/usr/bin/python3
TV=$($PYBIN -c "import torch;print(torch.__version__.split('+')[0])")
$PYBIN -m pip install -q "torch==$TV" rfdetr supervision trackers opencv-python-headless timm 2>&1 | tail -1
$PYBIN -m pip uninstall -q -y torchaudio 2>/dev/null || true
which ffmpeg || (apt-get update -qq && apt-get install -y -qq ffmpeg)
$PYBIN -c "import torch; assert torch.cuda.is_available(); print('torch OK')"
mkdir -p /work && cd /work
curl -s -L "{bundle_url}" -o b.tgz && tar xzf b.tgz
export PYTHONPATH=/work/src
mkdir -p data && ln -sfn /work/clips /work/data/clips
mkdir -p runs/dets_cache runs/anchors runs/jersey out
cp jersey/*.pt runs/jersey/
RC=0
if [ "{anchors_only}" = "1" ]; then CAMS=""; else CAMS="{cams}"; fi
for CAM in $CAMS; do
  echo "=== detect $CAM ==="
  $PYBIN scripts/track.py --video "clips/{game}_${{CAM}}_{tag}.mp4" --cam $CAM \\
    --weights weights/best.pth --model small --resolution 1280 --threshold 0.25 \\
    --out "out/ignore_$CAM.jsonl" || RC=1
  # incremental upload of caches so far
  tar czf results.tar.gz runs/dets_cache runs/anchors 2>/dev/null
  curl -sS -o /dev/null -T results.tar.gz "{results_url}" || true
done
echo "=== anchors (stride 2) ==="
echo '{offsets_json}' > offs.json
$PYBIN scripts/extract_jersey_anchors.py --game {game} --tag {tag} --stride 2 || RC=1
tar czf results.tar.gz runs/dets_cache runs/anchors
CODE=$(curl -sS --max-time 1800 -w '%{{http_code}}' -o /dev/null -T results.tar.gz "{results_url}")
echo "[boot] results upload http=$CODE rc=$RC"
curl -s -T /var/log/prep.log "$LOG_URL" >/dev/null 2>&1 || true
sleep 5; shutdown -h now
"""


def launch(a) -> None:
    import boto3
    aws = J._aws_cfg()
    J._guard_rotation(aws, a)
    region, bucket = aws.get("region", "us-east-1"), aws.get("s3_bucket")
    s3 = boto3.client("s3", region_name=region)

    tmp = Path(tempfile.mkdtemp()) / "blindprep_bundle.tar.gz"
    print("bundling clips + detector + jersey stack + sources...")
    with tarfile.open(tmp, "w:gz") as t:
        for ang in ANGLES:
            t.add(REPO / f"data/clips/{a.game}_{ang}_{a.tag}.mp4",
                  arcname=f"clips/{a.game}_{ang}_{a.tag}.mp4")
        t.add(REPO / "scripts/track.py", arcname="scripts/track.py")
        t.add(REPO / "scripts/extract_jersey_anchors.py",
              arcname="scripts/extract_jersey_anchors.py")
        t.add(REPO / "runs/rfdetr-s-1280-ourdata-v1/best.pth", arcname="weights/best.pth")
        for w in (REPO / "runs/jersey").glob("*.pt"):
            t.add(w, arcname=f"jersey/{w.name}")
        def flt(ti):
            return None if "__pycache__" in ti.name else ti
        t.add(REPO / "src", arcname="src", filter=flt)
        if a.anchors_only:
            for c in (REPO / "runs/dets_cache").glob(f"{a.game}_*_{a.tag}_*.dets.npz"):
                t.add(c, arcname=f"runs/dets_cache/{c.name}")
    tag2 = f"prep_{a.game[:3]}"
    bundle_key = f"{J.PREFIX}/blindprep_bundle_{tag2}.tar.gz"
    print(f"uploading bundle ({tmp.stat().st_size // 1_000_000} MB)...")
    s3.upload_file(str(tmp), bucket, bundle_key)

    def presign(op, key_, exp):
        return s3.generate_presigned_url("get_object" if op == "get" else "put_object",
                                         Params={"Bucket": bucket, "Key": key_}, ExpiresIn=exp)

    ud = userdata(presign("get", bundle_key, 28800),
                  presign("put", J.results_key(tag2), 86400),
                  presign("put", J.log_key(tag2), 86400),
                  a.game, a.tag, a.offsets, "1" if a.anchors_only else "0")
    ec2 = boto3.client("ec2", region_name=region)
    r = ec2.run_instances(
        ImageId=aws.get("ami"), InstanceType=aws.get("instance_type", "g5.2xlarge"),
        MinCount=1, MaxCount=1, InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                              "Ebs": {"VolumeSize": 100, "VolumeType": "gp3",
                                      "DeleteOnTermination": True}}],
        UserData=ud,
        TagSpecifications=[{"ResourceType": "instance",
                            "Tags": [{"Key": "Name", "Value": f"uball-blindprep-{a.game[:3]}"}]}])
    iid = r["Instances"][0]["InstanceId"]
    print(f"launched {iid} — detection + anchors, incremental uploads, 2h failsafe")
    print(f"log:     s3://{bucket}/{J.log_key(tag2)}")
    print(f"results: s3://{bucket}/{J.results_key(tag2)}")


def fetch(a) -> None:
    import boto3
    aws = J._aws_cfg()
    region, bucket = aws.get("region", "us-east-1"), aws.get("s3_bucket")
    s3 = boto3.client("s3", region_name=region)
    tag2 = f"prep_{a.game[:3]}"
    tmp = Path(tempfile.mkdtemp()) / "r.tar.gz"
    s3.download_file(bucket, J.results_key(tag2), str(tmp))
    with tarfile.open(tmp) as t:
        t.extractall(REPO)                       # runs/dets_cache + runs/anchors
    print("fetched:")
    for p in sorted((REPO / "runs/dets_cache").glob(f"{a.game}_*_{a.tag}_*.npz")):
        print(" ", p.name)
    ap_ = REPO / f"runs/anchors/{a.game}_{a.tag}.jersey_anchors.json"
    if ap_.exists():
        d = json.loads(ap_.read_text())
        print(f"  anchors: {len(d['anchors'])} events, offsets {d['offsets']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--offsets", default="", help="JSON per-camera frame offsets (unused by extract, which audio-syncs)")
    ap.add_argument("--anchors-only", action="store_true", help="skip detection (bundle existing dets caches)")
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
