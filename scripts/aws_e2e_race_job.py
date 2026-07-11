#!/usr/bin/env python3
"""Launch the END-TO-END detector race on one A10G: raw video -> full pipeline ->
GT accuracy, per detector, nothing cached across lanes (see scripts/e2e_race.py).

Est. ~60-80 min ≈ $1.3-1.6; 2.5h failsafe caps $3.03.

  export UBALL_AWS_CREDS_ROTATED=1
  python scripts/aws_e2e_race_job.py --i-rotated-creds
  python scripts/aws_e2e_race_job.py --fetch
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
GAME, TAG = "c2a354fe", "300_60"
STAGE_SCRIPTS = ("e2e_race.py", "extract_jersey_anchors.py", "extract_pose.py",
                 "annotate_anchor_kits.py", "hybrid_track.py", "solve_player_xcam.py")
LANES = (("rfdetr_s_fp16", "runs/rfdetr-s-1280-ourdata-v1/best.pth"),
         ("yolo11s", "runs/yolo11s-1280-ourdata-v1_fetch/runs/detect/runs/yolo11s-1280-ourdata-v1/weights/best.pt"),
         ("yolo11m", "runs/yolo11m-1280-ourdata-v1_fetch/runs/detect/runs/yolo11m-1280-ourdata-v1/weights/best.pt"),
         ("yolo26s", "runs/yolo26s-1280-ourdata-v1_fetch/runs/detect/runs/yolo26s-1280-ourdata-v1/weights/best.pt"))


def userdata(bundle_url, results_url, log_url) -> str:
    lanes_arg = ",".join(
        f"{n}:weights/{n}{'.pth' if n.startswith('rfdetr') else '.pt'}" for n, _ in LANES)
    return f"""#!/bin/bash
exec > /var/log/e2e.log 2>&1
export HOME=/root PYTHONUNBUFFERED=1 YOLO_CONFIG_DIR=/tmp/Ultralytics
LOG_URL="{log_url}"
(while true; do sleep 30; curl -s -T /var/log/e2e.log "$LOG_URL" >/dev/null 2>&1 || true; done) &
(sleep 9000; echo "[boot] 2.5h failsafe"; shutdown -h now) &
PYBIN=""
for P in /opt/pytorch/bin/python /usr/bin/python3; do
  if [ -x "$P" ] && $P -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then PYBIN=$P; break; fi
done
[ -z "$PYBIN" ] && PYBIN=/usr/bin/python3
TV=$($PYBIN -c "import torch;print(torch.__version__.split('+')[0])")
$PYBIN -m pip install -q "torch==$TV" rfdetr ultralytics supervision trackers \\
  opencv-python-headless timm pytorch-lightning nltk rtmlib onnxruntime-gpu 2>&1 | tail -1
$PYBIN -m pip uninstall -q -y torchaudio 2>/dev/null || true
which ffmpeg || (apt-get update -qq && apt-get install -y -qq ffmpeg)
mkdir -p /work && cd /work
curl -s -L "{bundle_url}" -o b.tgz && tar xzf b.tgz
export PYTHONPATH=/work/src
mkdir -p runs/dets_cache runs/anchors runs/pose_cache runs/jersey runs/tracking/ledger out
cp jersey/*.pt runs/jersey/
(while true; do sleep 180; tar czf results.tar.gz out 2>/dev/null; \\
 curl -s -o /dev/null -T results.tar.gz "{results_url}" || true; done) &
RC=0
$PYBIN scripts/e2e_race.py --game {GAME} --tag {TAG} --device cuda \\
  --dual-numbers 1,3,5 --gt-ref gt_ref --lanes "{lanes_arg}" || RC=1
tar czf results.tar.gz out
CODE=$(curl -sS --max-time 1800 -w '%{{http_code}}' -o /dev/null -T results.tar.gz "{results_url}")
echo "[boot] rc=$RC upload http=$CODE"
curl -s -T /var/log/e2e.log "$LOG_URL" >/dev/null 2>&1 || true
sleep 5; shutdown -h now
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--i-rotated-creds", action="store_true")
    a = ap.parse_args()
    import boto3
    aws = J._aws_cfg()
    region, bucket = aws.get("region", "us-east-1"), aws.get("s3_bucket")
    s3 = boto3.client("s3", region_name=region)
    if a.fetch:
        tmp = Path(tempfile.mkdtemp()) / "r.tar.gz"
        s3.download_file(bucket, J.results_key("e2e"), str(tmp))
        dst = REPO / "runs/e2e_race_fetch"
        dst.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tmp) as t:
            t.extractall(dst)
        print(f"fetched -> {dst}")
        rj = dst / "out/e2e_results.json"
        if rj.exists():
            print(rj.read_text())
        return 0
    J._guard_rotation(aws, a)

    tmp = Path(tempfile.mkdtemp()) / "e2e_bundle.tar.gz"
    print("bundling clips + 4 detectors + jersey stack + GT + sources...")
    with tarfile.open(tmp, "w:gz") as t:
        for ang in ANGLES:
            t.add(REPO / f"data/clips/{GAME}_{ang}_{TAG}.mp4",
                  arcname=f"data/clips/{GAME}_{ang}_{TAG}.mp4")
        for name, w in LANES:
            ext = ".pth" if name.startswith("rfdetr") else ".pt"
            t.add(REPO / w, arcname=f"weights/{name}{ext}")
        for w in (REPO / "runs/jersey").glob("*.pt"):
            t.add(w, arcname=f"jersey/{w.name}")
        for s in STAGE_SCRIPTS:
            t.add(REPO / "scripts" / s, arcname=f"scripts/{s}")
        for c in (REPO / "configs/calib").glob("*.json"):
            t.add(c, arcname=f"configs/calib/{c.name}")
        t.add(REPO / f"data/gt_players/{GAME}_{TAG}.json",
              arcname=f"data/gt_players/{GAME}_{TAG}.json")
        # ORIGINAL dets the GT indices reference — grading only
        for ang in ANGLES:
            f = f"{GAME}_{ang}_{TAG}_small_1280_t0.25.dets.npz"
            t.add(REPO / "runs/dets_cache" / f, arcname=f"gt_ref/{f}")
        def flt(ti):
            return None if "__pycache__" in ti.name else ti
        t.add(REPO / "src", arcname="src", filter=flt)
    key = f"{J.PREFIX}/e2e_bundle.tar.gz"
    print(f"uploading bundle ({tmp.stat().st_size // 1_000_000} MB)...")
    s3.upload_file(str(tmp), bucket, key)

    def presign(op, key_, exp):
        return s3.generate_presigned_url("get_object" if op == "get" else "put_object",
                                         Params={"Bucket": bucket, "Key": key_}, ExpiresIn=exp)

    ud = userdata(presign("get", key, 28800), presign("put", J.results_key("e2e"), 86400),
                  presign("put", J.log_key("e2e"), 86400))
    ec2 = boto3.client("ec2", region_name=region)
    r = ec2.run_instances(
        ImageId=aws.get("ami"), InstanceType=aws.get("instance_type", "g5.2xlarge"),
        MinCount=1, MaxCount=1, InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                              "Ebs": {"VolumeSize": 100, "VolumeType": "gp3",
                                      "DeleteOnTermination": True}}],
        UserData=ud,
        TagSpecifications=[{"ResourceType": "instance",
                            "Tags": [{"Key": "Name", "Value": "uball-e2e-race"}]}])
    print(f"launched {r['Instances'][0]['InstanceId']} — e2e race, 2.5h failsafe")
    print(f"log: s3://{bucket}/{J.log_key('e2e')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
