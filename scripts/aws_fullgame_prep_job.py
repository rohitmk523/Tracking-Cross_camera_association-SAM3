#!/usr/bin/env python3
"""FULL-GAME cache prep for the EVENTS workstream: detection (yolo26s, all classes
incl. BALL) + dense stride-1 batched jersey anchors + pose, for the whole game,
chunked across N instances by time.

Each instance slices its chunks from the full-game S3 videos via presigned URLs
(ffmpeg -ss over https, in-region), then per chunk: detect -> pose -> anchors
(Phase-1 batched stack, inline shades, explicit camera-clock offsets). Caches
upload incrementally per chunk.

Cost: ~6h GPU total for a ~55-min game; 3 instances ~2.2h wall ≈ $8.
(User-approved beyond the $5 cap for the events workstream, 2026-07-11.)

  export UBALL_AWS_CREDS_ROTATED=1
  python scripts/aws_fullgame_prep_job.py --gid8 e6fba750 --duration-s 3400 \
      --instances 3 --i-rotated-creds
  python scripts/aws_fullgame_prep_job.py --gid8 e6fba750 --fetch
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
CHUNK_S = 600                                   # 10-min chunks
OFFSETS = {"e6fba750": {"FL": 0, "FR": -11, "NL": -1, "NR": -1},
           "c2a354fe": {"FL": 0, "FR": 1, "NL": 2, "NR": -1}}
WEIGHTS = "runs/yolo26s-1280-ourdata-v1_fetch/runs/detect/runs/yolo26s-1280-ourdata-v1/weights/best.pt"


def userdata(bundle_url, video_urls, chunks, results_url, log_url, gid8, offsets) -> str:
    dls = "\n".join(
        f'curl -s -L "{u}" -o "videos/{gid8}_{ang}.mp4" & DLPIDS="$DLPIDS $!"'
        for ang, u in video_urls.items())
    chunk_lines = " ".join(f"{s}_{d}" for s, d in chunks)
    return f"""#!/bin/bash
exec > /var/log/prep.log 2>&1
export HOME=/root PYTHONUNBUFFERED=1 YOLO_CONFIG_DIR=/tmp/Ultralytics
LOG_URL="{log_url}"
(while true; do sleep 30; curl -s -T /var/log/prep.log "$LOG_URL" >/dev/null 2>&1 || true; done) &
(sleep 14400; echo "[boot] 4h failsafe"; shutdown -h now) &
PYBIN=""
for P in /opt/pytorch/bin/python /usr/bin/python3; do
  if [ -x "$P" ] && $P -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then PYBIN=$P; break; fi
done
[ -z "$PYBIN" ] && PYBIN=/usr/bin/python3
TV=$($PYBIN -c "import torch;print(torch.__version__.split('+')[0])")
$PYBIN -m pip install -q "torch==$TV" ultralytics pytorch-lightning nltk rtmlib \\
  "onnxruntime-gpu==1.20.1" opencv-python-headless timm 2>&1 | tail -1
$PYBIN -m pip uninstall -q -y torchaudio 2>/dev/null || true
NVLIB=$($PYBIN -c "import os, glob, torch; b=os.path.dirname(os.path.dirname(torch.__file__)); print(':'.join(sorted(glob.glob(os.path.join(b,'nvidia','*','lib')))))")
export LD_LIBRARY_PATH="$NVLIB:$LD_LIBRARY_PATH"
which ffmpeg || (apt-get update -qq && apt-get install -y -qq ffmpeg)
mkdir -p /work && cd /work
curl -s -L "{bundle_url}" -o b.tgz && tar xzf b.tgz
export PYTHONPATH=/work/src
mkdir -p videos data/clips runs/dets_cache runs/anchors runs/pose_cache runs/jersey out
cp jersey/*.pt runs/jersey/
echo "[dl] full-game videos (in-region)"
DLPIDS=""
{dls}
wait $DLPIDS
ls -la videos/
RC=0
for CH in {chunk_lines}; do
  S="${{CH%_*}}"; D="${{CH#*_}}"
  echo "===== CHUNK $CH ====="
  for ANG in FL FR NL NR; do
    ffmpeg -hide_banner -loglevel error -ss "$S" -i "videos/{gid8}_${{ANG}}.mp4" \\
      -t "$D" -c copy -y "data/clips/{gid8}_${{ANG}}_${{CH}}.mp4" || RC=1
  done
  $PYBIN scripts/build_dets_cache_yolo.py --game {gid8} --tag "$CH" \\
    --weights weights/yolo26s.pt --out-dir runs/dets_cache --device cuda 2>&1 | grep "detections ->" || RC=1
  $PYBIN scripts/extract_pose.py --game {gid8} --tag "$CH" --device cuda 2>&1 | grep "poses in" || RC=1
  $PYBIN scripts/extract_jersey_anchors.py --game {gid8} --tag "$CH" --stride 1 \\
    --offsets '{offsets}' 2>&1 | grep -E "TOTAL|confident" || RC=1
  rm -f data/clips/{gid8}_*_"$CH".mp4
  tar czf results.tar.gz runs/dets_cache runs/anchors runs/pose_cache
  curl -sS -o /dev/null -T results.tar.gz "{results_url}" || true
  echo "chunk $CH done, uploaded"
done
echo "[boot] rc=$RC ALL CHUNKS DONE"
curl -s -T /var/log/prep.log "$LOG_URL" >/dev/null 2>&1 || true
sleep 5; shutdown -h now
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid8", default="e6fba750")
    ap.add_argument("--duration-s", type=int, default=3400)
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--i-rotated-creds", action="store_true")
    a = ap.parse_args()
    import boto3
    aws = J._aws_cfg()
    region, bucket = aws.get("region", "us-east-1"), aws.get("s3_bucket")
    s3 = boto3.client("s3", region_name=region)

    if a.fetch:
        dst = REPO / f"runs/fullgame_{a.gid8}"
        dst.mkdir(parents=True, exist_ok=True)
        for i in range(a.instances):
            tmp = Path(tempfile.mkdtemp()) / "r.tar.gz"
            try:
                s3.download_file(bucket, J.results_key(f"fg{a.gid8[:3]}_{i}"), str(tmp))
            except Exception as e:
                print(f"instance {i}: no results ({e})")
                continue
            with tarfile.open(tmp) as t:
                t.extractall(dst)
            print(f"instance {i} -> {dst}")
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

    video_urls = {ang: presign("get", f"{pfx}/{date}_{full}_{ang}.mp4", 28800)
                  for ang in ANGLES}

    tmp = Path(tempfile.mkdtemp()) / "fg_bundle.tar.gz"
    print("bundling weights + scripts + src...")
    with tarfile.open(tmp, "w:gz") as t:
        t.add(REPO / WEIGHTS, arcname="weights/yolo26s.pt")
        for w in (REPO / "runs/jersey").glob("*.pt"):
            t.add(w, arcname=f"jersey/{w.name}")
        for scr in ("build_dets_cache_yolo.py", "extract_pose.py", "extract_jersey_anchors.py"):
            t.add(REPO / "scripts" / scr, arcname=f"scripts/{scr}")
        def flt(ti):
            return None if "__pycache__" in ti.name else ti
        t.add(REPO / "src", arcname="src", filter=flt)
    key = f"{J.PREFIX}/fg_bundle.tar.gz"
    print(f"uploading bundle ({tmp.stat().st_size // 1_000_000} MB)...")
    s3.upload_file(str(tmp), bucket, key)
    bundle_url = presign("get", key, 28800)

    all_chunks = [(s, min(CHUNK_S, a.duration_s - s)) for s in range(0, a.duration_s, CHUNK_S)]
    per_inst = [all_chunks[i::a.instances] for i in range(a.instances)]
    ec2 = boto3.client("ec2", region_name=region)
    for i, chunks in enumerate(per_inst):
        ud = userdata(bundle_url, video_urls, chunks,
                      presign("put", J.results_key(f"fg{a.gid8[:3]}_{i}"), 86400),
                      presign("put", J.log_key(f"fg{a.gid8[:3]}_{i}"), 86400),
                      a.gid8, offsets)
        r = ec2.run_instances(
            ImageId=aws.get("ami"), InstanceType=aws.get("instance_type", "g5.2xlarge"),
            MinCount=1, MaxCount=1, InstanceInitiatedShutdownBehavior="terminate",
            BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                                  "Ebs": {"VolumeSize": 150, "VolumeType": "gp3",
                                          "DeleteOnTermination": True}}],
            UserData=ud,
            TagSpecifications=[{"ResourceType": "instance",
                                "Tags": [{"Key": "Name", "Value": f"uball-fg-{a.gid8[:3]}-{i}"}]}])
        iid = r["Instances"][0]["InstanceId"]
        print(f"instance {i}: {iid} chunks={[f'{s}_{d}' for s, d in chunks]}")
        print(f"  log: s3://{bucket}/{J.log_key(f'fg{a.gid8[:3]}_{i}')}")
    print(f"total {len(all_chunks)} chunks across {a.instances} instances, 4h failsafe each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
