#!/bin/bash
# End-to-end pipeline on ONE game window: pull -> track -> teams -> jerseys -> reid ->
# fuse -> scorecard row. Usage: run_window.sh <gid8> <start_s> [dur]
set -e
cd "$(dirname "$0")/.."
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0
GID=$1; START=$2; DUR=${3:-14}
TAG=${START}_${DUR}
W=${UBALL_WEIGHTS:-runs/rfdetr-s-1280-ourdata-v1/best.pth}
REID_W=$(python -c "from uball_cc.tracking.reid import default_reid_weights; print(default_reid_weights())")

echo "== [$GID $TAG] pull =="
for ANG in FL FR NL NR; do
  C=data/clips/${GID}_${ANG}_${TAG}.mp4
  [ -f "$C" ] || python scripts/pull_clip.py --gid8 $GID --angle $ANG --start $START --dur $DUR --out "$C"
done
echo "== [$GID $TAG] per-camera =="
for ANG in FL FR NL NR; do
  T=runs/tracking/${GID}_${ANG}_${TAG}.json
  [ -f "$T" ] || python scripts/track.py --video data/clips/${GID}_${ANG}_${TAG}.mp4 --cam $ANG --weights $W --model small --out "$T"
  TT=runs/tracking/${GID}_${ANG}_${TAG}_teams.json
  [ -f "$TT" ] || python scripts/assign_teams.py --video data/clips/${GID}_${ANG}_${TAG}.mp4 --tracks "$T" --out "$TT"
  if [ "$ANG" = "NL" ] || [ "$ANG" = "NR" ]; then
    python scripts/read_jerseys_stack.py --video data/clips/${GID}_${ANG}_${TAG}.mp4 --tracks "$TT"
  fi
  R=runs/tracking/${GID}_${ANG}_${TAG}_reid.npz
  [ -f "$R" ] || python scripts/extract_reid.py --video data/clips/${GID}_${ANG}_${TAG}.mp4 --tracks "$TT" --model-path "$REID_W" --out "$R"
done
echo "== [$GID $TAG] fuse =="
python scripts/fuse_cams.py --ref FL \
  --cam FL runs/tracking/${GID}_FL_${TAG}_teams.json configs/calib/FL.json data/clips/${GID}_FL_${TAG}.mp4 \
  --cam FR runs/tracking/${GID}_FR_${TAG}_teams.json configs/calib/FR.json data/clips/${GID}_FR_${TAG}.mp4 \
  --cam NL runs/tracking/${GID}_NL_${TAG}_teams.json configs/calib/NL.json data/clips/${GID}_NL_${TAG}.mp4 \
  --cam NR runs/tracking/${GID}_NR_${TAG}_teams.json configs/calib/NR.json data/clips/${GID}_NR_${TAG}.mp4 \
  --save-worldstate runs/tracking/${GID}_${TAG}_worldstate.json
echo "== [$GID $TAG] scorecard row =="
python - "$GID" "$TAG" <<'PY'
import json, sys
import statistics as st
from collections import Counter
gid, tag = sys.argv[1], sys.argv[2]
ws = json.loads(open(f"runs/tracking/{gid}_{tag}_worldstate.json").read())
frames = ws["frames"]
core = frames[15:-15] if len(frames) > 40 else frames
ppf = [len(f["tracks"]) for f in core]
life = Counter()
for f in frames:
    for t in f["tracks"]:
        life[t["global_id"]] += 1
teams = Counter(p.get("team") for p in ws["players"])
named = [(p["team"], p["jersey"]) for p in ws["players"] if p["jersey"] is not None]
row = {"window": f"{gid}_{tag}", "ids": ws["n_global_ids"],
       "stable": sum(1 for c in life.values() if c >= 0.5 * len(frames)),
       "ghosts": sum(1 for c in life.values() if c < 0.1 * len(frames)),
       "ppf_med": st.median(ppf), "teams": dict(teams),
       "named": [f"{t} #{n}" for t, n in sorted(named, key=lambda r: (str(r[0]), r[1]))]}
print("SCORECARD:", json.dumps(row))
scpath = "runs/tracking/scorecard.jsonl"
with open(scpath, "a") as fh:
    fh.write(json.dumps(row) + "\n")
PY
