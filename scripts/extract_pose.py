#!/usr/bin/env python3
"""RTMPose keypoints for every cached player detection (the pose layer of the
post-SAM3 pipeline). Top-down on RF-DETR boxes via rtmlib (ONNX, no mmcv).

17 COCO keypoints per detection; ankles are 15 (L) and 16 (R). Output aligns 1:1
with the detection cache so downstream code can join on (frame, det_index).

Output: runs/pose_cache/{game}_{ang}_{tag}.pose.npz
  frame_idx (N,), det_idx (N,), kpts (N,17,2) image px, kscores (N,17)

  python scripts/extract_pose.py --game e6fba750 --tag 44_60
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--angles", default=",".join(ANGLES))
    ap.add_argument("--device", default="mps", help="rtmlib device: mps (CoreML) | cpu")
    ap.add_argument("--min-score", type=float, default=0.3, help="detection score floor")
    a = ap.parse_args()

    from rtmlib import RTMPose
    pose = RTMPose(
        onnx_model="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
                   "onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
        model_input_size=(192, 256), backend="onnxruntime", device=a.device)

    outd = REPO / "runs/pose_cache"
    outd.mkdir(parents=True, exist_ok=True)
    for ang in a.angles.split(","):
        z = np.load(REPO / f"runs/dets_cache/{a.game}_{ang}_{a.tag}_small_1280_t0.25.dets.npz")
        by_f: dict[int, list[tuple[int, list]]] = {}
        for di, (b, s, c, f) in enumerate(zip(z["boxes"], z["scores"], z["classes"], z["frame_idx"])):
            if int(c) == 0 and float(s) >= a.min_score:
                by_f.setdefault(int(f), []).append((di, [float(v) for v in b]))

        cap = cv2.VideoCapture(str(REPO / f"data/clips/{a.game}_{ang}_{a.tag}.mp4"))
        rows_f, rows_d, rows_k, rows_s = [], [], [], []
        t0, n_crops = time.time(), 0
        pos = -1
        for f in sorted(by_f):
            if f != pos + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            pos = f
            if not ok:
                continue
            dis = [di for di, _ in by_f[f]]
            boxes = [b for _, b in by_f[f]]
            kpts, kscores = pose(img, bboxes=boxes)
            for di, kp, ks in zip(dis, kpts, kscores):
                rows_f.append(f); rows_d.append(di)
                rows_k.append(kp.astype(np.float32)); rows_s.append(ks.astype(np.float32))
            n_crops += len(boxes)
            if f % 300 == 0:
                r = n_crops / max(1e-6, time.time() - t0)
                print(f"{ang} frame {f}: {n_crops} crops, {r:.0f} crops/s", flush=True)
        cap.release()
        out = outd / f"{a.game}_{ang}_{a.tag}.pose.npz"
        np.savez_compressed(out, frame_idx=np.array(rows_f), det_idx=np.array(rows_d),
                            kpts=np.stack(rows_k), kscores=np.stack(rows_s))
        print(f"{ang}: {len(rows_f)} poses in {time.time()-t0:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
