#!/usr/bin/env python3
"""Visual confirmation: run the NEW RF-DETR near-angle ball+hoop model on a clip
and write an annotated MP4 (+ sample frames). Mirrors the detect->draw->write path
of uball_shot_detection_dual_fusion_v2 (class 0=Basketball/green, 1=Hoop/blue,
full-frame inference) but with RF-DETR instead of the YOLO .pt model.

  python scripts/viz_near_rfdetr.py --video clip.mp4 --out annotated.mp4

NO ultralytics/YOLO anywhere.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BALL, HOOP = 0, 1
COL = {BALL: (90, 220, 90), HOOP: (40, 150, 255)}   # BGR: ball=green, hoop=blue
NAME = {BALL: "ball", HOOP: "hoop"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default=str(REPO / "runs/rfdetr-rim-near-v1/best.pth"))
    ap.add_argument("--out", default=str(REPO / "runs/viz/near_rfdetr_annotated.mp4"))
    ap.add_argument("--resolution", type=int, default=1280)   # matches RF-DETR training
    ap.add_argument("--conf", type=float, default=0.20)       # matches the repo's near conf
    ap.add_argument("--fps", type=float, default=6.0)         # output video fps
    ap.add_argument("--sample-dir", default=str(REPO / "runs/viz/near_rfdetr_frames"))
    a = ap.parse_args()

    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
    import cv2
    from rfdetr import RFDETRSmall

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.sample_dir).mkdir(parents=True, exist_ok=True)

    print(f"loading RF-DETR near @ {a.resolution}: {a.weights}")
    model = RFDETRSmall(pretrain_weights=a.weights, resolution=a.resolution)

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"clip: {w}x{h}, {n} frames")
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))

    stats = {"frames": 0, "ball_frames": 0, "hoop_frames": 0,
             "ball_conf_sum": 0.0, "hoop_conf_sum": 0.0}
    saved = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        d = model.predict(rgb, threshold=a.conf)
        has = {BALL: False, HOOP: False}
        for i in range(len(d.xyxy)):
            cls = int(d.class_id[i])
            if cls not in COL:
                continue
            conf = float(d.confidence[i])
            x1, y1, x2, y2 = (int(v) for v in d.xyxy[i])
            cv2.rectangle(frame, (x1, y1), (x2, y2), COL[cls], 3)
            cv2.putText(frame, f"{NAME[cls]} {conf:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL[cls], 2)
            has[cls] = True
            stats[f"{NAME[cls]}_conf_sum"] += conf
        stats["frames"] += 1
        stats["ball_frames"] += has[BALL]
        stats["hoop_frames"] += has[HOOP]
        cv2.putText(frame, f"RF-DETR near  f{idx}  ball={has[BALL]} hoop={has[HOOP]}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        vw.write(frame)
        # save a handful of sample frames where BOTH are detected (best evidence)
        if has[BALL] and has[HOOP] and saved < 8 and idx % 13 == 0:
            cv2.imwrite(str(Path(a.sample_dir) / f"f{idx:05d}.png"), frame)
            saved += 1
        if idx % 100 == 0:
            print(f"  {idx} frames | ball {stats['ball_frames']} hoop {stats['hoop_frames']}")
    cap.release()
    vw.release()

    f = max(1, stats["frames"])
    print("\n=== RF-DETR NEAR detection summary ===")
    print(f"  frames processed: {stats['frames']}")
    print(f"  ball detected:    {stats['ball_frames']} ({100*stats['ball_frames']/f:.1f}%)  "
          f"avg conf {stats['ball_conf_sum']/max(1,stats['ball_frames']):.2f}")
    print(f"  hoop detected:    {stats['hoop_frames']} ({100*stats['hoop_frames']/f:.1f}%)  "
          f"avg conf {stats['hoop_conf_sum']/max(1,stats['hoop_frames']):.2f}")
    print(f"  annotated video -> {a.out}")
    print(f"  sample frames   -> {a.sample_dir} ({saved} saved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
