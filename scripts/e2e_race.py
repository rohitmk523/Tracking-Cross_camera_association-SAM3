#!/usr/bin/env python3
"""END-TO-END detector race: raw video -> full SAM3-free pipeline -> GT accuracy.

Nothing is inherited between lanes: every lane recomputes detection, jersey-OCR anchors,
pose, kit tags, tracking and cross-camera correction from its OWN detector's output.
The only shared inputs are the raw clips, the GT file, and gt_ref/ (the original
detector's boxes that the GT det-indices reference — grading only, never pipeline).

Window: c2a354fe_300_60 — one minute x 4 cameras with full human GT (#3W #3B #5B).

  python scripts/e2e_race.py --game c2a354fe --tag 300_60 --device cuda \
      --dual-numbers 1,3,5 --gt-ref gt_ref \
      --lanes rfdetr_s_fp16:weights/best.pth,yolo11s:weights/yolo11s_best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")
GAME_MIN = 40                                   # full-game extrapolation basis
GPU_USD_H = 1.212


def detect_lane(name: str, weights: str, game: str, tag: str, device: str) -> int:
    """Run the lane's detector over all 4 clips, write canonical dets npz. Returns frames."""
    import cv2
    outd = REPO / "runs/dets_cache"
    outd.mkdir(parents=True, exist_ok=True)
    total = 0
    if name.startswith("rfdetr"):
        from uball_cc.detection.base import RFDETRDetector   # FP16 auto on CUDA
        det = RFDETRDetector(weights, resolution=1280, threshold=0.25, model="small")
        def infer(img):
            return [(d.box_xyxy, d.score, d.class_id) for d in det.predict(img)]
    else:
        from ultralytics import YOLO
        model = YOLO(weights)
        def infer(img):
            r = model.predict(img, imgsz=1280, conf=0.25, half=(device == "cuda"),
                              device=device, verbose=False)[0]
            return list(zip(r.boxes.xyxy.cpu().numpy(),
                            r.boxes.conf.cpu().numpy(),
                            r.boxes.cls.cpu().numpy()))
    for ang in ANGLES:
        cap = cv2.VideoCapture(str(REPO / f"data/clips/{game}_{ang}_{tag}.mp4"))
        boxes, scores, classes, fidx = [], [], [], []
        f = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            for b, s, c in infer(img):
                boxes.append(np.asarray(b, dtype=np.float32))
                scores.append(float(s))
                classes.append(int(c))
                fidx.append(f)
            f += 1
        cap.release()
        total += f
        np.savez_compressed(
            outd / f"{game}_{ang}_{tag}_small_1280_t0.25.dets.npz",
            boxes=np.stack(boxes) if boxes else np.zeros((0, 4), np.float32),
            scores=np.array(scores), classes=np.array(classes),
            frame_idx=np.array(fidx))
        print(f"[detect {name}] {ang}: {f} frames, {len(boxes)} dets", flush=True)
    return total


def run(cmd: list[str], env_extra: dict | None = None) -> str:
    env = {**os.environ, **(env_extra or {})}
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    sys.stdout.write(r.stdout[-3000:])
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:])
        raise RuntimeError(f"stage failed: {' '.join(cmd[:3])}")
    return r.stdout


def warmup(device: str) -> None:
    """Pay one-time model downloads BEFORE any lane is timed."""
    from rtmlib import RTMPose
    RTMPose(onnx_model="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
                       "onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
            model_input_size=(192, 256), backend="onnxruntime", device=device)
    print("[warmup] rtmpose cached", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="c2a354fe")
    ap.add_argument("--tag", default="300_60")
    ap.add_argument("--lanes", required=True,
                    help="comma list of name:weights (name rfdetr* -> RF-DETR lane)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dual-numbers", default="1,3,5")
    ap.add_argument("--gt-ref", default="gt_ref",
                    help="dir (relative to repo) with ORIGINAL dets the GT indices reference")
    a = ap.parse_args()
    key = f"{a.game}_{a.tag}"
    lanes = [tuple(x.split(":", 1)) for x in a.lanes.split(",")]

    warmup(a.device)
    results = {}
    for name, weights in lanes:
        print(f"\n===== LANE {name} =====", flush=True)
        # scrub every per-window artifact so nothing leaks across lanes
        for ang in ANGLES:
            for p in (REPO / f"runs/dets_cache/{a.game}_{ang}_{a.tag}_small_1280_t0.25.dets.npz",
                      REPO / f"runs/pose_cache/{a.game}_{ang}_{a.tag}.pose.npz"):
                p.unlink(missing_ok=True)
        (REPO / f"runs/anchors/{key}.jersey_anchors.json").unlink(missing_ok=True)
        shutil.rmtree(REPO / f"runs/hybrid_{name}", ignore_errors=True)

        t, frames = {}, 0
        t0 = time.monotonic()
        frames = detect_lane(name, weights, a.game, a.tag, a.device)
        t["detect"] = time.monotonic() - t0

        t0 = time.monotonic()
        run([sys.executable, "scripts/extract_jersey_anchors.py",
             "--game", a.game, "--tag", a.tag, "--stride", "2"])
        t["ocr_anchors"] = time.monotonic() - t0

        t0 = time.monotonic()
        run([sys.executable, "scripts/extract_pose.py",
             "--game", a.game, "--tag", a.tag, "--device", a.device])
        t["pose"] = time.monotonic() - t0

        t0 = time.monotonic()
        run([sys.executable, "scripts/annotate_anchor_kits.py",
             "--game", a.game, "--tag", a.tag, "--dual-numbers", a.dual_numbers])
        t["kits"] = time.monotonic() - t0

        t0 = time.monotonic()
        run([sys.executable, "scripts/hybrid_track.py",
             "--game", a.game, "--tag", a.tag, "--out-dir", f"runs/hybrid_{name}"])
        # dual-number streams get kit-suffixed copies (n3 -> n3B/n3W); the scorer
        # disambiguates via kit-tagged anchors, the stream file is shared
        for num in a.dual_numbers.split(","):
            for ang in ANGLES:
                src = REPO / f"runs/hybrid_{name}/{key}__n{num}__{ang}.json"
                if src.exists():
                    for kk in ("B", "W"):
                        shutil.copy2(src, src.with_name(f"{key}__n{num}{kk}__{ang}.json"))
        t["track"] = time.monotonic() - t0

        t0 = time.monotonic()
        out = run([sys.executable, "scripts/solve_player_xcam.py",
                   "--game", a.game, "--tag", a.tag, "--pose",
                   "--sam3-dir", f"runs/hybrid_{name}"],
                  env_extra={"GT_DETS_DIR": a.gt_ref})
        t["xcam_score"] = time.monotonic() - t0

        rep = json.loads(out[out.index("{"):])["players"]
        strict = [v["strict_all_angles"] for v in rep.values() if v.get("strict_all_angles")]
        fused = [v["fused_coverage"] for v in rep.values()]
        total = sum(t.values())
        # full-game extrapolation: this window is 1 min of game time across 4 cams
        window_min = frames / 4 / 30 / 60
        game_h = total / window_min * GAME_MIN / 3600
        results[name] = {
            "stage_seconds": {k: round(v, 1) for k, v in t.items()},
            "total_seconds": round(total, 1),
            "effective_fps_all4cams": round(frames / total, 1),
            "full_game_hours": round(game_h, 2),
            "usd_per_game_pipeline": round(game_h * GPU_USD_H, 2),
            "strict_all_angles_mean": round(float(np.mean(strict)), 3) if strict else None,
            "fused_mean": round(float(np.mean(fused)), 3),
            "players": {pl: {"strict": v.get("strict_all_angles"),
                             "fused": v.get("fused_coverage")} for pl, v in rep.items()},
        }
        print(f"[LANE {name}] total {total:.0f}s | strict "
              f"{results[name]['strict_all_angles_mean']} | fused "
              f"{results[name]['fused_mean']}", flush=True)
        # keep the lane's artifacts for fetch/reuse
        keep = REPO / f"out/{name}"
        keep.mkdir(parents=True, exist_ok=True)
        for ang in ANGLES:
            src = REPO / f"runs/dets_cache/{a.game}_{ang}_{a.tag}_small_1280_t0.25.dets.npz"
            if src.exists():
                shutil.copy2(src, keep / src.name)
        ap_ = REPO / f"runs/anchors/{key}.jersey_anchors.json"
        if ap_.exists():
            shutil.copy2(ap_, keep / ap_.name)
        (keep / "lane_result.json").write_text(json.dumps(results[name], indent=1))

    print("\nE2E_RESULTS_JSON " + json.dumps(results))
    (REPO / "out/e2e_results.json").write_text(json.dumps(
        {"window": key, "lanes": results}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
