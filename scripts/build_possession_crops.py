#!/usr/bin/env python3
"""Possession-crop dataset from SAM3 teacher labels: on frames where SAM3's ball
lands inside exactly one SAM3 person box, that person is the HOLDER (has_ball crop);
other people in the same frame are no_ball crops. Pose-based possession — the signal
that separates the dribbler from the defender running beside him (Roboflow's
player-in-possession idea, as a crop classifier so the detector dataset stays clean).

Needs BOTH a SAM3 ball json (runs/sam3_ball*) and a SAM3 person json
(runs/sam3_ref | runs/sam3_distill) for a clip. TRAIN-split games only.

Output: data/possession_crops/{train,val}/{has_ball,no_ball}/*.jpg
(val = --val-games, default 2399cfac — cross-game validation)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

BALL_DIRS = ("runs/sam3_ball3", "runs/sam3_ball2", "runs/sam3_ball")
PERSON_DIRS = ("runs/sam3_ref", "runs/sam3_distill")


def _train_games() -> set[str]:
    import yaml
    cfg = yaml.safe_load((REPO / "configs/detection_dataset.yaml").read_text())
    return {g for g, s in cfg["split_map"].items() if s == "train"}


def _find(stem: str, dirs) -> Path | None:
    for d in dirs:
        p = REPO / d / f"{stem}.sam3.json"
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-dir", default="data/clips")
    ap.add_argument("--out", default="data/possession_crops")
    ap.add_argument("--val-games", default="2399cfac")
    ap.add_argument("--ball-score", type=float, default=0.5)
    ap.add_argument("--person-score", type=float, default=0.6)
    ap.add_argument("--neg-per-frame", type=int, default=3)
    ap.add_argument("--frame-step", type=int, default=3,
                    help="sample every Nth holder frame — adjacent frames are near-dups")
    a = ap.parse_args()

    import cv2

    sys.path.insert(0, str(REPO / "scripts"))
    from build_ball_dataset import _drop_static

    train_games = _train_games()
    val_games = set(a.val_games.split(","))
    out = REPO / a.out
    for sp in ("train", "val"):
        for cl in ("has_ball", "no_ball"):
            (out / sp / cl).mkdir(parents=True, exist_ok=True)

    n_pos = n_neg = 0
    for clip in sorted((REPO / a.clip_dir).glob("*.mp4")):
        stem = clip.stem
        game = stem.split("_")[0]
        if game not in train_games:
            continue
        bp, pp = _find(stem, BALL_DIRS), _find(stem, PERSON_DIRS)
        if bp is None or pp is None:
            continue
        bd = json.loads(bp.read_text())
        if bd.get("mode") == "gold-stub":
            continue
        pd = json.loads(pp.read_text())
        ball_frames = _drop_static(bd["frames"], bd.get("n_frames", 400))

        # holder frames: ball centre inside exactly one person box
        picks: list[tuple[int, list, list]] = []      # (frame, holder_box, others)
        for f_str, prows in pd["frames"].items():
            brows = [r for r in ball_frames.get(f_str, []) if r["score"] >= a.ball_score]
            if len(brows) != 1:
                continue
            bx = (brows[0]["box"][0] + brows[0]["box"][2]) / 2
            by = (brows[0]["box"][1] + brows[0]["box"][3]) / 2
            people = [r["box"] for r in prows if r["score"] >= a.person_score]
            holders = [b for b in people if b[0] <= bx <= b[2] and b[1] <= by <= b[3]]
            if len(holders) != 1:
                continue
            others = [b for b in people if b is not holders[0]]
            picks.append((int(f_str), holders[0], others))
        picks = sorted(picks)[::a.frame_step]
        if not picks:
            continue

        split = "val" if game in val_games else "train"
        cap = cv2.VideoCapture(str(clip))
        wrote = 0
        for f, holder, others in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            ih, iw = img.shape[:2]

            def crop(box):
                x1, y1, x2, y2 = (max(0, int(box[0])), max(0, int(box[1])),
                                  min(iw, int(box[2])), min(ih, int(box[3])))
                return img[y1:y2, x1:x2] if x2 - x1 > 15 and y2 - y1 > 30 else None

            c = crop(holder)
            if c is None:
                continue
            cv2.imwrite(str(out / split / "has_ball" / f"{stem}_f{f:04d}.jpg"), c,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            n_pos += 1
            wrote += 1
            for k, b in enumerate(sorted(others, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
                                  [:a.neg_per_frame]):
                c = crop(b)
                if c is not None:
                    cv2.imwrite(str(out / split / "no_ball" / f"{stem}_f{f:04d}_n{k}.jpg"), c,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                    n_neg += 1
        cap.release()
        print(f"{stem} [{split}]: {wrote} holder frames")

    print(f"\ncrops: {n_pos} has_ball / {n_neg} no_ball -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
