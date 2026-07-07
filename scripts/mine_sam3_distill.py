#!/usr/bin/env python3
"""Distill SAM3 into the detector: mine frames where SAM3 sees on-court people our
pipeline missed, and emit them as a YOLO source pool for build_detection_dataset.

Selection: a SAM3 person box (court-filtered, score>=min-score) counts as a MISS when
no box in our per-camera track json overlaps it (IoU < iou-miss) on that frame. Frames
are ranked by miss count; up to max-per-clip frames are kept with a min frame gap.

Labels on kept frames: ALL SAM3 person boxes (cls 0 player / 1 referee — SAM3 is the
teacher on these frames, not just where it disagrees) + our RF-DETR's confident ball
detections (cls 2) so ball supervision isn't silently negative.

TRAIN-split games only (hard guard): pseudo-labels must never touch valid/test.

  python scripts/mine_sam3_distill.py --sam3-dirs runs/sam3_ref runs/sam3_distill
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ANGLES = ("FL", "FR", "NL", "NR")


def _iou(a: list, b: list) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua


def _train_games() -> set[str]:
    import yaml
    cfg = yaml.safe_load((REPO / "configs/detection_dataset.yaml").read_text())
    return {g for g, s in cfg["split_map"].items() if s == "train"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3-dirs", nargs="+", default=["runs/sam3_ref", "runs/sam3_distill"])
    ap.add_argument("--tracks-dir", default="runs/tracking")
    ap.add_argument("--clip-dir", default="data/clips")
    ap.add_argument("--calib", default="configs/calib")
    ap.add_argument("--out", default="data/distill_pool")
    ap.add_argument("--min-score", type=float, default=0.6)
    ap.add_argument("--iou-miss", type=float, default=0.3)
    ap.add_argument("--max-per-clip", type=int, default=30)
    ap.add_argument("--min-gap", type=int, default=8)
    ap.add_argument("--ball-conf", type=float, default=0.4)
    ap.add_argument("--no-ball", action="store_true", help="skip the RF-DETR ball pass")
    a = ap.parse_args()

    import cv2

    from uball_cc.fusion.court import LENGTH, WIDTH
    from uball_cc.fusion.homography import load_calib, project_pixels

    train_games = _train_games()
    out = REPO / a.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    detector = None
    if not a.no_ball:
        from uball_cc.detection.base import RFDETRDetector
        detector = RFDETRDetector(weights="runs/rfdetr-s-1280-ourdata-v1/best.pth",
                                  model="small")

    total_frames = total_boxes = 0
    for sdir in a.sam3_dirs:
        for sp in sorted((REPO / sdir).glob("*.sam3.json")):
            stem = sp.stem.replace(".sam3", "")
            game, ang = stem.split("_")[0], stem.split("_")[1]
            if game not in train_games:
                print(f"{stem}: game not in TRAIN split — skipped (pseudo-labels never "
                      f"touch valid/test)")
                continue
            clip = REPO / a.clip_dir / f"{stem}.mp4"
            tp = REPO / a.tracks_dir / f"{stem}.json"
            if not clip.exists() or not tp.exists():
                print(f"{stem}: missing {'clip' if not clip.exists() else 'tracks'} — skipped")
                continue
            calib = load_calib(f"{a.calib}/{ang}.json")
            sam3 = json.loads(sp.read_text())
            if sam3.get("mode") == "gold-stub":
                continue
            ours: dict[int, list] = {}
            for r in json.loads(tp.read_text())["tracks"]:
                ours.setdefault(r["frame"], []).append(r["box_xyxy"])

            # court-filter SAM3 boxes, count misses per frame
            people: dict[int, list] = {}
            misses: dict[int, int] = {}
            for f_str, rows in sam3["frames"].items():
                f = int(f_str)
                good = [r for r in rows if r["score"] >= a.min_score]
                if not good:
                    continue
                feet = [((r["box"][0] + r["box"][2]) / 2, r["box"][3]) for r in good]
                court = project_pixels(feet, calib)
                on_court = [r for r, (cx, cy) in zip(good, court)
                            if -100 <= cx <= LENGTH + 100 and -100 <= cy <= WIDTH + 100]
                if not on_court:
                    continue
                people[f] = on_court
                mine = ours.get(f, [])
                misses[f] = sum(1 for r in on_court
                                if all(_iou(r["box"], b) < a.iou_miss for b in mine))

            # pick miss-heavy frames, min gap apart
            picked: list[int] = []
            for f in sorted(misses, key=lambda f: -misses[f]):
                if misses[f] == 0 or len(picked) >= a.max_per_clip:
                    break
                if all(abs(f - p) >= a.min_gap for p in picked):
                    picked.append(f)
            if not picked:
                print(f"{stem}: no miss frames")
                continue

            cap = cv2.VideoCapture(str(clip))
            n_written = 0
            for f in sorted(picked):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, img = cap.read()
                if not ok:
                    continue
                ih, iw = img.shape[:2]
                lines = []
                for r in people[f]:
                    x1, y1, x2, y2 = r["box"]
                    cx, cy = (x1 + x2) / 2 / iw, (y1 + y2) / 2 / ih
                    w, h = (x2 - x1) / iw, (y2 - y1) / ih
                    lines.append(f"{r.get('cls', 0)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                if detector is not None:
                    for d in detector.predict(img):
                        if d.class_id == 2 and d.score >= a.ball_conf:
                            x1, y1, x2, y2 = d.box_xyxy
                            lines.append(f"2 {(x1 + x2) / 2 / iw:.6f} {(y1 + y2) / 2 / ih:.6f} "
                                         f"{(x2 - x1) / iw:.6f} {(y2 - y1) / ih:.6f}")
                name = f"{stem}_distill_f{f:04d}"
                cv2.imwrite(str(out / "images" / f"{name}.jpg"), img,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")
                n_written += 1
                total_boxes += len(lines)
            cap.release()
            total_frames += n_written
            print(f"{stem}: {n_written} frames mined "
                  f"(top miss count {misses[max(misses, key=misses.get)]})")

    print(f"\npool: {total_frames} frames / {total_boxes} boxes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
