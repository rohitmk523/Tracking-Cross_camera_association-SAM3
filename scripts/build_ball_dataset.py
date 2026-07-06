#!/usr/bin/env python3
"""Assemble the trained-ball-detector dataset from SAM3 pseudo-labels (the teacher).

For every SAM3 ball detection (runs/sam3_ball/<clip>.sam3.json) with usable confidence,
extract the FRAME TRIPLET (f-1, f, f+1) from the local clip — motion context is the whole
point (TrackNet/WASB) — downscale, store side-by-side as one JPEG + the centre-frame ball
position. Split is LEAVE-GAMES-OUT (md5), same discipline as the jersey sets.

  python scripts/build_ball_dataset.py            # -> data/ball_dataset/{images,labels.json}
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
W, H = 512, 288                    # training resolution (per frame)
STATIC_RADIUS_PX = 30              # cluster radius for the logo filter
STATIC_OCC = 0.35                  # a cluster present in >35% of frames...
STATIC_MOVE_PX = 4.0               # ...that basically never moves = painted logo, drop it


def _drop_static(frames_dict: dict, n_frames: int) -> dict:
    """Remove PAINTED-BALL false positives (court/wall logos): clusters of detections that
    occupy a large share of frames with ~zero frame-to-frame movement. Rim-area clusters
    survive — the ball visits repeatedly but in moving bursts, not as a fixed point."""
    import numpy as np
    rows = [(int(f), r) for f, rr in frames_dict.items() for r in rr]
    if not rows:
        return frames_dict
    ctr = np.array([[(r["box"][0] + r["box"][2]) / 2, (r["box"][1] + r["box"][3]) / 2]
                    for _, r in rows])
    fr = np.array([f for f, _ in rows])
    drop = np.zeros(len(rows), bool)
    unassigned = np.ones(len(rows), bool)
    while unassigned.any():
        i = int(np.argmax(unassigned))
        d2 = ((ctr - ctr[i]) ** 2).sum(1)
        m = unassigned & (d2 < STATIC_RADIUS_PX ** 2)
        unassigned &= ~m
        idx = np.where(m)[0]
        occ = len(set(fr[idx])) / max(1, n_frames)
        if occ > STATIC_OCC:
            order = idx[np.argsort(fr[idx])]
            steps = np.linalg.norm(np.diff(ctr[order], axis=0), axis=1)
            if len(steps) and float(np.median(steps)) < STATIC_MOVE_PX:
                drop[idx] = True
    out: dict[str, list] = {}
    for k, (f, r) in enumerate(rows):
        if not drop[k]:
            out.setdefault(str(f), []).append(r)
    n_drop = int(drop.sum())
    if n_drop:
        print(f"    static-logo filter dropped {n_drop}/{len(rows)} boxes")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3-dir", default="runs/sam3_ball")
    ap.add_argument("--clip-dir", default="data/clips")
    ap.add_argument("--out", default="data/ball_dataset")
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--val-games", default=None,
                    help="comma list: force these games to val (overrides md5 split)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import cv2

    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    labels: dict[str, dict] = {}
    n_games: set[str] = set()
    for sp in sorted(Path(a.sam3_dir).glob("*.sam3.json")):
        d = json.loads(sp.read_text())
        clip = Path(a.clip_dir) / f"{sp.stem.replace('.sam3', '')}.mp4"
        if not clip.exists():
            print(f"skip {sp.stem}: clip missing")
            continue
        game = sp.stem.split("_")[0]
        n_games.add(game)
        frames_clean = _drop_static(d["frames"], d.get("n_frames", 400))
        wanted: dict[int, tuple] = {}
        for f_str, rows in frames_clean.items():
            good = [r for r in rows if r["score"] >= a.min_score]
            if len(good) == 1:                      # unambiguous single ball only
                b = good[0]["box"]
                wanted[int(f_str)] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        cap = cv2.VideoCapture(str(clip))
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(img, (W, H)))
        cap.release()
        sx, sy = W / (d.get("orig_w") or 1920), H / (d.get("orig_h") or 1080)
        kept = 0
        for f, (bx, by) in wanted.items():
            if not (1 <= f < len(frames) - 1):
                continue
            trip = cv2.hconcat([frames[f - 1], frames[f], frames[f + 1]])
            name = f"{sp.stem.replace('.sam3', '')}_f{f:04d}.jpg"
            cv2.imwrite(str(out / "images" / name), trip,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            if a.val_games:
                split = "val" if game in a.val_games.split(",") else "train"
            else:
                split_hash = hashlib.md5(f"{a.seed}:{game}".encode()).hexdigest()
                split = "val" if int(split_hash[:8], 16) % 1000 < a.val_frac * 1000 else "train"
            labels[name] = {"x": round(bx * sx, 2), "y": round(by * sy, 2),
                            "game": game, "split": split}
            kept += 1
        print(f"{sp.stem}: {kept} triplets (of {len(wanted)} single-ball frames)")
    (out / "labels.json").write_text(json.dumps(labels))
    tr = sum(1 for v in labels.values() if v["split"] == "train")
    print(f"dataset: {len(labels)} triplets across {len(n_games)} games "
          f"({tr} train / {len(labels) - tr} val, leave-games-out) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
