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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3-dir", default="runs/sam3_ball")
    ap.add_argument("--clip-dir", default="data/clips")
    ap.add_argument("--out", default="data/ball_dataset")
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.25)
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
        wanted: dict[int, tuple] = {}
        for f_str, rows in d["frames"].items():
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
            split_hash = hashlib.md5(f"{a.seed}:{game}".encode()).hexdigest()
            labels[name] = {"x": round(bx * sx, 2), "y": round(by * sy, 2),
                            "game": game,
                            "split": "val" if int(split_hash[:8], 16) % 1000 < a.val_frac * 1000
                            else "train"}
            kept += 1
        print(f"{sp.stem}: {kept} triplets (of {len(wanted)} single-ball frames)")
    (out / "labels.json").write_text(json.dumps(labels))
    tr = sum(1 for v in labels.values() if v["split"] == "train")
    print(f"dataset: {len(labels)} triplets across {len(n_games)} games "
          f"({tr} train / {len(labels) - tr} val, leave-games-out) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
