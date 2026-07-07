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


def _split(stem: str, game: str, a) -> str:
    if a.val_clips:
        return "val" if any(s in stem for s in a.val_clips.split(",")) else "train"
    if a.val_games:
        return "val" if game in a.val_games.split(",") else "train"
    h = hashlib.md5(f"{a.seed}:{game}".encode()).hexdigest()
    return "val" if int(h[:8], 16) % 1000 < a.val_frac * 1000 else "train"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3-dir", default="runs/sam3_ball")
    ap.add_argument("--clip-dir", default="data/clips")
    ap.add_argument("--out", default="data/ball_dataset")
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--val-games", default=None,
                    help="comma list: force these games to val (overrides md5 split)")
    ap.add_argument("--val-clips", default=None,
                    help="comma list of SUBSTRINGS: clips matching go to val (highest precedence) — hold out exam WINDOWS while training on the same game's other footage")
    ap.add_argument("--neg-ratio", type=float, default=0.5,
                    help="negatives per positive (ball-absent triplets)")
    ap.add_argument("--gold-dir", default="data/gt_ball",
                    help="operator ball clicks {frame:[x,y]|none|unclear} (original px); clicks override the teacher, none-frames become weighted TRUE negatives")
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
        gold_p = Path(a.gold_dir) / f"{sp.stem.replace('.sam3', '')}.json"
        gold = json.loads(gold_p.read_text()) if gold_p.exists() else {}
        wanted: dict[int, tuple] = {}
        multis: dict[int, list] = {}
        for f_str, rows in frames_clean.items():
            good = [r for r in rows if r["score"] >= a.min_score]
            if len(good) == 1:
                b = good[0]["box"]
                wanted[int(f_str)] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            elif len(good) > 1:
                multis[int(f_str)] = [((r["box"][0] + r["box"][2]) / 2,
                                       (r["box"][1] + r["box"][3]) / 2) for r in good]
        # temporal selection: resolve multi-candidate frames from neighbours (gold eval
        # showed the teacher's candidates CONTAIN the ball 98-100% of the time; the old
        # exactly-one-box rule threw those frames away)
        n_resolved = 0
        for _ in range(6):
            changed = False
            for f, cands in list(multis.items()):
                anchor = wanted.get(f - 1) or wanted.get(f + 1)
                if anchor is None:
                    continue
                d, best = min((((cx - anchor[0]) ** 2 + (cy - anchor[1]) ** 2) ** 0.5, (cx, cy))
                              for cx, cy in cands)
                if d <= 80:
                    wanted[f] = best
                    del multis[f]
                    n_resolved += 1
                    changed = True
            if not changed:
                break
        if n_resolved:
            print(f"    temporal selection recovered {n_resolved} multi-candidate frames")
        cap = cv2.VideoCapture(str(clip))
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(img, (W, H)))
        cap.release()
        sx, sy = W / (d.get("orig_w") or 1920), H / (d.get("orig_h") or 1080)
        # NEGATIVES: frames where the (logo-filtered) teacher saw NO ball at all —
        # without them the model learns "there is always a ball" and cannot abstain
        # (measured: saturated 0.99 confidence on every frame, exam failure).
        empty = [f for f in range(1, len(frames) - 1)
                 if not frames_clean.get(str(f)) and f not in wanted]
        n_neg = int(len(wanted) * a.neg_ratio)
        neg = empty[::max(1, len(empty) // max(1, n_neg))][:n_neg]
        kept = 0
        for dup in range(3):                      # gold TRUE negatives, weighted x3
            for f in gold_negs:
                if not (1 <= f < len(frames) - 1):
                    continue
                trip = cv2.hconcat([frames[f - 1], frames[f], frames[f + 1]])
                name = f"{sp.stem.replace('.sam3', '')}_f{f:04d}_gneg{dup}.jpg"
                cv2.imwrite(str(out / "images" / name), trip, [cv2.IMWRITE_JPEG_QUALITY, 88])
                labels[name] = {"neg": True, "game": game, "split": _split(sp.stem, game, a)}
        for f in neg:
            if f in gold:                            # teacher-absent but operator saw it?
                continue                             # gold owns those frames now
            trip = cv2.hconcat([frames[f - 1], frames[f], frames[f + 1]])
            name = f"{sp.stem.replace('.sam3', '')}_f{f:04d}_neg.jpg"
            cv2.imwrite(str(out / "images" / name), trip, [cv2.IMWRITE_JPEG_QUALITY, 88])
            labels[name] = {"neg": True, "game": game, "split": _split(sp.stem, game, a)}
        # gold overrides: clicks replace teacher labels; unclear frames are dropped
        gold_negs = []
        for f_str, v in gold.items():
            f = int(f_str)
            if isinstance(v, list):
                wanted[f] = (v[0], v[1])
            elif v == "none":
                wanted.pop(f, None)
                gold_negs.append(f)
            else:
                wanted.pop(f, None)                  # unclear: no supervision
        for f, (bx, by) in wanted.items():
            if not (1 <= f < len(frames) - 1):
                continue
            trip = cv2.hconcat([frames[f - 1], frames[f], frames[f + 1]])
            name = f"{sp.stem.replace('.sam3', '')}_f{f:04d}.jpg"
            cv2.imwrite(str(out / "images" / name), trip,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            labels[name] = {"x": round(bx * sx, 2), "y": round(by * sy, 2),
                            "game": game, "split": _split(sp.stem, game, a)}
            kept += 1
        print(f"{sp.stem}: {kept} pos + {len(neg)} neg triplets "
              f"(of {len(wanted)} single-ball frames)")
    (out / "labels.json").write_text(json.dumps(labels))
    tr = sum(1 for v in labels.values() if v["split"] == "train")
    print(f"dataset: {len(labels)} triplets across {len(n_games)} games "
          f"({tr} train / {len(labels) - tr} val, leave-games-out) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
