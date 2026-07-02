#!/usr/bin/env python3
"""Turn jersey annotations ({number, box} / 'unclear' / 'none') into THREE training sets:

  1. LOCALIZER   — YOLO detection: player crop -> tight number box (class 'number'), plus
                   BACKGROUND images from 'none' labels (no number visible -> empty label file).
  2. RECOGNIZER  — padded number sub-crops foldered by number string. NOTE (2026-07-02 audit):
                   train an open-vocabulary STR (PARSeq-style) on these, NOT a closed-set
                   folder classifier — unseen numbers must be readable, not force-classified.
  3. LEGIBILITY  — binary player-crop classifier data: legible/ (numeric+box) vs illegible/
                   ('unclear') — the gate that keeps garbage crops away from the recognizer.

Splits are GROUP-based and deterministic (md5, never builtin hash() — that is salted per
process): by GAME when >=4 labeled games (leave-games-out = the honest cross-game val),
else by play-group (<gid8>_<cam>_<event>_t<sec>), so same-play near-duplicate crops can
never straddle train/val (audit measured ~73% val contamination under per-crop splitting).

  python scripts/build_jersey_dataset.py --pool data/jersey_pool --out data/jersey_dataset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MIN_GAMES_FOR_GAME_SPLIT = 4


def _game(crop: str) -> str:
    return crop.split("_")[0]


def _playgroup(crop: str) -> str:
    return "_".join(crop.split("_")[:-2])        # strip _f<idx>_p<n>.jpg


def _is_val(group: str, seed: int, val_frac: float) -> bool:
    d = hashlib.md5(f"{seed}:{group}".encode()).hexdigest()
    return int(d[:8], 16) % 1000 < val_frac * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/jersey_pool")
    ap.add_argument("--out", default="data/jersey_dataset")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--pad", type=float, default=0.15, help="pad around the number box for the reader crop")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import cv2

    pool, out = Path(a.pool), Path(a.out)
    if out.exists():
        shutil.rmtree(out)                        # rebuild clean (no stale split remnants)
    labels = json.loads((pool / "labels.json").read_text())

    usable, unclear, none = [], [], []
    for crop, lab in labels.items():
        if not isinstance(lab, dict) or not (pool / "crops" / crop).exists():
            continue
        num, box = str(lab.get("number", "")).strip(), lab.get("box")
        if num == "unclear":
            unclear.append(crop)
        elif num == "none":
            none.append(crop)
        elif box and num.isdigit():
            usable.append((crop, num, box))
    usable.sort()

    games = sorted({_game(c) for c, _, _ in usable})
    by_game = len(games) >= MIN_GAMES_FOR_GAME_SPLIT
    group = _game if by_game else _playgroup
    mode = "game (leave-games-out)" if by_game else f"play-group (interim until {MIN_GAMES_FOR_GAME_SPLIT}+ games labeled)"

    def split(crop: str) -> str:
        return "val" if _is_val(group(crop), a.seed, a.val_frac) else "train"

    for sub in ("localizer/train/images", "localizer/train/labels",
                "localizer/val/images", "localizer/val/labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    n = Counter()
    per_num: Counter = Counter()
    for crop, num, box in usable:
        s = split(crop)
        src = pool / "crops" / crop
        # 1. localizer: image + YOLO number box (class 0)
        shutil.copy(src, out / f"localizer/{s}/images/{crop}")
        x1, y1, x2, y2 = box
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        (out / f"localizer/{s}/labels/{Path(crop).stem}.txt").write_text(
            f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        n[f"loc_{s}"] += 1
        # 2. recognizer: padded number region, foldered by number (feed an STR, not a classifier)
        img = cv2.imread(str(src))
        if img is not None:
            H, W = img.shape[:2]
            px, py = (x2 - x1) * W * a.pad, (y2 - y1) * H * a.pad
            rx1, ry1 = max(0, int(x1 * W - px)), max(0, int(y1 * H - py))
            rx2, ry2 = min(W, int(x2 * W + px)), min(H, int(y2 * H + py))
            num_crop = img[ry1:ry2, rx1:rx2]
            if num_crop.size:
                d = out / f"recognizer/{s}/{num}"
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / crop), num_crop)
                n[f"rec_{s}"] += 1
                per_num[num] += 1
        # 3. legibility positive
        d = out / f"legibility/{s}/legible"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, d / crop)
    for crop in unclear:                          # legibility negatives (NOT localizer bg —
        s = split(crop)                           # an unclear number is still a number blob)
        d = out / f"legibility/{s}/illegible"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(pool / "crops" / crop, d / crop)
        n[f"leg_ill_{s}"] += 1
    for crop in none:                             # no number visible -> localizer background
        s = split(crop)
        shutil.copy(pool / "crops" / crop, out / f"localizer/{s}/images/{crop}")
        (out / f"localizer/{s}/labels/{Path(crop).stem}.txt").write_text("")
        d = out / f"legibility/{s}/illegible"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(pool / "crops" / crop, d / crop)
        n[f"loc_bg_{s}"] += 1

    (out / "localizer/data.yaml").write_text(
        f"path: {out.resolve()}/localizer\ntrain: train/images\nval: val/images\nnc: 1\nnames: [number]\n")
    card = [f"# Jersey dataset — {len(usable)} usable / {len(unclear)} unclear / {len(none)} none "
            f"(of {len(labels)} labels)", "",
            f"- split: **{mode}**, deterministic md5(seed={a.seed}), val_frac={a.val_frac} "
            f"(groups never straddle train/val)",
            f"- labeled games: {len(games)} -> {games}",
            f"- localizer: train {n['loc_train']}+{n['loc_bg_train']}bg / val {n['loc_val']}+{n['loc_bg_val']}bg",
            f"- recognizer crops: train {n['rec_train']} / val {n['rec_val']} across {len(per_num)} numbers "
            f"(STR training data — do NOT train a closed-set classifier on these)",
            f"- legibility: legible {n['loc_train'] + n['loc_val']} vs illegible "
            f"{n['leg_ill_train'] + n['leg_ill_val'] + n['loc_bg_train'] + n['loc_bg_val']}",
            f"- per-number counts: {dict(per_num.most_common())}"]
    (out / "DATASET_CARD.md").write_text("\n".join(card) + "\n")
    print("\n".join(card))
    print(f"\n-> {out}  (localizer/ + recognizer/ + legibility/ + data.yaml)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
