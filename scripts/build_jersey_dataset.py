#!/usr/bin/env python3
"""Turn jersey annotations ({number, box}) into TWO training sets (docs/05):

  1. LOCALIZER  — YOLO detection: player crop -> tight number box (class 'number').
                  Trains a cross-game number-localizer (the e6 one doesn't generalise).
  2. RECOGNISER — the cropped number region -> its digit string (ImageFolder-per-class).
                  Trains the number reader.

Only crops that have BOTH a box and a numeric label are used (none/unclear/no-box skipped).

  python scripts/build_jersey_dataset.py --pool data/jersey_pool --out data/jersey_dataset
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/jersey_pool")
    ap.add_argument("--out", default="data/jersey_dataset")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--pad", type=float, default=0.15, help="pad around the number box for the reader crop")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import cv2

    pool, out = Path(a.pool), Path(a.out)
    labels = json.loads((pool / "labels.json").read_text())
    # keep only usable items: numeric number + a box
    usable = []
    for crop, lab in labels.items():
        if not isinstance(lab, dict):
            continue
        num, box = str(lab.get("number", "")).strip(), lab.get("box")
        if box and num and num.isdigit() and (pool / "crops" / crop).exists():
            usable.append((crop, num, box))
    usable.sort()
    # deterministic split (hash the crop name -> no RNG import needed / stable across runs)
    val = {c for c, _, _ in usable if (hash(c) % 1000) / 1000.0 < a.val_frac}

    for sub in ("localizer/train/images", "localizer/train/labels",
                "localizer/val/images", "localizer/val/labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    n_loc, n_rec, per_num = 0, 0, {}
    for crop, num, box in usable:
        split = "val" if crop in val else "train"
        src = pool / "crops" / crop
        # 1. localizer: copy image + YOLO number box (class 0)
        shutil.copy(src, out / f"localizer/{split}/images/{crop}")
        x1, y1, x2, y2 = box
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        (out / f"localizer/{split}/labels/{Path(crop).stem}.txt").write_text(
            f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        n_loc += 1
        # 2. recogniser: crop the padded number region, foldered by number
        img = cv2.imread(str(src))
        if img is not None:
            H, W = img.shape[:2]
            px, py = (x2 - x1) * W * a.pad, (y2 - y1) * H * a.pad
            rx1, ry1 = max(0, int(x1 * W - px)), max(0, int(y1 * H - py))
            rx2, ry2 = min(W, int(x2 * W + px)), min(H, int(y2 * H + py))
            num_crop = img[ry1:ry2, rx1:rx2]
            if num_crop.size:
                d = out / f"recognizer/{split}/{num}"
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / crop), num_crop)
                n_rec += 1
                per_num[num] = per_num.get(num, 0) + 1

    (out / "localizer/data.yaml").write_text(
        f"path: {out.resolve()}/localizer\ntrain: train/images\nval: val/images\nnc: 1\nnames: [number]\n")
    card = [f"# Jersey dataset ({len(usable)} usable of {len(labels)} labels)", "",
            f"- localizer YOLO images: {n_loc}  (train/val split ~{1 - a.val_frac:.0%}/{a.val_frac:.0%})",
            f"- recogniser number crops: {n_rec}  across {len(per_num)} numbers",
            f"- per-number counts: {dict(sorted(per_num.items(), key=lambda kv: -kv[1]))}"]
    (out / "DATASET_CARD.md").write_text("\n".join(card) + "\n")
    print("\n".join(card))
    print(f"\n-> {out}  (localizer/ + recognizer/ + data.yaml)")
    print("next (when enough labels): train localizer (RF-DETR-nano, 1 class) + recogniser (ResNet classifier)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
