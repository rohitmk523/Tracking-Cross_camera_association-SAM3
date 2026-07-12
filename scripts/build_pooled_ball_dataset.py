#!/usr/bin/env python3
"""Pool ALL ball labels across the Training_frameworks into ONE ball+hoop specialist
dataset — no new annotation (user directive 2026-07-12). Sources, all on our rig:

  4Cam Detection  (class 2 ball)             -> Basketball(0); no hoop
  Far Angle       (0 Basketball, 1 Hoop)     -> kept (far FL/FR)
  Near Angle      (0 Basketball, 1 Hoop)     -> kept (near NL/NR)

Only images that carry >=1 Basketball label are pooled (a ball specialist must not
learn "empty" frames as negatives-of-everything — player/ref boxes are dropped).
Whole-GAME holdout across the union to prevent frame leakage: a game's frames all
land in the same split. Output: data/ball_pooled/{train,valid,test}/{images,labels}
+ data.yaml (nc:2 Basketball,Hoop). Symlinks images (no copy).

  python scripts/build_pooled_ball_dataset.py
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FW = REPO.parent / "Training_frameworks"
OUT = REPO / "data/ball_pooled"

SOURCES = [
    ("cons", REPO / "data/detect_consolidated", {2: 0}),         # consolidated ball->Basketball
    ("far", FW / "Uball Far Angle/data", {0: 0, 1: 1}),          # keep
    ("near", FW / "Uball Near Angle/data", {0: 0, 1: 1}),        # keep
]
EXCLUDE_GAMES = {"e6fba750"}                                     # events eval game — keep OUT
GID_RE = re.compile(r"([0-9a-f]{8})")


def game_of(stem: str) -> str:
    m = GID_RE.search(stem)
    return m.group(1) if m else stem[:8]


def main() -> int:
    # collect (image_path, label_lines) per source, remapped, ball-bearing only
    items = []                                    # (game, img_path, remapped_lines)
    per_src = defaultdict(int)
    for name, base, remap in SOURCES:
        for split in ("train", "valid", "val", "test"):
            limg = base / split / "images"
            llab = base / split / "labels"
            if not limg.exists():
                continue
            for lf in llab.glob("*.txt"):
                lines = [ln.split() for ln in lf.read_text().splitlines() if ln.strip()]
                out_lines, has_ball = [], False
                for ln in lines:
                    c = int(ln[0])
                    if c not in remap:
                        continue
                    nc = remap[c]
                    if nc == 0:
                        has_ball = True
                    out_lines.append(" ".join([str(nc)] + ln[1:]))
                if not has_ball or game_of(lf.stem) in EXCLUDE_GAMES:
                    continue
                img = next((limg / (lf.stem + ext) for ext in (".jpg", ".png", ".jpeg")
                            if (limg / (lf.stem + ext)).exists()), None)
                if img is None:
                    continue
                items.append((game_of(lf.stem), img, out_lines))
                per_src[name] += 1

    # whole-game split: 80/12/8 by game, deterministic
    games = sorted({g for g, _, _ in items})
    n = len(games)
    test_g = set(games[int(n * 0.92):])
    val_g = set(games[int(n * 0.80):int(n * 0.92)])
    print(f"pooled ball-bearing images: {len(items)} from {n} games; per source {dict(per_src)}")

    counts = defaultdict(lambda: [0, 0])          # split -> [imgs, ball boxes]
    for split in ("train", "valid", "test"):
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
    for g, img, lines in items:
        split = "test" if g in test_g else "valid" if g in val_g else "train"
        # unique name (source games can share stems across angles rarely)
        dst_stem = f"{img.parent.parent.parent.name}_{img.stem}"[:120]
        di = OUT / split / "images" / (dst_stem + img.suffix)
        dl = OUT / split / "labels" / (dst_stem + ".txt")
        if not di.exists():
            di.symlink_to(img.resolve())
        dl.write_text("\n".join(lines) + "\n")
        counts[split][0] += 1
        counts[split][1] += sum(1 for ln in lines if ln.startswith("0 "))
    for split in ("train", "valid", "test"):
        print(f"  {split}: {counts[split][0]} imgs, {counts[split][1]} Basketball boxes")

    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: 2\nnames:\n  0: Basketball\n  1: Basketball Hoop\n")
    print(f"-> {OUT}/data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
