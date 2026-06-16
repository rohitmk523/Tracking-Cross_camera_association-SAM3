#!/usr/bin/env python3
"""Prepare the far + near ball/hoop datasets for RF-DETR training (NO YOLO model).

The annotated far/near ball+hoop data already exists in Training_frameworks in
YOLO *format* (which RF-DETR reads directly via dataset_file="yolo"). This wraps
each into a clean in-repo dataset dir with train/valid/test + data.yaml (renaming
near's `val`->`valid`), plus a tiny 10-image smoke subset to validate the RF-DETR
training path before the full overnight AWS run.

  python scripts/prep_rim_datasets.py

Outputs (gitignored data/): data/rim_far, data/rim_near, data/rim_smoke.
"""
from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TF = REPO.parent / "Training_frameworks"
CLASSES = ["Basketball", "Basketball Hoop"]   # ball + rim (far/near shot detection)

# name -> (source dataset dir, {out_split: in_split})
SOURCES = {
    "rim_far": (TF / "Uball Far Angle" / "data_rfdetr",
                {"train": "train", "valid": "valid", "test": "test"}),
    "rim_near": (TF / "Uball Near Angle" / "data" / "yolo_split",
                 {"train": "train", "valid": "val", "test": "test"}),  # val -> valid
}


def _data_yaml(out: Path) -> str:
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    return (f"# RF-DETR ball+hoop dataset (OUR footage; YOLO format, no ultralytics).\n"
            f"path: {out}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
            f"\nnc: {len(CLASSES)}\nnames:\n{names}\n")


def _count(p: Path) -> int:
    img = p / "images"
    if not img.is_dir():
        return 0
    return sum(1 for f in img.iterdir()
              if f.suffix.lower() in (".jpg", ".jpeg", ".png"))


def wrap(name: str, src: Path, mapping: dict[str, str]) -> dict:
    """Create data/<name> with train/valid/test as dir symlinks into the source
    YOLO dataset + a data.yaml. Symlinks keep it zero-copy; the AWS bundler
    resolves them to real files."""
    out = REPO / "data" / name
    if out.exists() or out.is_symlink():
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    counts = {}
    for out_split, in_split in mapping.items():
        target = (src / in_split).resolve()
        link = out / out_split
        if target.is_dir():
            link.symlink_to(target)
            counts[out_split] = _count(target)
    (out / "data.yaml").write_text(_data_yaml(out))
    return counts


def make_smoke(n: int = 10) -> dict:
    """Tiny real-file subset from rim_far for a fast local RF-DETR smoke train."""
    src = REPO / "data" / "rim_far"
    out = REPO / "data" / "rim_smoke"
    if out.exists():
        shutil.rmtree(out)
    counts = {}
    for split, k in (("train", n), ("valid", max(2, n // 3)), ("test", 2)):
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
        imgs = sorted((src / split / "images").iterdir())[:k]
        for img in imgs:
            lab = src / split / "labels" / f"{img.stem}.txt"
            shutil.copy2(img.resolve(), out / split / "images" / img.name)
            if lab.exists():
                shutil.copy2(lab.resolve(), out / split / "labels" / f"{img.stem}.txt")
        counts[split] = len(imgs)
    (out / "data.yaml").write_text(_data_yaml(out))
    return counts


def main() -> int:
    for name, (src, mapping) in SOURCES.items():
        if not src.is_dir():
            print(f"SKIP {name}: source missing {src}")
            continue
        counts = wrap(name, src, mapping)
        print(f"{name}: {counts}  -> data/{name} (symlinked, classes={CLASSES})")
    smoke = make_smoke()
    print(f"rim_smoke: {smoke}  -> data/rim_smoke (real files, for local RF-DETR smoke)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
