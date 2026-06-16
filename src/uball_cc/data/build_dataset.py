"""Consolidate OUR player/referee/ball YOLO annotations into ONE RF-DETR dataset.

Driven by configs/detection_dataset.yaml. Produces a YOLO-format dataset that
RF-DETR reads directly (`dataset_file="yolo"`): `<out>/{train,valid,test}/
{images,labels}` + `data.yaml`, PLUS a COCO ground-truth JSON per split for the
evaluation harness (pycocotools). Splits are whole-game (cross-game), never
random-frame (docs/04). Frames are deduped across sources by canonical
(gid8, angle, frame) key, lowest `priority` winning.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .provenance import ANGLES, GameInfo, parse_stem, resolve_game

VALID_SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class FrameItem:
    out_stem: str                 # normalized {gid8}_{angle}_{frame}
    src_image: Path
    lines: tuple[str, ...]        # remapped canonical YOLO lines
    split: str
    gid8: str
    angle: str | None
    instances: tuple[int, ...]    # canonical class ids in this frame


@dataclass
class BuildReport:
    out_dir: str
    target_classes: list[str]
    val_mode: str
    split_map: dict[str, str]
    # split -> {"images": n, "instances": {class: n}, "games": {gid8: n}, "angles": {a: n}}
    splits: dict[str, dict] = field(default_factory=dict)
    sources_used: list[str] = field(default_factory=list)
    duplicates_skipped: int = 0
    frames_dropped_empty: int = 0
    warnings: list[str] = field(default_factory=list)


def _frame_token(stem: str, angle: str | None) -> str:
    parts = stem.split("_")
    if angle and angle in parts:
        i = parts.index(angle)
        return "_".join(parts[i + 1:]) or "f0"
    return "_".join(parts[1:]) or "f0"


def _iter_source_frames(root: Path):
    """Yield (image_path, label_path) for non-symlinked train/valid splits."""
    for split in ("train", "valid", "val"):
        img_dir, lab_dir = root / split / "images", root / split / "labels"
        if not img_dir.is_dir() or img_dir.is_symlink():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            yield img, lab_dir / f"{img.stem}.txt"


def _remap_lines(label_path: Path, id_to_canonical: dict[int, int]) -> list[str]:
    """Rewrite YOLO lines to canonical class ids; drop unmapped ids."""
    if not label_path.exists():
        return []
    out: list[str] = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        toks = line.split()
        raw = int(float(toks[0]))
        cid = id_to_canonical.get(raw)
        if cid is None:
            continue
        out.append(" ".join([str(cid), *toks[1:]]))
    return out


def collect_frames(cfg: dict, tf_root: Path,
                   games: dict[str, GameInfo]) -> tuple[list[FrameItem], BuildReport]:
    target = list(cfg["target_classes"])
    name_to_id = {n: i for i, n in enumerate(target)}
    split_map: dict[str, str] = dict(cfg["split_map"])
    report = BuildReport(out_dir=cfg["out_dir"], target_classes=target,
                         val_mode=cfg["val_mode"], split_map=split_map)

    sources = sorted([s for s in cfg["sources"] if s.get("include")],
                     key=lambda s: s.get("priority", 99))
    report.sources_used = [s["name"] for s in sources]
    seen: dict[tuple[str, str | None, str], int] = {}   # key -> winning priority
    items: dict[tuple[str, str | None, str], FrameItem] = {}

    for src in sources:
        root = tf_root / src["path"]
        if not root.is_dir():
            report.warnings.append(f"source missing: {src['path']}")
            continue
        cmap = src.get("class_map") or {}
        id_to_canonical = {int(k): name_to_id[v] for k, v in cmap.items()
                           if v in name_to_id}
        prio = src.get("priority", 99)
        for img, lab in _iter_source_frames(root):
            game_key, angle = parse_stem(img.stem)
            gi = resolve_game(game_key, games)
            if gi is None or gi.gid8 not in split_map:
                continue
            split = split_map[gi.gid8]
            ftok = _frame_token(img.stem, angle)
            key = (gi.gid8, angle, ftok)
            if key in seen:
                report.duplicates_skipped += 1
                if prio >= seen[key]:
                    continue                       # keep the higher-priority one
            lines = _remap_lines(lab, id_to_canonical)
            if not lines and not cfg.get("keep_empty_frames"):
                report.frames_dropped_empty += 1
                seen.setdefault(key, prio)
                continue
            seen[key] = prio
            inst = tuple(int(ln.split()[0]) for ln in lines)
            ang = angle if angle in ANGLES else None
            out_stem = f"{gi.gid8}_{ang or 'NA'}_{ftok}"
            items[key] = FrameItem(out_stem, img, tuple(lines), split,
                                   gi.gid8, ang, inst)
    return list(items.values()), report


def _assign_val(items: list[FrameItem], cfg: dict) -> list[FrameItem]:
    """Apply val_mode: held_out_game (val==test game) or temporal_holdout."""
    mode = cfg["val_mode"]
    if mode == "held_out_game":
        # Mirror the test game into valid (RF-DETR wants a 'valid' split for
        # early stopping). test stays as-is; valid is a copy of the test frames.
        test_items = [it for it in items if it.split == "test"]
        valid = [FrameItem(it.out_stem, it.src_image, it.lines, "valid",
                           it.gid8, it.angle, it.instances) for it in test_items]
        return items + valid
    if mode == "temporal_holdout":
        frac = float(cfg.get("val_fraction", 0.1))
        train = [it for it in items if it.split == "train"]
        # deterministic per-angle tail holdout (no leakage of nearby frames)
        by_angle: dict[str | None, list[FrameItem]] = {}
        for it in train:
            by_angle.setdefault(it.angle, []).append(it)
        promote: set[str] = set()
        for angle, group in by_angle.items():
            group.sort(key=lambda it: it.out_stem)
            n_val = max(1, int(len(group) * frac))
            for it in group[-n_val:]:
                promote.add(it.out_stem)
        out = []
        for it in items:
            if it.split == "train" and it.out_stem in promote:
                out.append(FrameItem(it.out_stem, it.src_image, it.lines, "valid",
                                     it.gid8, it.angle, it.instances))
            else:
                out.append(it)
        return out
    raise ValueError(f"unknown val_mode: {mode}")


def _yolo_to_coco_box(line: str, w: int, h: int):
    _, cx, cy, bw, bh = (float(t) for t in line.split())
    x = (cx - bw / 2) * w
    y = (cy - bh / 2) * h
    return [x, y, bw * w, bh * h]


def write_coco(items: list[FrameItem], split: str, target: list[str],
               img_dir: Path, out_json: Path) -> None:
    """COCO GT for a split (category ids 1-indexed for pycocotools)."""
    from PIL import Image
    images, annotations = [], []
    categories = [{"id": i + 1, "name": n} for i, n in enumerate(target)]
    ann_id = 1
    for img_id, it in enumerate(sorted([i for i in items if i.split == split],
                                       key=lambda x: x.out_stem), start=1):
        p = img_dir / f"{it.out_stem}{it.src_image.suffix}"
        with Image.open(p) as im:
            w, h = im.size
        images.append({"id": img_id, "file_name": p.name, "width": w, "height": h})
        for line in it.lines:
            cid = int(line.split()[0])
            box = _yolo_to_coco_box(line, w, h)
            annotations.append({"id": ann_id, "image_id": img_id,
                                "category_id": cid + 1, "bbox": box,
                                "area": box[2] * box[3], "iscrowd": 0})
            ann_id += 1
    out_json.write_text(json.dumps(
        {"images": images, "annotations": annotations, "categories": categories}))


def build(cfg: dict, tf_root: Path, games: dict[str, GameInfo],
          repo_root: Path) -> BuildReport:
    items, report = collect_frames(cfg, tf_root, games)
    items = _assign_val(items, cfg)
    out = repo_root / cfg["out_dir"]
    if out.exists():
        shutil.rmtree(out)
    for split in VALID_SPLITS:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    mode = cfg.get("image_mode", "copy")
    for it in items:
        dst_img = out / it.split / "images" / f"{it.out_stem}{it.src_image.suffix}"
        dst_lab = out / it.split / "labels" / f"{it.out_stem}.txt"
        if not dst_img.exists():
            if mode == "symlink":
                dst_img.symlink_to(it.src_image.resolve())
            else:
                shutil.copy2(it.src_image, dst_img)
        dst_lab.write_text("\n".join(it.lines) + "\n")

    # data.yaml (RF-DETR YOLO format)
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(report.target_classes))
    (out / "data.yaml").write_text(
        f"# OUR FOOTAGE ONLY. Built by scripts/build_detection_dataset.py\n"
        f"path: {out}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        f"\nnc: {len(report.target_classes)}\nnames:\n{names}\n")

    # per-split report + COCO GT
    for split in VALID_SPLITS:
        sp = [it for it in items if it.split == split]
        inst: Counter = Counter()
        games_c: Counter = Counter()
        ang_c: Counter = Counter()
        for it in sp:
            for cid in it.instances:
                inst[report.target_classes[cid]] += 1
            games_c[it.gid8] += 1
            ang_c[it.angle or "NA"] += 1
        report.splits[split] = {
            "images": len(sp),
            "instances": dict(sorted(inst.items())),
            "games": dict(games_c.most_common()),
            "angles": dict(sorted(ang_c.items())),
        }
        write_coco(items, split, report.target_classes,
                   out / split / "images", out / split / "_annotations.coco.json")
    return report
