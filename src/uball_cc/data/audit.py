"""Audit OUR existing YOLO annotation folders in Training_frameworks/.

Produces a provenance manifest: per source dataset, the taxonomy, per-class
instance counts, per-game/per-angle image counts, and a role classification
(`detection` = player/referee/ball, the target taxonomy for docs/04; `rim` =
Basketball/Basketball Hoop, a *different* shot-detection task). This is the
authoritative "report image/instance counts per class and split" deliverable
for Week-1 D1-2 and the input to the consolidation step (build_dataset).
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .provenance import GameInfo, is_image, parse_stem, resolve_game

# Source datasets relative to the Training_frameworks root. Declared role is a
# hint; the real taxonomy is read from each data.yaml.
DEFAULT_SOURCES: list[dict] = [
    {"name": "4cam_detection", "path": "Uball 4Cam Detection/data"},
    {"name": "e6_demo",        "path": "Uball E6 Demo/data"},
    {"name": "e6_perfect",     "path": "Uball E6 Demo/data_perfect"},
    {"name": "far_angle_rim",  "path": "Uball Far Angle/data_rfdetr"},
    {"name": "near_angle_rim", "path": "Uball Near Angle/data/yolo_split"},
]

# The target detection taxonomy (docs/04). Class-name normalization handles
# the various spellings/casings used across folders.
TARGET_CLASSES = ("player", "referee", "ball")
RIM_CLASSES = ("basketball", "basketball hoop")

_SPLIT_DIRNAMES = ("train", "valid", "val", "test")


def _norm(name: str) -> str:
    return str(name).strip().lower()


def _classify_role(class_names: list[str]) -> str:
    norm = {_norm(c) for c in class_names}
    if norm and norm <= set(TARGET_CLASSES):
        return "detection"
    if norm and norm <= set(RIM_CLASSES):
        return "rim"
    return "other"


def _read_class_names(root: Path) -> dict[int, str]:
    """Read class id->name from a YOLO data.yaml; tolerate missing/empty files."""
    for cand in ("data.yaml", "dataset.yaml", "dataset_verified.yaml"):
        f = root / cand
        if f.exists() and f.stat().st_size > 0:
            doc = yaml.safe_load(f.read_text()) or {}
            names = doc.get("names")
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
            if isinstance(names, list):
                return {i: str(v) for i, v in enumerate(names)}
    return {}


def _iter_split_dirs(root: Path) -> list[tuple[str, Path, Path, bool]]:
    """Return [(split, images_dir, labels_dir, is_symlink)] for a source.

    Handles both `<root>/<split>/images` layouts and a flat `<root>/images`.
    `is_symlink` flags the common `test/images -> valid/images` symlink (4Cam,
    E6) so the summary does not double-count the held-out game.
    """
    out: list[tuple[str, Path, Path, bool]] = []
    for split in _SPLIT_DIRNAMES:
        img, lab = root / split / "images", root / split / "labels"
        if img.is_dir():
            out.append((("valid" if split == "val" else split), img, lab,
                        img.is_symlink()))
    if not out and (root / "images").is_dir():
        out.append(("all", root / "images", root / "labels", False))
    return out


@dataclass
class SplitStat:
    images: int = 0
    images_with_labels: int = 0
    images_empty: int = 0          # no label file or empty label file
    is_symlink: bool = False       # e.g. test/images -> valid/images (don't double-count)
    class_instances: Counter = field(default_factory=Counter)   # name -> count
    game_images: Counter = field(default_factory=Counter)        # gid8/key -> imgs
    angle_images: Counter = field(default_factory=Counter)       # angle -> imgs
    unresolved_keys: Counter = field(default_factory=Counter)


@dataclass
class SourceStat:
    name: str
    path: str
    exists: bool
    role: str
    class_names: dict[int, str]
    splits: dict[str, SplitStat] = field(default_factory=dict)
    # gid8 -> docs/14 split, for games seen in this source
    games: dict[str, str] = field(default_factory=dict)


def _scan_split(images_dir: Path, labels_dir: Path,
                id2name: dict[int, str],
                games: dict[str, GameInfo]) -> tuple[SplitStat, dict[str, str]]:
    st = SplitStat()
    games_seen: dict[str, str] = {}
    with os.scandir(images_dir) as it:
        entries = [e for e in it if is_image(e.name)]
    for e in entries:
        st.images += 1
        stem = Path(e.name).stem
        key, angle = parse_stem(stem)
        gi = resolve_game(key, games)
        if gi:
            st.game_images[gi.gid8] += 1
            games_seen[gi.gid8] = gi.split
        else:
            st.unresolved_keys[key] += 1
        if angle:
            st.angle_images[angle] += 1
        lf = labels_dir / f"{stem}.txt"
        if lf.exists():
            txt = lf.read_text().strip()
            if txt:
                st.images_with_labels += 1
                for line in txt.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    cid = int(float(line.split()[0]))
                    st.class_instances[id2name.get(cid, f"cls{cid}")] += 1
            else:
                st.images_empty += 1
        else:
            st.images_empty += 1
    return st, games_seen


def audit_sources(tf_root: Path, games: dict[str, GameInfo],
                  sources: list[dict] | None = None) -> list[SourceStat]:
    sources = sources or DEFAULT_SOURCES
    results: list[SourceStat] = []
    for spec in sources:
        root = tf_root / spec["path"]
        if not root.is_dir():
            results.append(SourceStat(spec["name"], spec["path"], False,
                                      "missing", {}))
            continue
        id2name = _read_class_names(root)
        role = _classify_role(list(id2name.values()))
        ss = SourceStat(spec["name"], spec["path"], True, role, id2name)
        for split, img_dir, lab_dir, is_symlink in _iter_split_dirs(root):
            stat, seen = _scan_split(img_dir, lab_dir, id2name, games)
            stat.is_symlink = is_symlink
            ss.splits[split] = stat
            ss.games.update(seen)
        results.append(ss)
    return results


def to_manifest(results: list[SourceStat]) -> dict:
    """Serializable manifest for configs/data_sources.yaml."""
    out: dict = {
        "_note": ("Auto-generated by scripts/audit_data.py. OUR-FOOTAGE-ONLY: "
                  "role=detection sources feed RF-DETR training; role=rim is a "
                  "separate ball/hoop shot-detection task (NOT player/ref) and "
                  "is excluded from the detection consolidation."),
        "target_taxonomy": list(TARGET_CLASSES),
        "sources": [],
    }
    for s in results:
        entry: dict = {
            "name": s.name, "path": s.path, "exists": s.exists,
            "role": s.role, "classes": [s.class_names[k] for k in sorted(s.class_names)],
            "games": s.games, "splits": {},
        }
        for split, st in s.splits.items():
            entry["splits"][split] = {
                "images": st.images,
                "images_with_labels": st.images_with_labels,
                "images_empty": st.images_empty,
                "is_symlink": st.is_symlink,
                "instances": dict(sorted(st.class_instances.items())),
                "game_images": dict(st.game_images.most_common()),
                "angle_images": dict(sorted(st.angle_images.items())),
                "unresolved_game_keys": dict(st.unresolved_keys.most_common()),
            }
        out["sources"].append(entry)
    return out
