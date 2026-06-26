#!/usr/bin/env python3
"""Consolidate OUR player/referee/ball annotations -> one RF-DETR/YOLO dataset.

  python scripts/build_detection_dataset.py
  python scripts/build_detection_dataset.py --config configs/detection_dataset.yaml

Writes the dataset under <out_dir> (gitignored data/) with train/valid/test +
data.yaml + per-split COCO GT, plus a DATASET_CARD.md provenance/counts summary.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from uball_cc.data.build_dataset import build
from uball_cc.data.provenance import load_games

REPO = Path(__file__).resolve().parents[1]


def _card(report) -> str:
    lines = [
        "# Detection dataset card (OUR FOOTAGE ONLY)", "",
        f"- target classes: `{report.target_classes}`",
        f"- val_mode: `{report.val_mode}`",
        f"- split_map: `{report.split_map}`",
        f"- sources used: `{report.sources_used}`",
        f"- duplicate frames skipped (cross-source): {report.duplicates_skipped}",
        f"- frames dropped (empty after remap): {report.frames_dropped_empty}",
        f"- frames UNRESOLVED to a known game (not admitted): {report.frames_unresolved}"
        + (f"  e.g. {report.unresolved_samples[:6]}" if report.unresolved_samples else ""),
        f"- frames off split_map (resolved but not selected): {report.frames_off_split}",
        "",]
    lines += [
        "## Per-split counts", "",
        "| split | images | " + " | ".join(report.target_classes) + " | games |",
        "|---|---|" + "---|" * len(report.target_classes) + "---|",
    ]
    for split in ("train", "valid", "test"):
        s = report.splits.get(split, {})
        inst = s.get("instances", {})
        cells = " | ".join(str(inst.get(c, 0)) for c in report.target_classes)
        games = ", ".join(f"{g}({n})" for g, n in s.get("games", {}).items())
        lines.append(f"| {split} | {s.get('images', 0)} | {cells} | {games} |")
    if report.val_mode == "held_out_game":
        lines += ["", "> NOTE: `valid` == `test` (single held-out game). Early-stopping "
                  "selects on the test set — reported test == val. Prefer "
                  "`temporal_holdout`, or annotate a 3rd game to make val != test."]
    elif report.val_mode == "temporal_holdout":
        lines += ["", "> NOTE: `valid` is an in-domain temporal tail of the TRAIN game "
                  "(for early-stopping); `test` (the held-out game) is untouched. "
                  "True cross-court val awaits a 2nd venue / 3rd game (docs/14)."]
    elif report.val_mode == "explicit":
        lines += ["", "> NOTE: `valid` is one or more WHOLE held-out games (cross-game "
                  "validation), distinct from `test`; no temporal-tail carving. `test` "
                  "(the held-out anchor game) stays untouched."]
    if report.warnings:
        lines += ["", "## Warnings", *[f"- {w}" for w in report.warnings]]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "detection_dataset.yaml"))
    ap.add_argument("--tf-root", default=str(REPO.parent / "Training_frameworks"))
    ap.add_argument("--games", default=str(REPO / "configs" / "games.json"))
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    games = load_games(Path(a.games))
    report = build(cfg, Path(a.tf_root), games, REPO)

    out = REPO / cfg["out_dir"]
    card = _card(report)
    (out / "DATASET_CARD.md").write_text(card)          # next to the (gitignored) data
    tracked = REPO / "docs" / "datasets" / f"{Path(cfg['out_dir']).name}_card.md"
    tracked.parent.mkdir(parents=True, exist_ok=True)
    tracked.write_text(card)                            # tracked git record of counts
    print(card)
    print(f"dataset -> {out}\n  data.yaml + train/valid/test + per-split "
          f"_annotations.coco.json + DATASET_CARD.md")
    print(f"tracked card -> {tracked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
