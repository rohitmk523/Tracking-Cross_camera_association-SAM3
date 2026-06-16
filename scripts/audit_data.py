#!/usr/bin/env python3
"""Audit OUR Training_frameworks annotations -> provenance manifest + report.

  python scripts/audit_data.py                       # print report + write manifest
  python scripts/audit_data.py --tf-root /path/to/Training_frameworks
  python scripts/audit_data.py --no-write             # report only

Writes configs/data_sources.yaml (machine-readable). See docs/04 + docs/10.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from uball_cc.data.audit import audit_sources, to_manifest
from uball_cc.data.provenance import load_games

REPO = Path(__file__).resolve().parents[1]


def _print_report(results) -> None:
    bar = "=" * 78
    print(bar)
    print("DATA AUDIT  (OUR FOOTAGE ONLY)  --  Training_frameworks detection annotations")
    print(bar)
    det_games_imgs: dict[str, int] = {}
    det_games_split: dict[str, str] = {}
    det_instances: dict[str, int] = {}
    for s in results:
        if not s.exists:
            print(f"\n[{s.name}]  MISSING ({s.path})")
            continue
        classes = ", ".join(s.class_names[k] for k in sorted(s.class_names)) or "(none)"
        tag = {"detection": "<< DETECTION (player/ref/ball)",
               "rim": "(rim/shot-detection -- excluded from detection train)",
               "other": "(other)"}.get(s.role, "")
        print(f"\n[{s.name}]  role={s.role} {tag}")
        print(f"  path:    {s.path}")
        print(f"  classes: {classes}")
        for split, st in s.splits.items():
            inst = "  ".join(f"{k}={v}" for k, v in sorted(st.class_instances.items()))
            sym = "  [symlink->valid; not counted in summary]" if st.is_symlink else ""
            print(f"  {split:6s}: {st.images:5d} imgs "
                  f"({st.images_with_labels} labeled, {st.images_empty} empty)"
                  f"  | {inst or 'no instances'}{sym}")
            games = "  ".join(f"{g}:{n}" for g, n in st.game_images.most_common())
            if games:
                print(f"          games: {games}")
            if st.angle_images:
                ang = " ".join(f"{a}:{n}" for a, n in sorted(st.angle_images.items()))
                print(f"          angles: {ang}")
            if st.unresolved_keys:
                print(f"          UNRESOLVED game keys: {dict(st.unresolved_keys.most_common())}")
            # Summarize detection taxonomy, skipping symlinked (test==valid) splits.
            if s.role == "detection" and not st.is_symlink:
                for g, n in st.game_images.items():
                    det_games_imgs[g] = det_games_imgs.get(g, 0) + n
                for k, v in st.class_instances.items():
                    det_instances[k] = det_instances.get(k, 0) + v
        if s.role == "detection":
            det_games_split.update(s.games)

    print("\n" + bar)
    print("DETECTION TAXONOMY SUMMARY (what feeds RF-DETR training)")
    print(bar)
    print("  Per-game images (player/ref/ball annotations), with docs/14 split:")
    for g, n in sorted(det_games_imgs.items(), key=lambda x: -x[1]):
        print(f"    {g}  split={det_games_split.get(g,'?'):6s}  images~{n}")
    print("  Total instances: " +
          "  ".join(f"{k}={v}" for k, v in sorted(det_instances.items())))
    games_present = set(det_games_imgs)
    print(f"\n  >> Player/ref/ball annotations exist for {len(games_present)} game(s): "
          f"{sorted(games_present)}")
    print("  >> NOTE: docs/14's split table was built for the MULTI-GAME shot-detection")
    print("     manifest (rim taxonomy). The player/ref/ball data is far narrower, so a")
    print("     true held-out *court* is not yet available -- the consolidation builds a")
    print("     cross-GAME split (config-driven; default holds c2a354fe per docs/14).")
    print(bar)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf-root", default=str(REPO.parent / "Training_frameworks"))
    ap.add_argument("--games", default=str(REPO / "configs" / "games.json"))
    ap.add_argument("--out", default=str(REPO / "configs" / "data_sources.yaml"))
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()

    games = load_games(Path(a.games))
    results = audit_sources(Path(a.tf_root), games)
    _print_report(results)

    if not a.no_write:
        manifest = to_manifest(results)
        Path(a.out).write_text(yaml.safe_dump(manifest, sort_keys=False, width=100))
        print(f"\nmanifest -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
