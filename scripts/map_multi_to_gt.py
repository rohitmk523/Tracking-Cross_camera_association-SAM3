#!/usr/bin/env python3
"""Map multi-object instance masklets (#3a/#3b per camera) to GT player names.

Instance suffixes are per-camera arbitrary (read order), so #3a in FL and #3a in NL
can be different physical players. For each (GT player, camera): the instance whose
seed box matches the GT-labelled detection at that camera's seed frame (IoU>=0.5)
IS that player there. Writes {key}__{gt_safe}__{cam}.json copies into --out-dir so
every existing scorer (fuse_score, xcam, joint) runs unchanged.

GT is used ONLY to name tracks (same-number disambiguation, the established
convention) — never to alter them.

  python scripts/map_multi_to_gt.py --game c2a354fe --tag 300_60 \
      --multi-dir runs/sam3_players_multi_c2a --out-dir runs/sam3_players_multi_c2a_named
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--multi-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iou-min", type=float, default=0.5)
    a = ap.parse_args()
    key = f"{a.game}_{a.tag}"
    ms = json.loads((REPO / f"runs/anchors/{key}.multiseeds.json").read_text())["cams"]
    gt = json.loads((REPO / f"data/gt_players/{key}.json").read_text())
    outd = REPO / a.out_dir
    outd.mkdir(parents=True, exist_ok=True)

    dets = {}
    for ang in ANGLES:
        z = np.load(REPO / f"runs/dets_cache/{a.game}_{ang}_{a.tag}_small_1280_t0.25.dets.npz")
        m = defaultdict(list)
        for b, s, c, f in zip(z["boxes"], z["scores"], z["classes"], z["frame_idx"]):
            if int(c) == 0:
                m[int(f)].append([float(v) for v in b])
        dets[ang] = m

    n_mapped = 0
    for pl, v in gt.items():
        if len(v.get("approved", {})) < 30:
            continue
        safe = pl.replace("#", "n").replace(" ", "")
        for ang in ANGLES:
            if ang not in ms:
                continue
            F = ms[ang]["seed_frame"]
            sel = v["frames"].get(str(F))
            if not sel or ang not in sel:
                continue                              # player not GT-visible at seed frame
            gb_list = dets[ang].get(F, [])
            if sel[ang] >= len(gb_list):
                continue
            gt_box = gb_list[sel[ang]]
            best, bi = 0.0, None
            for label, box in ms[ang]["players"].items():
                v_ = iou(box, gt_box)
                if v_ > best:
                    best, bi = v_, label
            if best < a.iou_min:
                continue
            inst_safe = bi.replace("#", "n")
            src = REPO / a.multi_dir / f"{key}__{inst_safe}__{ang}.json"
            if not src.exists():
                continue
            dst = outd / f"{key}__{safe}__{ang}.json"
            shutil.copy(src, dst)
            n_mapped += 1
            print(f"{pl} {ang}: instance {bi} (seed IoU {best:.2f})")
    print(f"mapped {n_mapped} (player,camera) masklets -> {outd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
