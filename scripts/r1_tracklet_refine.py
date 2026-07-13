#!/usr/bin/env python3
"""R1-full stage 1+2 (KPR venv) — embed each stream's sampled-crop timeline,
find FOREIGN SEGMENTS (sustained stretches where a stream's appearance
matches a different identity better than its own), emit reassignment patches.

GTA-Link-style split/merge adapted to claimed streams. Single-crop noise is
handled by median smoothing (the R1-lite lesson: never decide on one crop).

  KPR_CFG=configs/kpr/imagenet/kpr_uball_test_cyc2.yaml \
  /tmp/kpr/.venv310/bin/python scripts/r1_tracklet_refine.py --game c2a354fe
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
KPR = Path("/tmp/kpr")
sys.path.insert(0, str(KPR))
sys.path.insert(0, str(REPO / "scripts"))

WIN = 5            # median window (crops) — no single-crop decisions
TAU_MARGIN = 0.75  # foreign if other-proto dist < TAU_MARGIN * self-proto dist
MIN_SEG = 3        # sustained windows required


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    a = ap.parse_args()
    IN = REPO / f"runs/r1_samples_{a.game[:3]}"
    index = json.loads((IN / "index.json").read_text())
    print(f"{len(index)} crops to embed", flush=True)

    import torch
    from kpr_pilot import restricted_torch_load
    _orig = torch.load
    torch.load = restricted_torch_load()
    from torchreid.scripts.builder import build_config
    from torchreid.tools.feature_extractor import KPRFeatureExtractor
    cwd = os.getcwd()
    os.chdir(KPR)
    try:
        cfg = build_config(config_path=str(
            KPR / os.environ.get("KPR_CFG", "configs/kpr/imagenet/kpr_uball_test_cyc2.yaml")))
        cfg.use_gpu = False
        cfg.test.batch_size = 16
        ext = KPRFeatureExtractor(cfg)
    finally:
        os.chdir(cwd)
        torch.load = _orig

    embs, vises = [], []
    CH = 48
    for s0 in range(0, len(index), CH):
        batch = []
        for s in index[s0:s0 + CH]:
            img = cv2.imread(str(IN / s["path"]))
            batch.append({"image": img,
                          "keypoints_xyc": np.zeros((0, 3), np.float32),
                          "negative_kps": np.zeros((0, 17, 3), np.float32)})
        with torch.no_grad():
            _, e, v, _ = ext(batch)
        embs.append(e.detach().cpu())
        vises.append(v.detach().cpu())
        if (s0 // CH) % 20 == 0:
            print(f"  embedded {min(s0+CH, len(index))}/{len(index)}", flush=True)
    E = torch.cat(embs)
    V = torch.cat(vises)

    from torchreid.metrics.distance import compute_distance_matrix_using_bp_features
    sids = sorted({s["sid"] for s in index})
    # robust prototypes: per stream, gallery = its own crops
    idx_by_sid = {sid: [i for i, s in enumerate(index) if s["sid"] == sid] for sid in sids}

    def dist_to_gallery(qi, gi):
        D = compute_distance_matrix_using_bp_features(
            E[qi], E[gi], V[qi], V[gi], use_gpu=False)[0].numpy()
        return D

    patches = []
    for sid in sids:
        own = idx_by_sid[sid]
        if len(own) < 10:
            continue
        # per (cam) time series
        by_ang = defaultdict(list)
        for i in own:
            by_ang[index[i]["ang"]].append(i)
        for ang, seq in by_ang.items():
            seq.sort(key=lambda i: index[i]["gf"])
            if len(seq) < WIN * 2:
                continue
            # self-dist: leave-self-out vs own gallery (other cams + times)
            gal = [i for i in own if i not in seq] or own
            Dself = dist_to_gallery(seq, gal).min(axis=1)
            # dist to EACH other stream gallery (min across streams, remember argmin)
            oth_best = np.full(len(seq), 9.9)
            oth_who = [None] * len(seq)
            for sid2 in sids:
                if sid2 == sid or len(idx_by_sid[sid2]) < 10:
                    continue
                D2 = dist_to_gallery(seq, idx_by_sid[sid2]).min(axis=1)
                m = D2 < oth_best
                oth_best[m] = D2[m]
                for j in np.where(m)[0]:
                    oth_who[j] = sid2
            # median smoothing, then sustained-foreign detection
            k = WIN
            sm_self = np.array([np.median(Dself[max(0, j-k//2):j+k//2+1]) for j in range(len(seq))])
            sm_oth = np.array([np.median(oth_best[max(0, j-k//2):j+k//2+1]) for j in range(len(seq))])
            foreign = sm_oth < TAU_MARGIN * sm_self
            j = 0
            while j < len(seq):
                if not foreign[j]:
                    j += 1
                    continue
                j2 = j
                while j2 + 1 < len(seq) and foreign[j2 + 1]:
                    j2 += 1
                if j2 - j + 1 >= MIN_SEG:
                    whos = [oth_who[x] for x in range(j, j2 + 1) if oth_who[x]]
                    new = max(set(whos), key=whos.count) if whos else None
                    if new:
                        patches.append({"sid": sid, "ang": ang, "new": new,
                                        "gf0": index[seq[j]]["gf"],
                                        "gf1": index[seq[j2]]["gf"],
                                        "n": j2 - j + 1})
                j = j2 + 1
    (IN / "patches.json").write_text(json.dumps(patches, indent=1))
    print(f"foreign segments: {len(patches)} -> {IN}/patches.json")
    for p in patches[:12]:
        print(" ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
