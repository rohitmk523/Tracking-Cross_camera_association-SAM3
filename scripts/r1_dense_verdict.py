#!/usr/bin/env python3
"""R1-full v2 (KPR venv) — boundary-precise foreign segments. For each coarse
flagged range, embed the 4fps crops of the flagged stream, compare per-sample
against the stream's own prototypes vs the rival's; contiguous foreign runs
(>=8 samples = 2s) become NARROW patches with flip-point boundaries.
"""
from __future__ import annotations

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

MIN_RUN = 8         # samples @4fps = 2s sustained
MARGIN = 0.85       # rival must beat self by this factor


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    a = ap.parse_args()
    IN = REPO / f"runs/r1_samples_{a.game[:3]}"
    patches = json.loads((IN / "patches.json").read_text())
    dense = json.loads((IN / "dense_index.json").read_text())
    coarse = json.loads((IN / "index.json").read_text())

    # prototype crops: coarse samples OUTSIDE any flagged range for involved streams
    flagged = defaultdict(list)
    for p in patches:
        flagged[(p["sid"], p["ang"])].append((p["gf0"], p["gf1"]))
    involved = {p["sid"] for p in patches} | {p["new"] for p in patches}
    protos = []
    for s in coarse:
        if s["sid"] not in involved:
            continue
        if any(g0 <= s["gf"] <= g1 for g0, g1 in flagged.get((s["sid"], s["ang"]), [])):
            continue
        protos.append(s)
    by_sid = defaultdict(list)
    for i, s in enumerate(protos):
        by_sid[s["sid"]].append(i)
    by_sid = {k: v[:60] for k, v in by_sid.items()}
    keep_idx = sorted(i for v in by_sid.values() for i in v)
    protos = [protos[i] for i in keep_idx]
    by_sid = defaultdict(list)
    for i, s in enumerate(protos):
        by_sid[s["sid"]].append(i)

    samples = protos + dense
    print(f"embedding {len(samples)} crops ({len(protos)} protos + {len(dense)} dense)",
          flush=True)

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
    for s0 in range(0, len(samples), CH):
        batch = []
        for s in samples[s0:s0 + CH]:
            img = cv2.imread(str(IN / s["path"]))
            batch.append({"image": img,
                          "keypoints_xyc": np.zeros((17, 3), np.float32),
                          "negative_kps": np.zeros((0, 17, 3), np.float32)})
        with torch.no_grad():
            _, e, v, _ = ext(batch)
        embs.append(e.detach().cpu())
        vises.append(v.detach().cpu())
        if (s0 // CH) % 30 == 0:
            print(f"  {min(s0+CH, len(samples))}/{len(samples)}", flush=True)
    E = torch.cat(embs)
    V = torch.cat(vises)
    from torchreid.metrics.distance import compute_distance_matrix_using_bp_features

    def dmat(qi, gi):
        return compute_distance_matrix_using_bp_features(
            E[qi], E[gi], V[qi], V[gi], use_gpu=False)[0].numpy()

    n_proto = len(protos)
    dense_by_pi = defaultdict(list)
    for j, s in enumerate(dense):
        dense_by_pi[s["pi"]].append(n_proto + j)

    out = []
    for pi, p in enumerate(patches):
        qs = [j for j in dense_by_pi.get(pi, [])
              if samples[j]["sid"] == p["sid"]]
        if len(qs) < MIN_RUN or not by_sid.get(p["sid"]) or not by_sid.get(p["new"]):
            continue
        qs.sort(key=lambda j: samples[j]["gf"])
        Dself = dmat(qs, by_sid[p["sid"]]).min(axis=1)
        Doth = dmat(qs, by_sid[p["new"]]).min(axis=1)
        foreign = Doth < MARGIN * Dself
        # contiguous runs
        j = 0
        while j < len(qs):
            if not foreign[j]:
                j += 1
                continue
            j2 = j
            while j2 + 1 < len(qs) and foreign[j2 + 1]:
                j2 += 1
            if j2 - j + 1 >= MIN_RUN:
                out.append({"sid": p["sid"], "ang": p["ang"], "new": p["new"],
                            "gf0": samples[qs[j]]["gf"], "gf1": samples[qs[j2]]["gf"],
                            "n": j2 - j + 1})
            j = j2 + 1
    (IN / "patches_dense.json").write_text(json.dumps(out, indent=1))
    print(f"narrow patches: {len(out)} -> {IN}/patches_dense.json")
    for p in out[:10]:
        print(" ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
