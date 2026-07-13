#!/usr/bin/env python3
"""E3 stage B (KPR venv) — embed stage-A crops, match queries to prototypes.

Gallery = per-identity prototype embeddings; query = the under-ball crop of a
track-absent shot. Match via KPR's part-based bp-feature distance. Scores the
18 track-absent shots (labels = GT jersey num).

  /tmp/kpr/.venv310/bin/python scripts/e3_kpr_stageB.py
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
import sys as _s
IN = REPO / (_s.argv[_s.argv.index("--in-dir")+1] if "--in-dir" in _s.argv else "runs/e3_kpr")


def main() -> int:
    import torch
    from kpr_pilot import restricted_torch_load
    samples = json.loads((IN / "samples.json").read_text())
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
        cfg.test.batch_size = 8
        ext = KPRFeatureExtractor(cfg)
    finally:
        os.chdir(cwd)
        torch.load = _orig

    embs, vises = [], []
    CH = 32
    for s0 in range(0, len(samples), CH):
        chunk = samples[s0:s0 + CH]
        batch = []
        for s in chunk:
            img = cv2.imread(str(IN / s["path"]))
            ch, cw = img.shape[:2]
            k = np.array(s["kxyc"], dtype=np.float32)
            inside = ((k[:, 0] >= 0) & (k[:, 0] <= cw - 1)
                      & (k[:, 1] >= 0) & (k[:, 1] <= ch - 1))
            k[:, 0] = np.clip(k[:, 0], 0, cw - 1)
            k[:, 1] = np.clip(k[:, 1], 0, ch - 1)
            k[:, 2] = k[:, 2] * inside          # zero conf outside the crop
            batch.append({"image": img, "keypoints_xyc": k,
                          "negative_kps": np.zeros((0, 17, 3), dtype=np.float32)})
        with torch.no_grad():
            _, e, v, _ = ext(batch)
        embs.append(e.detach().cpu() if hasattr(e, "cpu") else torch.as_tensor(np.asarray(e)))
        vises.append(v.detach().cpu() if hasattr(v, "cpu") else torch.as_tensor(np.asarray(v)))
        print(f"embedded {min(s0+CH, len(samples))}/{len(samples)}", flush=True)
    E = torch.cat(embs)
    V = torch.cat(vises)

    from torchreid.metrics.distance import compute_distance_matrix_using_bp_features
    # R1-lite: self-consistency of candidate crops vs their OWN stream protos
    ci = [i for i, s in enumerate(samples) if s["kind"] == "cand"]
    gi0 = [i for i, s in enumerate(samples) if s["kind"] == "proto"]
    if ci:
        D = compute_distance_matrix_using_bp_features(
            E[ci], E[gi0], V[ci], V[gi0], use_gpu=False)[0].numpy()
        gal = [samples[i]["label"] for i in gi0]
        out = {}
        for row, i in enumerate(ci):
            stream, shot_t = samples[i]["label"].split("|")
            ds = [float(d) for d, gl in zip(D[row], gal) if gl == stream]
            oth = [float(d) for d, gl in zip(D[row], gal) if gl != stream]
            out.setdefault(shot_t, {})[stream] = {
                "self": min(ds) if ds else None,
                "other": min(oth) if oth else None}
        import json as _j
        (IN / "cand_scores.json").write_text(_j.dumps(out))
        print(f"cand self-consistency -> {IN}/cand_scores.json ({len(ci)} crops)")
    qi = [i for i, s in enumerate(samples) if s["kind"] == "query"]
    gi = [i for i, s in enumerate(samples) if s["kind"] == "proto"]
    D = compute_distance_matrix_using_bp_features(
        E[qi], E[gi], V[qi], V[gi], use_gpu=False)[0].numpy()

    gal_ids = [samples[i]["label"] for i in gi]     # '#7' style
    ok = tot = 0
    for row, i in enumerate(qi):
        gt_num = samples[i]["label"]                # jersey num string
        d = D[row]
        # per-identity min distance over its prototypes
        per_id = defaultdict(lambda: 1e9)
        for dist, gid in zip(d, gal_ids):
            per_id[gid] = min(per_id[gid], float(dist))
        pick = min(per_id.items(), key=lambda kv: kv[1])[0]
        pick_num = "".join(c for c in pick if c.isdigit())
        tot += 1
        hit = pick_num == gt_num
        ok += hit
        print(f"  query gt=#{gt_num:>2} -> pick {pick} d={per_id[pick]:.3f} "
              f"{'OK' if hit else 'MISS'}")
    print(f"\nE3 KPR track-absent recovery: {ok}/{tot} "
          f"(chance ~1/13 = 8%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
