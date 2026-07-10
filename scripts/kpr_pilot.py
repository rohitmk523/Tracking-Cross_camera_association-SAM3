#!/usr/bin/env python3
"""KPR pilot — can keypoint-promptable re-ID tell SAME-KIT teammates apart in pile-ups?

The 78->88 gap lives in crowded moments where kits are identical, numbers unreadable
and boxes overlap. KPR (ECCV24) embeds ONLY the prompted person (our RTMPose keypoints)
and compares only mutually-visible body parts — the first credible within-team
appearance signal.

Protocol (leakage-free): gallery = isolated GT crops per player (identity from GT,
appearance clean); queries = pile-up GT crops (target box overlaps another detection).
Score: nearest-gallery-identity top-1 accuracy, overall + the same-kit trio (#22/#6/#43).

Run with the KPR env:  /tmp/kpr/.venv310/bin/python scripts/kpr_pilot.py
Requires: /tmp/kpr checkout + pretrained weights (user-approved 2026-07-10),
          runs/pose_cache/*.pose.npz, runs/dets_cache, data/gt_players.
"""
from __future__ import annotations

import functools
import io
import json
import pickle
import sys
import types
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
KPR = Path("/tmp/kpr")
sys.path.insert(0, str(KPR))
ANGLES = ("FL", "FR", "NL", "NR")
OFFS = {"FL": 0, "FR": -11, "NL": -1, "NR": -1}
GAME, TAG = "e6fba750", "44_60"
PLAYERS = ("#11", "#22", "#43", "#6")
SAME_KIT = ("#22", "#43", "#6")
MIN_H = 90
GALLERY_PER_PLAYER = 10
MAX_QUERIES_PER_PLAYER = 25


def restricted_torch_load():
    import torch
    AUDITED = {
        ("__builtin__", "set"): set, ("builtins", "set"): set,
        ("_codecs", "encode"): __import__("_codecs").encode,
        ("collections", "OrderedDict"): __import__("collections").OrderedDict,
        ("numpy", "dtype"): __import__("numpy").dtype,
        ("numpy.core.multiarray", "scalar"):
            __import__("numpy.core.multiarray", fromlist=["scalar"]).scalar,
        ("numpy._core.multiarray", "scalar"):
            __import__("numpy.core.multiarray", fromlist=["scalar"]).scalar,
        ("torch", "FloatStorage"): torch.FloatStorage,
        ("torch", "LongStorage"): torch.LongStorage,
        ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
        ("yacs.config", "CfgNode"): __import__("yacs.config", fromlist=["CfgNode"]).CfgNode,
    }

    class RU(pickle.Unpickler):
        def find_class(self, m, n):
            if (m, n) in AUDITED:
                return AUDITED[(m, n)]
            raise pickle.UnpicklingError(f"blocked global: {m}.{n}")

    rp = types.ModuleType("rp")
    rp.Unpickler = RU
    rp.load = lambda f, **k: RU(f, **{x: v for x, v in k.items()
                                      if x in ("fix_imports", "encoding", "errors")}).load()
    rp.loads = lambda b, **k: rp.load(io.BytesIO(b), **k)
    return functools.partial(torch.load, pickle_module=rp, weights_only=False)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def main() -> int:
    import os
    import torch
    _orig = torch.load
    torch.load = restricted_torch_load()
    from torchreid.scripts.builder import build_config
    from torchreid.tools.feature_extractor import KPRFeatureExtractor
    from torchreid.metrics.distance import compute_distance_matrix_using_bp_features
    cwd = os.getcwd()
    os.chdir(KPR)                      # yaml's load_weights path is CWD-relative
    try:
        cfg = build_config(config_path=str(KPR / __import__("os").environ.get("KPR_CFG", "configs/kpr/imagenet/kpr_occ_posetrack_test.yaml")))
        cfg.use_gpu = False
        cfg.test.batch_size = 8                    # small batches: Swin-B on CPU, avoid OOM
        extractor = KPRFeatureExtractor(cfg)
    finally:
        os.chdir(cwd)
        torch.load = _orig

    key = f"{GAME}_{TAG}"
    gt = json.loads((REPO / f"data/gt_players/{key}.json").read_text())
    dets, pose = {}, {}
    for ang in ANGLES:
        z = np.load(REPO / f"runs/dets_cache/{GAME}_{ang}_{TAG}_small_1280_t0.25.dets.npz")
        m = defaultdict(list)
        for di, (b, c, f) in enumerate(zip(z["boxes"], z["classes"], z["frame_idx"])):
            m[int(f)].append((di, [float(v) for v in b], int(c)))
        dets[ang] = m
        p = np.load(REPO / f"runs/pose_cache/{GAME}_{ang}_{TAG}.pose.npz")
        pose[ang] = {(int(f), int(d)): (k, s) for f, d, k, s in
                     zip(p["frame_idx"], p["det_idx"], p["kpts"], p["kscores"])}

    # candidate crops per player: (isolated? , ang, cf, det_row)
    cands = {pl: {"iso": [], "pile": []} for pl in PLAYERS}
    for pl in PLAYERS:
        v = gt[pl]
        sel = {int(f): s for f, s in v["frames"].items() if s and str(f) in v.get("approved", {})}
        for f, s in sel.items():
            for ang in s:
                cf = f + OFFS[ang]
                rows = dets[ang].get(cf, [])
                if s[ang] >= len(rows):
                    continue
                di, box, _ = rows[s[ang]]
                if box[3] - box[1] < MIN_H:
                    continue
                overlaps = [b2 for dj, b2, c2 in rows if dj != di and c2 == 0 and iou(box, b2) > 0.15]
                bucket = "pile" if overlaps else "iso"
                cands[pl][bucket].append((ang, cf, di, box))

    caps = {ang: cv2.VideoCapture(str(REPO / f"data/clips/{GAME}_{ang}_{TAG}.mp4")) for ang in ANGLES}
    frame_cache = {}

    def read_frame(ang, cf):
        if (ang, cf) not in frame_cache:
            caps[ang].set(cv2.CAP_PROP_POS_FRAMES, cf)
            ok, img = caps[ang].read()
            frame_cache[(ang, cf)] = img if ok else None
            if len(frame_cache) > 40:
                frame_cache.pop(next(iter(frame_cache)))
        return frame_cache[(ang, cf)]

    def build_sample(ang, cf, di, box):
        img = read_frame(ang, cf)
        if img is None:
            return None
        ih, iw = img.shape[:2]
        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
        x2, y2 = min(iw, int(box[2])), min(ih, int(box[3]))
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        ch, cw = crop.shape[:2]

        def to_crop(k, ks):
            """Shift to crop coords; clip inside; zero conf for points that were outside."""
            kc = k - [x1, y1]
            inside = ((kc[:, 0] >= 0) & (kc[:, 0] <= cw - 1)
                      & (kc[:, 1] >= 0) & (kc[:, 1] <= ch - 1))
            kc[:, 0] = np.clip(kc[:, 0], 0, cw - 1)
            kc[:, 1] = np.clip(kc[:, 1], 0, ch - 1)
            return np.concatenate([kc, (ks * inside)[:, None]], axis=1), inside

        kp = pose[ang].get((cf, di))
        sample = {"image": crop}
        if kp is not None:
            kxyc, _ = to_crop(*kp)
            sample["keypoints_xyc"] = kxyc
        negs = []
        for dj, b2, c2 in dets[ang].get(cf, []):
            if dj == di or c2 != 0:
                continue
            kp2 = pose[ang].get((cf, dj))
            if kp2 is None:
                continue
            nxyc, inside = to_crop(*kp2)
            if inside.sum() >= 3:
                negs.append(nxyc)
        sample["negative_kps"] = np.array(negs) if negs else np.zeros((0, 17, 3))
        return sample

    rng = np.random.RandomState(0)
    gal_samples, gal_ids, q_samples, q_ids, q_meta = [], [], [], [], []
    for pl in PLAYERS:
        iso = cands[pl]["iso"]
        idx = rng.choice(len(iso), min(GALLERY_PER_PLAYER, len(iso)), replace=False)
        for i in idx:
            smp = build_sample(*iso[i])
            if smp:
                gal_samples.append(smp); gal_ids.append(pl)
        pile = cands[pl]["pile"]
        qidx = rng.choice(len(pile), min(MAX_QUERIES_PER_PLAYER, len(pile)), replace=False)
        for i in qidx:
            smp = build_sample(*pile[i])
            if smp:
                q_samples.append(smp); q_ids.append(pl); q_meta.append(pile[i][:2])
    for c in caps.values():
        c.release()
    print(f"gallery: {len(gal_samples)} isolated crops | queries: {len(q_samples)} pile-up crops")

    _, g_emb, g_vis, _ = extractor(gal_samples)
    _, q_emb, q_vis, _ = extractor(q_samples)
    D, _ = compute_distance_matrix_using_bp_features(q_emb, g_emb, q_vis, g_vis,
                                                     use_gpu=False, use_logger=False)
    D = D.cpu().detach().numpy()

    gal_ids = np.array(gal_ids)
    ok_all = ok_kit = n_kit = 0
    per_pl = defaultdict(lambda: [0, 0])
    for qi, true_pl in enumerate(q_ids):
        # identity score = mean of 3 closest gallery samples of that identity
        scores = {pl: np.sort(D[qi][gal_ids == pl])[:3].mean() for pl in PLAYERS}
        pred = min(scores, key=scores.get)
        hit = pred == true_pl
        ok_all += hit
        per_pl[true_pl][0] += hit; per_pl[true_pl][1] += 1
        if true_pl in SAME_KIT:
            n_kit += 1
            # same-kit-only decision: restrict candidates to the trio
            pred_kit = min(SAME_KIT, key=lambda p: scores[p])
            ok_kit += (pred_kit == true_pl)
    print(f"\nKPR pile-up identification (top-1 among 4 players, chance 25%): "
          f"{ok_all}/{len(q_ids)} = {ok_all/len(q_ids):.0%}")
    print(f"SAME-KIT trio only (#22/#43/#6, chance 33%): {ok_kit}/{n_kit} = {ok_kit/max(1,n_kit):.0%}")
    for pl, (h, n) in sorted(per_pl.items()):
        print(f"  {pl}: {h}/{n} = {h/max(1,n):.0%}")
    out = {"gallery": len(gal_samples), "queries": len(q_samples),
           "top1_all": round(ok_all/len(q_ids), 3), "top1_same_kit": round(ok_kit/max(1,n_kit), 3),
           "per_player": {pl: [h, n] for pl, (h, n) in per_pl.items()}}
    (REPO / f"runs/tracking/ledger/kprpilot_{key}.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
