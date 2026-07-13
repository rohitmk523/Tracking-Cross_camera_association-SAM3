#!/usr/bin/env python3
"""Run the shot-detection repo's P2 features + P3 make/miss on a P1 tracks
parquet (theirs OR our adapter's) and score vs GT labels. Measures whether
their frozen P3 transfers to OUR detector's features — the cheap alternative
to a $25-30 twenty-game re-extraction.

Two model variants per run:
  frozen   their production p3_model_angleaware.joblib (e6 was in its train
           set — optimistic on e6; fair for cross-parquet A/B deltas)
  logo     replication of their honest benchmark: retrain their exact recipe
           (seed 42) on their v8far features EXCLUDING the eval game, same
           balanced-threshold rule on their val split (eval game absent)

  .venv/bin/python scripts/shotdet_transfer_eval.py \
      --tracks /tmp/p1tracks/e6fba750-....parquet --label ours \
      [--restrict-to runs/shotdet_ab/their_e6_tracks.parquet]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SD = REPO.parent / "uball_shot_detection_dual_fusion_v2"
sys.path.insert(0, str(SD / "pipeline"))

GT_LOCAL = REPO / "runs/shotdet_ab/gt_windows.json"
THR = 0.310                     # recovered: reproduces their saved joblib preds
PARAMS = dict(max_depth=4, min_samples_leaf=40, learning_rate=0.08,
              max_iter=500, random_state=42)
SUFFIX = re.compile(r"_(FL|FR|NL|NR)$")


def angle_map() -> dict:
    gt = json.loads(GT_LOCAL.read_text())["games"]
    return {s["play_id"]: s.get("angle", "LEFT")
            for sh in gt.values() for s in sh}


def fold_angleaware(v: pd.DataFrame) -> pd.DataFrame:
    """Their p3_angleaware.build_angle_aware(), applied to an arbitrary
    feature frame instead of the fixed v8far parquet."""
    side = v.play_id.map(angle_map()).fillna("LEFT")
    is_left = (side == "LEFT").to_numpy()
    families: dict[str, dict] = {}
    for c in v.columns:
        m = SUFFIX.search(c)
        if m:
            families.setdefault(c[: m.start()], {})[m.group(1)] = c
    new_cols, drop_raw = {}, []
    for base, d in families.items():
        if all(k in d for k in ("FL", "FR", "NL", "NR")):
            fl, fr = v[d["FL"]].to_numpy(float), v[d["FR"]].to_numpy(float)
            nl, nr = v[d["NL"]].to_numpy(float), v[d["NR"]].to_numpy(float)
            new_cols[f"{base}_FARSIDE"] = np.where(is_left, fl, fr)
            new_cols[f"{base}_NEARSIDE"] = np.where(is_left, nl, nr)
            drop_raw += list(d.values())
    aa = pd.concat([v.drop(columns=drop_raw).reset_index(drop=True),
                    pd.DataFrame(new_cols)], axis=1)
    aa["side_is_left"] = is_left.astype(int)
    return aa


def score(tag: str, y: np.ndarray, prob: np.ndarray, thr: float,
          cls: pd.Series) -> None:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    yp = (prob >= thr).astype(int)
    print(f"  [{tag}] acc={accuracy_score(y, yp):.4f} "
          f"prec={precision_score(y, yp, zero_division=0):.3f} "
          f"rec={recall_score(y, yp, zero_division=0):.3f} "
          f"auc={roc_auc_score(y, prob):.4f} (n={len(y)}, thr={thr})")
    per = {c: round(float((yp[cls.values == c] == y[cls.values == c]).mean()), 3)
           for c in sorted(cls.unique())}
    print(f"    per-class acc: {per}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--label", default="tracks")
    ap.add_argument("--restrict-to", default=None,
                    help="second P1 parquet; keep only plays present in both")
    a = ap.parse_args()

    import p2_dataset as P2                      # their feature code, unchanged

    df = pd.read_parquet(a.tracks)
    gid = df.game_id.iloc[0]
    if a.restrict_to:
        keep = set(pd.read_parquet(a.restrict_to, columns=["play_id"])
                   .play_id.unique())
        df = df[df.play_id.isin(keep)]
    print(f"[{a.label}] {gid}: {df.play_id.nunique()} plays, {len(df)} rows")

    rows = P2._shot_rows(gid, df)
    feats = pd.DataFrame(rows)
    # g_* geometry features are track-derived too (build_v8far recipe):
    # recomputed from the SAME tracks parquet under eval.
    from geometry_features import build_geometry
    feats = feats.merge(build_geometry({gid: df}).drop(columns=["game_id"]),
                        on="play_id", how="left")
    aa = fold_angleaware(feats)

    canon = list(np.load(REPO / "runs/shotdet_ab/canon_cols.npy",
                         allow_pickle=True))
    # nm_* (net-motion) features come from their separate video-ROI pass
    # (extract_netmotion.py), NOT from detection tracks — identical for both
    # detectors (rim-anchored ROI on the same video). Join their values by
    # play_id so the A/B isolates the 180 track-derived features.
    nm_cols = [c for c in canon if c.startswith("nm_")]
    v8aa = pd.read_parquet(
        SD / "data/p2_features_v8far_angleaware.parquet",
        columns=["play_id"] + nm_cols)
    aa = aa.merge(v8aa, on="play_id", how="left")
    missing = [c for c in canon if c not in aa.columns]
    if missing:
        raise SystemExit(f"feature mismatch — {len(missing)} canon cols "
                         f"missing, e.g. {missing[:5]}")
    X = aa[canon].astype(float).values
    y = aa.label.values.astype(int)
    print(f"features: {X.shape[1]} cols x {X.shape[0]} shots "
          f"(makes={int(y.sum())}, misses={int(len(y)-y.sum())})")

    import joblib
    frozen = joblib.load(SD / "data/p3_model_angleaware.joblib")
    score("frozen P3", y, frozen.predict_proba(X)[:, 1], THR, aa.classification)

    # honest variant: retrain excluding the eval game (their LOGO protocol)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import precision_score, recall_score
    v8 = pd.read_parquet(SD / "data/p2_features_v8far_angleaware.parquet")
    tr = v8[(v8.game_id != gid) & (v8.split == "train")]
    va = v8[(v8.game_id != gid) & (v8.split == "val")]
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("clf", HistGradientBoostingClassifier(**PARAMS))])
    pipe.fit(tr[canon].astype(float).values, tr.label.values)
    pv = pipe.predict_proba(va[canon].astype(float).values)[:, 1]
    yv = va.label.values
    best = (0.5, -1e9)
    for t in np.arange(0.20, 0.85, 0.005):
        pred = (pv >= t).astype(int)
        if pred.sum() == 0:
            continue
        pr = precision_score(yv, pred, zero_division=0)
        rc = recall_score(yv, pred, zero_division=0)
        s = -(max(0.90 - pr, 0) + max(0.90 - rc, 0))
        if s > best[1]:
            best = (float(t), s)
    score(f"logo (excl {gid[:8]})", y, pipe.predict_proba(X)[:, 1],
          best[0], aa.classification)

    out = REPO / f"runs/shotdet_ab/eval_{a.label}.json"
    probs = frozen.predict_proba(X)[:, 1]
    recs = aa[["play_id", "classification", "label"]].copy()
    recs["prob_frozen"] = probs
    recs["pred_frozen"] = (probs >= THR).astype(int)
    recs.to_json(out, orient="records", indent=1)
    print(f"  per-shot predictions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
