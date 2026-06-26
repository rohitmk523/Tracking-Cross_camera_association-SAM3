#!/usr/bin/env python3
"""Per-track ReID embeddings (OSNet) + a GT-free sanity check (docs/05).

  python scripts/extract_reid.py --video data/clips/e6fba750_FL_47_12.mp4 \
      --tracks runs/tracking/e6fba750_FL_47_12_teams.json \
      --out runs/tracking/e6fba750_FL_47_12_reid.npz

Validation (no GT needed): mean ReID cosine similarity should be HIGHER within a
team than across teams — teammates share appearance — a cheap signal the embeddings
are meaningful for the cross-camera matching they'll feed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True, help="tracklets JSON (ideally with teams)")
    ap.add_argument("--sample-per-track", type=int, default=6)
    ap.add_argument("--model", default="osnet_x1_0")
    ap.add_argument("--model-path", default="", help="ReID-trained weights (e.g. market1501)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
    from uball_cc.tracking import Track
    from uball_cc.tracking.reid import OSNetEmbedder, track_embeddings

    data = json.loads(Path(a.tracks).read_text())
    tracks = [Track.from_record(r) for r in data["tracks"]]
    emb = track_embeddings(a.video, tracks, sample_per_track=a.sample_per_track,
                           embedder=OSNetEmbedder(a.model, a.model_path))
    print(f"player tracks embedded: {len(emb)}  (dim={next(iter(emb.values())).shape[0] if emb else 0})")

    # --- GT-free sanity: within-team vs across-team cosine similarity ---
    team_of = {t.track_id: t.team for t in tracks if t.class_id == 0}
    ids = [i for i in emb if team_of.get(i) in ("A", "B")]
    if len(ids) >= 3:
        x = np.stack([emb[i] for i in ids])               # already L2-normalized
        sim = x @ x.T
        teams = [team_of[i] for i in ids]
        within, across = [], []
        for p in range(len(ids)):
            for q in range(p + 1, len(ids)):
                (within if teams[p] == teams[q] else across).append(float(sim[p, q]))
        mw, ma = (np.mean(within) if within else float("nan")), (np.mean(across) if across else float("nan"))
        print(f"mean cosine — within-team: {mw:.3f}   across-team: {ma:.3f}")
        print(f"  -> within > across: {mw > ma}  (ReID captures team/appearance)")
    else:
        print("not enough teamed tracks for the within/across check")

    out = Path(a.out) if a.out else Path(a.tracks).with_name(Path(a.tracks).stem + "_reid.npz")
    np.savez(out, ids=np.array(list(emb.keys())),
             emb=np.stack(list(emb.values())) if emb else np.zeros((0, 512)))
    print(f"embeddings -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
