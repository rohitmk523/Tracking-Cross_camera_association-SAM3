#!/usr/bin/env python3
"""v2 — GTA-style tracklet CONNECT (per camera, before team/jersey assignment).

A track that breaks mid-court resumes seconds later under a new id; downstream that
becomes a duplicate global identity. This post-pass re-joins fragments: candidate
pairs (A ends before B starts, gap <= max-gap) are scored by OSNet appearance
(sampled crops per tracklet) + motion-extrapolated endpoint distance, then merged
greedily best-score-first with chain follow-up. Evidence basis: GTA connect stage,
ByteTrack+GTA on SportsMOT = +12.4 IDF1 (the continuity metric).

  python scripts/link_tracklets.py --video data/clips/X.mp4 \
      --tracks runs/tracking/X.json --out runs/tracking/X.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

N_CROPS = 8                      # sampled crops per tracklet for its appearance signature


def _embed_tracklets(video: str, rows_by_id: dict, model_path: str) -> dict:
    import cv2

    from uball_cc.tracking.reid import OSNetEmbedder
    emb = OSNetEmbedder(model_path="" if model_path is None else model_path)
    wanted: dict[int, list] = defaultdict(list)      # frame -> [(tid, box)]
    for tid, rows in rows_by_id.items():
        rs = sorted(rows, key=lambda r: -(r["box_xyxy"][2] - r["box_xyxy"][0]))[:N_CROPS]
        for r in rs:
            wanted[r["frame"]].append((tid, r["box_xyxy"]))
    crops: dict[int, list] = defaultdict(list)
    cap = cv2.VideoCapture(video)
    for f in sorted(wanted):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        ih, iw = img.shape[:2]
        for tid, (x1, y1, x2, y2) in wanted[f]:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(iw, int(x2)), min(ih, int(y2))
            if x2 - x1 > 10 and y2 - y1 > 20:
                crops[tid].append(img[y1:y2, x1:x2])
    cap.release()
    out = {}
    for tid, cs in crops.items():
        v = emb.embed(cs).mean(axis=0)
        out[tid] = v / (np.linalg.norm(v) + 1e-8)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", default=None, help="default: rewrite --tracks in place")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--max-gap", type=int, default=150, help="max frames between fragments")
    ap.add_argument("--min-cos", type=float, default=0.55)
    ap.add_argument("--base-radius", type=float, default=60.0,
                    help="px endpoint slack at gap=0; grows 6px/frame, capped 420px")
    a = ap.parse_args()

    from uball_cc.tracking.reid import default_reid_weights
    model_path = a.model_path or default_reid_weights()

    d = json.loads(Path(a.tracks).read_text())
    rows_by_id: dict[int, list] = defaultdict(list)
    for r in d["tracks"]:
        if r["class_id"] in (0, 1):
            rows_by_id[r["track_id"]].append(r)
    for rows in rows_by_id.values():
        rows.sort(key=lambda r: r["frame"])

    embs = _embed_tracklets(a.video, rows_by_id, model_path)

    def endpoint(rows, tail=False):
        r = rows[-1] if tail else rows[0]
        b = r["box_xyxy"]
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]), r["frame"]

    def velocity(rows):
        pts = [((r["box_xyxy"][0] + r["box_xyxy"][2]) / 2,
                (r["box_xyxy"][1] + r["box_xyxy"][3]) / 2) for r in rows[-8:]]
        if len(pts) < 2:
            return np.zeros(2)
        return (np.array(pts[-1]) - np.array(pts[0])) / max(1, len(pts) - 1)

    tids = sorted(rows_by_id)
    cands = []
    for i in tids:
        pi, fi = endpoint(rows_by_id[i], tail=True)
        vi = velocity(rows_by_id[i])
        cls_i = rows_by_id[i][0]["class_id"]
        for j in tids:
            if i == j:
                continue
            pj, fj = endpoint(rows_by_id[j], tail=False)
            gap = fj - fi
            if not (0 < gap <= a.max_gap):
                continue
            if rows_by_id[j][0]["class_id"] != cls_i:
                continue
            pred = pi + vi * min(gap, 30)            # extrapolate briefly, then trust radius
            dist = float(np.linalg.norm(pred - pj))
            radius = min(a.base_radius + 6.0 * gap, 420.0)
            if dist > radius:
                continue
            if i not in embs or j not in embs:
                continue
            cos = float(np.dot(embs[i], embs[j]))
            if cos < a.min_cos:
                continue
            cands.append((cos - dist / 1000.0, i, j))

    cands.sort(reverse=True)
    parent: dict[int, int] = {}
    tail_used: set[int] = set()
    head_used: set[int] = set()
    n_link = 0
    for _, i, j in cands:
        if i in tail_used or j in head_used:
            continue
        tail_used.add(i)
        head_used.add(j)
        parent[j] = i
        n_link += 1

    def root(t):
        seen = set()
        while t in parent and t not in seen:
            seen.add(t)
            t = parent[t]
        return t

    remap = {t: root(t) for t in tids}
    for r in d["tracks"]:
        r["track_id"] = remap.get(r["track_id"], r["track_id"])

    out = Path(a.out or a.tracks)
    out.write_text(json.dumps(d))
    n_before, n_after = len(tids), len({root(t) for t in tids})
    print(f"link: {n_link} joins, tracklets {n_before} -> {n_after} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
