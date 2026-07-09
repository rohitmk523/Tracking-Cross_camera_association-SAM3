#!/usr/bin/env python3
"""DENSE jersey anchors for the offline identity solver (v2 keystone).

Roboflow-article principle, executed with our stronger reader: read numbers
relentlessly — every camera, every frame, every crop the legibility gate passes —
and emit RAW per-frame anchor events (not per-track attributes). Each event clamps
identity at one instant; the solver repairs identity outward from every clamp.

Reads detection boxes from the version-independent cache, so anchors are valid for
any tracking version. Output: runs/anchors/{game}_{tag}.jersey_anchors.json
  [{frame(ref timeline), cam, box, number, conf}]

  python scripts/extract_jersey_anchors.py --game e6fba750 --tag 44_60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ANGLES = ("FL", "FR", "NL", "NR")
OFFS = {"e6fba750_44_60": {"FL": 0, "FR": -11, "NL": -1, "NR": -1},
        "c2a354fe_300_60": None}
MIN_BOX_H = 90                   # even far-ish crops: the legibility gate does the filtering


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--stride", type=int, default=1,
                    help="read every Nth frame (2 = ~15 reads/s, plenty of anchor density)")
    a = ap.parse_args()

    from uball_cc.tracking.jersey_stack import JerseyStack

    key = f"{a.game}_{a.tag}"
    offs = OFFS.get(key)
    if offs is None:
        from uball_cc.fusion.audiosync import audio_offset_seconds
        offs = {"FL": 0}
        ref = str(REPO / f"data/clips/{a.game}_FL_{a.tag}.mp4")
        for ang in ("FR", "NL", "NR"):
            s, _ = audio_offset_seconds(ref, str(REPO / f"data/clips/{a.game}_{ang}_{a.tag}.mp4"))
            offs[ang] = int(round(s * 29.97))

    stack = JerseyStack()
    anchors = []
    for ang in ANGLES:
        z = np.load(REPO / f"runs/dets_cache/{a.game}_{ang}_{a.tag}_small_1280_t0.25.dets.npz")
        by_f: dict[int, list] = {}
        for b, s, c, f in zip(z["boxes"], z["scores"], z["classes"], z["frame_idx"]):
            if int(c) == 0 and float(s) >= 0.3:
                by_f.setdefault(int(f), []).append([float(v) for v in b])
        cap = cv2.VideoCapture(str(REPO / f"data/clips/{a.game}_{ang}_{a.tag}.mp4"))
        n_read = n_hit = 0
        pos = -1
        for clip_f in sorted(by_f):
            if clip_f % a.stride:
                continue
            if clip_f != pos + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, clip_f)
            ok, img = cap.read()
            pos = clip_f
            if not ok:
                continue
            ih, iw = img.shape[:2]
            for b in by_f[clip_f]:
                if b[3] - b[1] < MIN_BOX_H:
                    continue
                x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
                x2, y2 = min(iw, int(b[2])), min(ih, int(b[3]))
                if x2 - x1 < 20 or y2 - y1 < MIN_BOX_H:
                    continue
                n_read += 1
                num, conf = stack.read_crop(img[y1:y2, x1:x2])
                if num is not None:
                    n_hit += 1
                    anchors.append({"frame": clip_f - offs[ang], "cam": ang,
                                    "box": [round(v, 1) for v in b],
                                    "number": int(num), "conf": round(conf, 3)})
        cap.release()
        print(f"{ang}: {n_read} crops gated, {n_hit} confident reads", flush=True)

    out = REPO / f"runs/anchors/{key}.jersey_anchors.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"game": a.game, "tag": a.tag, "stride": a.stride,
                               "offsets": offs, "anchors": anchors}))
    from collections import Counter
    per_num = Counter(x["number"] for x in anchors)
    print(f"TOTAL: {len(anchors)} anchor events -> {out}")
    print("per number:", dict(per_num.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
