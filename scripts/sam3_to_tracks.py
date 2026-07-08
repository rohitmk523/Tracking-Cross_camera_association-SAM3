#!/usr/bin/env python3
"""SAM3 video-tracker output -> our per-camera tracks json (drop-in for the pipeline).

SAM3's own masklet ids become local track ids (the chunk-architecture premise: SAM3
is detector AND within-clip tracker; our team/jersey/fusion layer rides on top).

  python scripts/sam3_to_tracks.py --sam3 runs/sam3_demo60/e6fba750_FL_44_60.sam3.json \
      --cam FL --out runs/tracking_sam3/e6fba750_FL_44_60.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3", required=True)
    ap.add_argument("--cam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-score", type=float, default=0.4)
    a = ap.parse_args()

    d = json.loads(Path(a.sam3).read_text())
    rows = []
    unassigned = 100000                      # SAM3 boxes without an id get unique ids
    for f_str, dets in d["frames"].items():
        f = int(f_str)
        for r in dets:
            if r["score"] < a.min_score:
                continue
            tid = r.get("id")
            if tid is None:
                tid = unassigned
                unassigned += 1
            x1, y1, x2, y2 = r["box"]
            rows.append({
                "cam": a.cam, "frame": f, "track_id": int(tid),
                "box_xyxy": [round(float(v), 2) for v in (x1, y1, x2, y2)],
                "foot_xy": [round(float(x1 + x2) / 2, 2), round(float(y2), 2)],
                "score": float(r["score"]),
                "class_id": int(r.get("cls", 0)),
                "class": "player" if int(r.get("cls", 0)) == 0 else "referee",
                "team": None, "jersey": None,
            })
    rows.sort(key=lambda r: (r["frame"], r["track_id"]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_ids = len({r["track_id"] for r in rows})
    out.write_text(json.dumps({"cam": a.cam, "detector": "sam3-video",
                               "source": a.sam3,
                               "stats": {"rows": len(rows), "track_ids": n_ids},
                               "tracks": rows}))
    print(f"{a.cam}: {len(rows)} rows, {n_ids} SAM3 track ids -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
