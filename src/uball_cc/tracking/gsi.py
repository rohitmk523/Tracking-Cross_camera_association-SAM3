"""Gap interpolation for per-camera tracks (v1 ByteTrack tuning, research plan).

A track that blinks out for a few frames (detector confidence dip, brief occlusion)
and resumes under the SAME id leaves a hole in its row sequence. Downstream fusion
sees the hole as a dropout and may coast/re-associate. Filling short gaps by linear
interpolation keeps the track continuous — rows are marked interp=True with a low
score so consumers can discount them (fusion's zone/score weighting already does).
"""
from __future__ import annotations

from collections import defaultdict

from .types import Track


def fill_gaps(tracks: list[Track], max_gap: int = 20) -> tuple[list[Track], int]:
    by_id: dict[tuple, list[Track]] = defaultdict(list)
    for t in tracks:
        by_id[(t.cam, t.track_id, t.class_id)].append(t)

    out = list(tracks)
    n_fill = 0
    for rows in by_id.values():
        rows.sort(key=lambda r: r.frame)
        for a, b in zip(rows, rows[1:]):
            gap = b.frame - a.frame
            if gap <= 1 or gap > max_gap:
                continue
            for k in range(1, gap):
                w = k / gap
                box = tuple(a.box_xyxy[i] + w * (b.box_xyxy[i] - a.box_xyxy[i])
                            for i in range(4))
                out.append(Track(cam=a.cam, frame=a.frame + k, track_id=a.track_id,
                                 box_xyxy=box, score=0.3, class_id=a.class_id))
                n_fill += 1
    out.sort(key=lambda r: (r.frame, r.cam, r.track_id))
    return out, n_fill
