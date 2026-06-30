#!/usr/bin/env python3
"""Drop pre-label PLAYER boxes that are really the BALL (the 3-class detector stamps a
ball-sized 'player' box on top of every ball). These duplicates slow annotation AND would
teach 'ball = player' if trained on. Removes a class-0 box only when it tightly overlaps a
class-2 box and is ball-sized (so real shooters holding the ball are kept).

  python scripts/dedupe_ball_player.py --pool data/annotate_pool_events
"""
from __future__ import annotations

import argparse
from pathlib import Path

IOU_MIN = 0.45        # tight overlap with a ball box
AREA_MULT = 4.0       # ...and player area < 4x the ball box => it's the ball, not a person


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def dedupe_lines(lines: list[str]) -> tuple[list[str], int]:
    rows = []
    for ln in lines:
        p = ln.split()
        if len(p) >= 5:
            rows.append((int(p[0]), [float(x) for x in p[1:5]], ln))
    balls = [r[1] for r in rows if r[0] == 2]
    kept, dropped = [], 0
    for cls, box, ln in rows:
        if cls == 0 and any(_iou(box, b) > IOU_MIN and box[2] * box[3] < AREA_MULT * b[2] * b[3]
                            for b in balls):
            dropped += 1
            continue
        kept.append(ln)
    return kept, dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/annotate_pool_events")
    a = ap.parse_args()
    lbl_dir = Path(a.pool) / "labels"
    total_dropped, files_changed = 0, 0
    for f in sorted(lbl_dir.glob("*.txt")):
        lines = [ln for ln in f.read_text().splitlines() if ln.strip()]
        kept, dropped = dedupe_lines(lines)
        if dropped:
            f.write_text("\n".join(kept) + ("\n" if kept else ""))
            total_dropped += dropped
            files_changed += 1
    print(f"removed {total_dropped} ball-on-player duplicate player boxes "
          f"across {files_changed} files -> {lbl_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
