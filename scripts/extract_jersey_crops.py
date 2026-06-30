#!/usr/bin/env python3
"""Build a jersey-NUMBER annotation pool: extract LARGE player crops from the event-anchored
frames (docs/05). The number is only legible when the player is near a camera and big, so we
keep only tall crops -- the same "where it's annotatable" logic as the ball.

Reuses the operator's player boxes in data/annotate_pool_events/labels (class 0). Each kept
crop becomes one item the operator labels with a jersey number (or 'none'/'unclear').

  python scripts/extract_jersey_crops.py --pool data/annotate_pool_events \
      --out data/jersey_pool --min-h 150
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/annotate_pool_events", help="source frames+labels")
    ap.add_argument("--out", default="data/jersey_pool")
    ap.add_argument("--min-h", type=int, default=150, help="min player-box height (px) to keep a crop")
    ap.add_argument("--pad", type=float, default=0.08, help="fraction of box size padded around the crop")
    a = ap.parse_args()

    import cv2

    img_dir = Path(a.pool) / "images"
    lbl_dir = Path(a.pool) / "labels"
    out_dir = Path(a.out) / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    items, kept, total = [], 0, 0
    for lab in sorted(lbl_dir.glob("*.txt")):
        img_path = img_dir / f"{lab.stem}.jpg"
        if not img_path.exists():
            continue
        rows = [ln.split() for ln in lab.read_text().splitlines() if ln.strip()]
        players = [r for r in rows if r and r[0] == "0"]
        if not players:
            continue
        img = None
        for j, r in enumerate(players):
            total += 1
            cx, cy, bw, bh = (float(x) for x in r[1:5])
            img = cv2.imread(str(img_path)) if img is None else img
            if img is None:
                break
            H, W = img.shape[:2]
            if bh * H < a.min_h:                              # too small -> number unreadable
                continue
            px, py = bw * W * a.pad, bh * H * a.pad
            x1, y1 = max(0, int((cx - bw / 2) * W - px)), max(0, int((cy - bh / 2) * H - py))
            x2, y2 = min(W, int((cx + bw / 2) * W + px)), min(H, int((cy + bh / 2) * H + py))
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            name = f"{lab.stem}_p{j}.jpg"
            cv2.imwrite(str(out_dir / name), crop)
            items.append({"crop": name, "frame": lab.stem, "box_h_px": round(bh * H)})
            kept += 1
    (Path(a.out) / "items.json").write_text(json.dumps(items))
    print(f"kept {kept} LARGE player crops (of {total} player boxes, min_h={a.min_h}px) -> {out_dir}")
    print(f"manifest -> {Path(a.out) / 'items.json'}  | next: scripts/annotate_jersey.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
