#!/usr/bin/env python3
"""SAM2 scrum-window prototype — the ladder's last rung.

For each wrong-WHO crowd shot: seed SAM2 with every candidate's box at a
CLEAN frame ~2.2s before release (identities known there), propagate masks
through the scrum, and at release pick the candidate whose mask is at the
ball. Scores SAM2's WHO vs GT on the 20-shot probe set.

  .venv/bin/python scripts/sam2_scrum_proto.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FPS = 29.97


def main() -> int:
    sel = json.loads(Path("/tmp/sam2_sel.json").read_text())
    from ultralytics.models.sam import SAM2VideoPredictor
    ok = tot = pred_was = 0
    for i, s in enumerate(sel):
        clip = REPO / f"runs/sam2_proto/w{i:02d}.mp4"
        if not clip.exists():
            continue
        cands = s["cands"][:6]
        boxes = [c[1] for c in cands]
        overrides = dict(conf=0.25, task="segment", mode="predict",
                         imgsz=1024, model="sam2.1_s.pt", save=False,
                         verbose=False)
        predictor = SAM2VideoPredictor(overrides=overrides)
        try:
            results = predictor(source=str(clip), bboxes=boxes, stream=True)
            rel_idx = s["rel"] - s["seed_f"]          # frame of release in clip
            # ball position at release (image coords of arc cam)
            ball_xy = s.get("ball_xy")
            masks_at_rel = None
            for fi, r in enumerate(results):
                if fi == rel_idx:
                    masks_at_rel = (r.masks.data.cpu().numpy()
                                    if r.masks is not None else None)
                    break
        except Exception as e:
            print(f"w{i:02d} SAM2 error: {e}")
            continue
        tot += 1
        gt_last = s["gt"].split()[-1]
        pred_was += bool(s["pred"] and s["pred"].rstrip("?").split()[-1] == gt_last)
        if masks_at_rel is None or ball_xy is None:
            print(f"w{i:02d}: no masks/ball at release")
            continue
        bx, by = ball_xy
        best_j, best_d = None, 1e9
        H, W = masks_at_rel.shape[-2:]
        for j, m in enumerate(masks_at_rel):
            ys, xs = np.where(m > 0.5)
            if len(xs) == 0:
                continue
            d = float(np.min(np.hypot(xs - bx, ys - by)))
            if d < best_d:
                best_d, best_j = d, j
        if best_j is None:
            print(f"w{i:02d}: empty masks")
            continue
        pick = cands[best_j][0]
        # stream -> name last token via roster
        roster = json.loads((REPO / "data/rosters/c2a354fe.json").read_text())
        from collections import defaultdict
        by_num = defaultdict(list)
        for p in roster["players"]:
            by_num[p["num"]].append(p)
        core = pick.lstrip("n")
        kit = core[-1] if core[-1] in ("B", "W") else None
        num = int(core.rstrip("BW"))
        c = by_num.get(num, [])
        if len(c) == 1:
            nm = c[0]["name"]
        else:
            m2 = [p for p in c if p["team"] == {"B": 1, "W": 2}.get(kit)]
            nm = m2[0]["name"] if len(m2) == 1 else (c[0]["name"] if c else pick)
        hit = nm.split()[-1] == gt_last
        ok += hit
        print(f"w{i:02d}: gt={s['gt'][:18]:<18} sam2={nm[:18]:<18} "
              f"{'OK' if hit else 'MISS'} (mask-ball d={best_d:.0f}px)")
    print(f"\nSAM2 scrum WHO: {ok}/{tot} recovered "
          f"(baseline was {pred_was}/{tot} — all wrong by construction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
