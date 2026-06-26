#!/usr/bin/env python3
"""Pre-label the annotation pool so the operator CORRECTS boxes instead of drawing
from scratch (~10x faster, proven in prior work). Writes canonical YOLO labels
(0=player, 1=referee, 2=ball) per image. All RF-DETR (no YOLO):

  * player + referee  <- e6 RF-DETR (player/ref). Decent cross-game for players;
                         referee is weaker cross-game so you'll fix most refs.
  * ball              <- RF-DETR near ball+hoop model (Basketball class only).

  python scripts/prelabel.py                       # pre-label data/annotate_pool

Operator then corrects in the annotation UI. Pre-labels are a HEAD START, not GT.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TF = REPO.parent / "Training_frameworks"
# canonical ids
PLAYER, REFEREE, BALL = 0, 1, 2


def _yolo_line(cid: int, box, w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    return (f"{cid} {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} "
            f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(REPO / "data" / "annotate_pool"))
    # default to the NEW cross-game model (11 games, val mAP@50 0.93, player held-out 0.884) so
    # new games are pre-labeled by the strongest detector -> fewer operator corrections. The old
    # e6-only nano model is still selectable via --player-weights for comparison.
    ap.add_argument("--player-weights",
                    default=str(REPO / "runs" / "rfdetr-s-1280-ourdata-v1" / "best.pth"))
    ap.add_argument("--player-model", default="small", choices=("nano", "small"))
    ap.add_argument("--ball-weights",
                    default=str(REPO / "runs" / "rfdetr-rim-near-v1" / "best.pth"))
    ap.add_argument("--ball-model", default="small", choices=("nano", "small"))
    ap.add_argument("--player-thr", type=float, default=0.35)
    ap.add_argument("--ball-thr", type=float, default=0.30)
    ap.add_argument("--resolution", type=int, default=1280)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
    import cv2
    from rfdetr import RFDETRNano, RFDETRSmall

    def _load(weights, model):
        cls = RFDETRSmall if model == "small" else RFDETRNano
        return cls(pretrain_weights=weights, resolution=a.resolution)

    print(f"player/ref detector: {a.player_model} {a.player_weights}")
    pdet = _load(a.player_weights, a.player_model)
    print(f"ball detector:       {a.ball_model} {a.ball_weights}")
    bdet = _load(a.ball_weights, a.ball_model)

    img_dir = Path(a.pool) / "images"
    lbl_dir = Path(a.pool) / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted(img_dir.glob("*.jpg"))
    n = {"player": 0, "referee": 0, "ball": 0, "images": 0}

    for i, img in enumerate(imgs, 1):
        out = lbl_dir / f"{img.stem}.txt"
        if out.exists() and not a.overwrite:
            continue
        bgr = cv2.imread(str(img))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lines: list[str] = []
        # players (cls 0) + referees (cls 1) from the player detector
        pd = pdet.predict(rgb, threshold=a.player_thr)
        for j in range(len(pd.xyxy)):
            c = int(pd.class_id[j])
            cid = REFEREE if c == 1 else PLAYER
            lines.append(_yolo_line(cid, pd.xyxy[j], w, h))
            n["referee" if cid == REFEREE else "player"] += 1
        # ball (Basketball cls 0) from the ball detector; ignore hoop (cls 1)
        bd = bdet.predict(rgb, threshold=a.ball_thr)
        for j in range(len(bd.xyxy)):
            if int(bd.class_id[j]) == 0:
                lines.append(_yolo_line(BALL, bd.xyxy[j], w, h))
                n["ball"] += 1
        out.write_text("\n".join(lines) + ("\n" if lines else ""))
        n["images"] += 1
        if i % 100 == 0:
            print(f"  {i}/{len(imgs)} | player {n['player']} ref {n['referee']} ball {n['ball']}")

    print(f"\nPRE-LABELED {n['images']} imgs: {n['player']} player + {n['referee']} "
          f"referee + {n['ball']} ball -> {lbl_dir}")
    print("next: open the annotation UI and CORRECT (don't trust pre-labels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
