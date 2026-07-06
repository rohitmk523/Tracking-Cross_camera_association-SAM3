#!/usr/bin/env python3
"""Run BallNet over a clip -> per-frame ball position + confidence (pixel coords).

  python scripts/ballnet_infer.py --video data/clips/e6fba750_NR_47_12.mp4 \
      --out runs/ball/e6fba750_NR_47_12.ball.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from train_ballnet import H, W, build_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default="runs/ball/ballnet.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    import cv2
    import numpy as np
    import torch

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device).eval()
    model.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=True))
    model = model.to(device)

    cap = cv2.VideoCapture(a.video)
    ow = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920
    oh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(img, (W, H)).astype(np.float32) / 255.0)
    cap.release()

    out: dict[str, list] = {}
    trip_idx = list(range(1, len(frames) - 1))
    with torch.no_grad():
        for s in range(0, len(trip_idx), a.batch):
            batch_f = trip_idx[s:s + a.batch]
            x = torch.stack([torch.from_numpy(
                np.concatenate([frames[f - 1], frames[f], frames[f + 1]], axis=2))
                .permute(2, 0, 1) for f in batch_f]).to(device)
            logits = model(x)
            b, _, h, w = logits.shape
            flat = logits.view(b, -1)
            peak, arg = flat.max(1)
            conf = torch.sigmoid(peak).cpu().numpy()
            px = (arg % w).float().cpu().numpy() * (ow / W)
            py = (arg // w).float().cpu().numpy() * (oh / H)
            for k, f in enumerate(batch_f):
                out[str(f)] = [round(float(px[k]), 1), round(float(py[k]), 1),
                               round(float(conf[k]), 3)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out))
    confs = [v[2] for v in out.values()]
    import numpy as np
    print(f"{Path(a.video).stem}: {len(out)} frames | conf median "
          f"{np.median(confs):.2f} p90 {np.percentile(confs, 90):.2f} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
