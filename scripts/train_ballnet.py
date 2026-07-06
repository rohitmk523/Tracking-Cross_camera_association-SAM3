#!/usr/bin/env python3
"""Train BallNet — the small motion-aware ball detector (TrackNet/WASB-class design).

Input: frame TRIPLET (9ch, 512x288) -> tiny U-Net -> ball heatmap. Deliberately small so
it can run in the Jetson's real-time loop; the motion context is what lets it find a ball
the appearance detector misses mid-court. Trained on SAM3 pseudo-labels
(build_ball_dataset.py), validated leave-games-out; the final exam is the operator GT
(scripts/eval_events_gt.py) — beat 14% possession / stop inventing turnovers.

  python scripts/train_ballnet.py                 # -> runs/ball/ballnet.pt + report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
W, H = 512, 288
SIGMA = 4.0                    # gaussian target radius (px at 512x288)
HIT_PX = 10.0                  # eval: peak within this of GT = detected


def build_model():
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))

    class BallNet(nn.Module):
        def __init__(self, base=16):
            super().__init__()
            import torch.nn as nn
            self.e1, self.e2, self.e3 = block(9, base), block(base, base * 2), block(base * 2, base * 4)
            self.pool = nn.MaxPool2d(2)
            self.mid = block(base * 4, base * 8)
            self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
            self.d3 = block(base * 8, base * 4)
            self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
            self.d2 = block(base * 4, base * 2)
            self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
            self.d1 = block(base * 2, base)
            self.head = nn.Conv2d(base, 1, 1)

        def forward(self, x):
            import torch
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            m = self.mid(self.pool(e3))
            d3 = self.d3(torch.cat([self.u3(m), e3], 1))
            d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.head(d1)                     # logits heatmap (B,1,H,W)
    return BallNet()


def make_loader(root: Path, split: str, batch: int):
    import cv2
    import numpy as np
    import torch

    labels = json.loads((root / "labels.json").read_text())
    items = [(k, v) for k, v in sorted(labels.items()) if v["split"] == split]

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, i):
            name, lab = items[i]
            trip = cv2.imread(str(root / "images" / name))
            xs = np.split(trip, 3, axis=1)                       # 3x (H,W,3)
            is_neg = bool(lab.get("neg"))
            lx, ly = (-1e6, -1e6) if is_neg else (lab["x"], lab["y"])
            if split == "train":
                if np.random.rand() < 0.5:                       # horizontal flip
                    xs = [f[:, ::-1] for f in xs]
                    lx = W - 1 - lx
                a = np.random.uniform(0.7, 1.3)                  # brightness/contrast jitter
                b = np.random.uniform(-25, 25)
                xs = [np.clip(f.astype(np.float32) * a + b, 0, 255) for f in xs]
            x = np.concatenate([np.ascontiguousarray(f) for f in xs],
                               axis=2).astype(np.float32) / 255.0
            x = torch.from_numpy(x).permute(2, 0, 1)             # 9,H,W
            yy, xx = np.mgrid[0:H, 0:W]
            hm = np.exp(-((xx - lx) ** 2 + (yy - ly) ** 2) / (2 * SIGMA ** 2))
            return x, torch.from_numpy(hm.astype(np.float32))[None], \
                torch.tensor([float(lx), float(ly)])   # neg -> hm==0, xy==-1e6 sentinel

    return torch.utils.data.DataLoader(DS(), batch_size=batch, shuffle=(split == "train"),
                                       num_workers=0), len(items)


def evaluate(model, loader, device):
    """(hit-rate on POSITIVE frames, mean peak prob on NEGATIVE frames)."""
    import torch
    model.eval()
    hits = n_pos = 0
    neg_peaks = []
    with torch.no_grad():
        for x, _, xy in loader:
            logits = model(x.to(device))
            b, _, h, w = logits.shape
            flat = logits.view(b, -1)
            peak, arg = flat.max(1)
            px, py = (arg % w).float().cpu(), (arg // w).float().cpu()
            is_neg = xy[:, 0] < -1e5
            d = ((px - xy[:, 0]) ** 2 + (py - xy[:, 1]) ** 2) ** 0.5
            hits += int((d[~is_neg] <= HIT_PX).sum())
            n_pos += int((~is_neg).sum())
            if is_neg.any():
                neg_peaks.extend(torch.sigmoid(peak[is_neg.to(device)]).cpu().tolist())
    neg_mean = sum(neg_peaks) / len(neg_peaks) if neg_peaks else 0.0
    return hits / max(1, n_pos), neg_mean


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ball_dataset")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="runs/ball/ballnet.pt")
    a = ap.parse_args()
    import torch

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    train, n_tr = make_loader(Path(a.data), "train", a.batch)
    val, n_va = make_loader(Path(a.data), "val", a.batch)
    print(f"train {n_tr} / val {n_va} triplets on {device}")
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    # heavily class-imbalanced heatmap: weighted BCE keeps the peak from collapsing to 0
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(200.0, device=device))
    best, best_state = -10.0, None
    best_acc, best_neg = 0.0, 1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for x, hm, _ in train:
            loss = lossf(model(x.to(device)), hm.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        sched.step()
        acc, neg_peak = evaluate(model, val, device)
        score = acc - neg_peak                        # reward hits, punish phantom peaks
        star = ""
        if score > best:
            best = score
            best_acc, best_neg = acc, neg_peak
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            star = "  <- best"
        print(f"epoch {ep:>2}: loss {tot / max(1, len(train)):.4f} | "
              f"val hit@{HIT_PX:.0f}px {acc:.3f} | neg-peak {neg_peak:.3f}{star}", flush=True)
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state or model.state_dict(), out)
    (out.parent / "ballnet_report.json").write_text(json.dumps(
        {"val_hit_at_px": best_acc, "val_neg_peak": best_neg,
         "hit_px": HIT_PX, "n_train": n_tr, "n_val": n_va,
         "note": "leave-games-out val; teacher = SAM3 pseudo-labels; final exam = "
                 "eval_events_gt vs operator GT"}, indent=2))
    print(f"\nbest: hit@{HIT_PX:.0f}px {best_acc:.3f}, neg-peak {best_neg:.3f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
