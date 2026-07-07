#!/usr/bin/env python3
"""Train the LEGIBILITY gate: is a jersey number readable in this player crop?

The gate that keeps garbage away from the STR recognizer (SoccerNet winners treat it as
essential — Koshkina CVPRW'24). Data comes free from annotation: 'legible' = crops the
operator boxed+numbered, 'illegible' = the 'unclear'/'none' marks that used to be thrown
away. ResNet-18 binary fine-tune; rebuild data + re-run as labels grow.

  python scripts/train_legibility.py     # -> runs/possession/possession_resnet18.pt + report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLASSES = ("no_ball", "has_ball")      # index = label


def _loader(root: Path, tf, batch: int, shuffle: bool):
    import torch
    from PIL import Image

    items = [(p, CLASSES.index(p.parent.name)) for p in sorted(root.rglob("*.jpg"))
             if p.parent.name in CLASSES]

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, k):
            p, y = items[k]
            return tf(Image.open(p).convert("RGB")), y

    n_pos = sum(y for _, y in items)
    return torch.utils.data.DataLoader(DS(), batch_size=batch, shuffle=shuffle), len(items), n_pos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/possession_crops")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="runs/possession/possession_resnet18.pt")
    a = ap.parse_args()

    import torch
    from torchvision import models
    from torchvision import transforms as T

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    base = [T.Resize((160, 96)), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    train_tf = T.Compose([T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3)], 0.6),
                          T.RandomHorizontalFlip()] + base)
    train, n_tr, pos_tr = _loader(Path(a.data) / "train", train_tf, a.batch, True)
    val, n_va, pos_va = _loader(Path(a.data) / "val", T.Compose(base), a.batch, False)
    print(f"train {n_tr} ({pos_tr} legible) / val {n_va} ({pos_va} legible) on {device}")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    lossf = torch.nn.CrossEntropyLoss()

    def evaluate() -> dict:
        model.eval()
        tp = tn = fp = fn = 0
        with torch.no_grad():
            for x, y in val:
                pred = model(x.to(device)).argmax(1).cpu()
                for p, g in zip(pred.tolist(), y.tolist()):
                    tp += (p == 1 and g == 1); tn += (p == 0 and g == 0)
                    fp += (p == 1 and g == 0); fn += (p == 0 and g == 1)
        n = tp + tn + fp + fn
        return {"acc": round((tp + tn) / n, 3) if n else 0.0,
                "legible_precision": round(tp / (tp + fp), 3) if tp + fp else None,
                "legible_recall": round(tp / (tp + fn), 3) if tp + fn else None}

    best, best_state = -1.0, None
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in train:
            loss = lossf(model(x.to(device)), y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        m = evaluate()
        star = ""
        if m["acc"] > best:
            best, best_state = m["acc"], {k: v.detach().cpu().clone()
                                          for k, v in model.state_dict().items()}
            star = "  <- best"
        print(f"epoch {ep:>2}: loss {tot / max(1, len(train)):.3f} | val {m}{star}", flush=True)

    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out)
    (out.parent / "legibility_report.json").write_text(json.dumps(
        {"val_best_acc": best, "n_train": n_tr, "n_val": n_va, "classes": CLASSES,
         "note": "one-game data so far — retrain as multi-game labels land"}, indent=2))
    print(f"\nbest val acc {best:.3f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
