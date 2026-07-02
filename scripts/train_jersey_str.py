#!/usr/bin/env python3
"""Fine-tune PARSeq (STR) on our jersey-number crops — the audit-approved recognizer plan.

Open-vocabulary scene-text recognition instead of a closed-set classifier: unseen numbers
stay readable, and PARSeq starts at 0.78 zero-shot on these crops (runs/jersey/
parseq_zeroshot.json). Trains with the model's OWN training_step (permutation-LM loss)
in a plain loop — no Lightning Trainer, so it survives PL version drift. MPS/CPU-friendly
at this data scale; re-run any time more labels land (data via build_jersey_dataset.py).

  python scripts/train_jersey_str.py                          # ~minutes on MPS
  # -> runs/jersey/parseq_jersey.pt (+ report.json with val before/after)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _loader(root: Path, tf, batch: int, shuffle: bool):
    import torch
    from PIL import Image

    items = [(p, p.parent.name) for p in sorted(root.rglob("*.jpg"))]

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, k):
            p, lab = items[k]
            return tf(Image.open(p).convert("RGB")), lab

    return torch.utils.data.DataLoader(DS(), batch_size=batch, shuffle=shuffle), len(items)


def _evaluate(model, loader, device) -> tuple[float, list]:
    import torch
    model.eval()
    ok = n = 0
    rows = []
    with torch.no_grad():
        for imgs, labs in loader:
            logits = model(imgs.to(device))
            texts, _ = model.tokenizer.decode(logits.softmax(-1))
            for t, gt in zip(texts, labs):
                pred = "".join(ch for ch in t if ch.isdigit())
                ok += (pred == gt)
                n += 1
                rows.append({"gt": gt, "pred": pred})
    return (ok / n if n else 0.0), rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/jersey_dataset/recognizer")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=7e-5)
    ap.add_argument("--out", default="runs/jersey/parseq_jersey.pt")
    a = ap.parse_args()

    import torch
    from torchvision import transforms as T

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True).to(device)
    base = [T.Resize((32, 128), T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(0.5, 0.5)]
    train_tf = T.Compose([T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3, 0.05)], 0.7),
                          T.RandomApply([T.GaussianBlur(3)], 0.3),
                          T.RandomAffine(6, translate=(0.06, 0.06), scale=(0.9, 1.1),
                                         shear=4, fill=127)] + base)
    val_tf = T.Compose(base)

    train, n_tr = _loader(Path(a.data) / "train", train_tf, a.batch, True)
    val, n_va = _loader(Path(a.data) / "val", val_tf, a.batch, False)
    print(f"train {n_tr} / val {n_va} crops on {device}")
    acc0, _ = _evaluate(model, val, device)
    print(f"val zero-shot: {acc0:.3f}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    best, best_state = acc0, None
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for k, (imgs, labs) in enumerate(train):
            # the model's own PL training_step computes the permutation-LM loss
            loss = model.training_step((imgs.to(device), list(labs)), k)
            if isinstance(loss, dict):
                loss = loss["loss"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
        acc, _ = _evaluate(model, val, device)
        star = ""
        if acc > best:
            best = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            star = "  <- best"
        print(f"epoch {ep:>2}: loss {tot / max(1, len(train)):.3f} | val {acc:.3f}{star}", flush=True)

    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state or model.state_dict(), out)
    _, rows = _evaluate(model, val, device)
    (out.parent / "parseq_jersey_report.json").write_text(json.dumps(
        {"val_zero_shot": round(acc0, 3), "val_best": round(best, 3),
         "n_train": n_tr, "n_val": n_va, "epochs": a.epochs, "device": device,
         "note": "group-split val (no same-play leakage); rebuild data as labels grow"},
        indent=2))
    print(f"\nbest val {best:.3f} (zero-shot {acc0:.3f}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
