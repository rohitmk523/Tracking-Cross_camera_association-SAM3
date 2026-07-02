#!/usr/bin/env python3
"""PARSeq ZERO-SHOT baseline on our annotated jersey-number crops (2026-07-02 audit, fix #2).

SoccerNet jersey-recognition winners fine-tune scene-text-recognition (STR) models rather than
train closed-set classifiers; PARSeq scores ~85% zero-shot on comparable sports crops (Koshkina
CVPRW'24). This measures it on OUR crops with NO training — the result calibrates how many
annotations the recognizer actually needs and validates the STR plan before any AWS spend.

Evaluates on data/jersey_dataset/recognizer/{train,val} (GT = folder name). Predictions are
digit-filtered. Reports overall / val-only accuracy, per-number recall, confusions (1-vs-7!).

  python scripts/eval_parseq_zeroshot.py            # -> runs/jersey/parseq_zeroshot.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/jersey_dataset/recognizer")
    ap.add_argument("--out", default="runs/jersey/parseq_zeroshot.json")
    ap.add_argument("--model", default="parseq", help="hub entry: parseq | parseq_tiny | abinet | trba")
    a = ap.parse_args()

    import torch
    from PIL import Image
    from torchvision import transforms as T

    model = torch.hub.load("baudm/parseq", a.model, pretrained=True, trust_repo=True).eval()
    tf = T.Compose([T.Resize((32, 128), T.InterpolationMode.BICUBIC), T.ToTensor(),
                    T.Normalize(0.5, 0.5)])   # PARSeq training transform (aspect handled by STR)

    rows = []
    for img_path in sorted(Path(a.data).rglob("*.jpg")):
        gt = img_path.parent.name
        split = img_path.parts[-3]
        img = tf(Image.open(img_path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            logits = model(img)
        text, conf = model.tokenizer.decode(logits.softmax(-1))
        pred_raw = text[0]
        pred = "".join(ch for ch in pred_raw if ch.isdigit())
        c = float(conf[0].min()) if len(conf[0]) else 0.0
        rows.append({"crop": img_path.name, "split": split, "gt": gt,
                     "pred": pred, "pred_raw": pred_raw, "conf": round(c, 3)})

    def acc(rs):
        return round(sum(r["pred"] == r["gt"] for r in rs) / max(1, len(rs)), 3)

    val = [r for r in rows if r["split"] == "val"]
    per_num = {n: acc([r for r in rows if r["gt"] == n])
               for n in sorted({r["gt"] for r in rows}, key=int)}
    conf_pairs = Counter((r["gt"], r["pred"]) for r in rows if r["pred"] != r["gt"])
    report = {"model": a.model, "n": len(rows), "acc_all": acc(rows),
              "n_val": len(val), "acc_val": acc(val),
              "per_number_acc": per_num,
              "top_confusions": [{"gt": g, "pred": p, "n": n} for (g, p), n in conf_pairs.most_common(10)],
              "rows": rows}
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"PARSeq({a.model}) zero-shot on {len(rows)} crops: ALL acc {report['acc_all']}"
          f" | VAL ({len(val)}) acc {report['acc_val']}")
    print("per-number:", per_num)
    print("top confusions:", report["top_confusions"][:6])
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
