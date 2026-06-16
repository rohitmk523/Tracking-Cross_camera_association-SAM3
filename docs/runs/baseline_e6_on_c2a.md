# Baseline — e6 RF-DETR-nano on c2a (held-out, cross-game)

The "current models" baseline (docs/13) the first RF-DETR-S run must beat. Model:
the existing `e6_rfdetr_best.pth` (RF-DETR-**nano**, trained on e6 player/referee),
evaluated by `scripts/eval_detection.py` on the **c2a354fe** held-out game (1200
imgs, all 4 angles). Cross-game, never trained on c2a.

Repro from a fresh clone (the eval JSON + dataset live in gitignored `data/`,
`runs/` — rebuild them; baseline weights come from the sibling Training_frameworks):
```
pip install -e '.[baseline]'
python scripts/build_detection_dataset.py          # build data/detect_consolidated
python scripts/eval_detection.py --detector rfdetr \
  --weights "../Training_frameworks/Uball E6 Demo/runs/e6_rfdetr_best.pth" \
  --model nano --resolution 1280 --threshold 0.25 --split test \
  --out runs/eval/detection_rfdetr_e6_on_c2a_test.json
```

## Numbers (runs/eval/detection_rfdetr_e6_on_c2a_test.json)
| metric | value | read |
|---|---|---|
| overall mAP@50 | 0.397 | dragged down by referee |
| overall mAP@[50:95] | 0.338 | |
| **player** mAP@50 | **0.787** | player detection generalizes cross-game ✅ |
| player mAP@[50:95] | 0.671 | |
| player **AP_small** (COCO) | **0.000** | tiniest objects are missed — the far-endline problem |
| **referee** mAP@50 | **0.007** | referee detection collapses cross-game ❌ |
| ball | n/a | no ball GT / no ball in model (reported as None) |
| **far-endline band** player recall@0.5 | **0.774** | smallest-quartile players |
| — FL | 0.597 | far cam, weak |
| — **FR** | **0.234** | far cam, weakest angle |
| — NL | 0.835 | near cam, strong |
| — NR | 0.780 | near cam |

> Band-recall numbers are **post-fix** (review #1): the pre-fix harness filtered
> predictions to the band before matching, deflating recall (overall 0.745, and
> FR 0.063 — most of that "FR near-total miss" was a measurement artifact, since
> the e6 detector predicts slightly over-sized boxes on FR that the buggy metric
> discarded). Corrected: overall 0.774, FR 0.234.

## What this tells the first training run
1. **Player generalizes** (78.7 mAP@50 cross-game) — RF-DETR-S @1280 should push higher.
2. **Referee is broken cross-game** (0.7 mAP@50). Root cause likely too few/!consistent
   ref labels (e6 train = 428 ref). **Add referee data** + check player/ref confusion.
3. **Small / far-endline players are the headline gap**: AP_small=0 and far-cam band
   recall is 23% (FR) / 60% (FL) vs 78-84% on the near cams. This is the quantified
   justification for **SAHI + far-endline-ROI tiling + targeted endline data**
   (docs/04). FR is genuinely the weakest angle (not the artifactual 6%) — far-right
   small players are the hardest case to close.
4. ball=0 everywhere (no labels) — unmeasurable until annotated.

> Targets for run 1 (RF-DETR-S @1280, e6→c2a): beat player mAP@50 0.787 and — the
> figure of merit — lift far-endline band recall (esp. FL/FR) above the nano
> baseline; referee is a data problem, not a model-size problem.
