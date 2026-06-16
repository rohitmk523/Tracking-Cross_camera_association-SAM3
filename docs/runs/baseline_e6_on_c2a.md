# Baseline — e6 RF-DETR-nano on c2a (held-out, cross-game)

The "current models" baseline (docs/13) the first RF-DETR-S run must beat. Model:
the existing `e6_rfdetr_best.pth` (RF-DETR-**nano**, trained on e6 player/referee),
evaluated by `scripts/eval_detection.py` on the **c2a354fe** held-out game (1200
imgs, all 4 angles). Cross-game, never trained on c2a.

```
python scripts/eval_detection.py --detector rfdetr \
  --weights "../Training_frameworks/Uball E6 Demo/runs/e6_rfdetr_best.pth" \
  --model nano --resolution 1280 --threshold 0.25 --split test
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
| ball | n/a (-1) | no ball GT / no ball in model |
| **far-endline band** player recall@0.5 | **0.745** | smallest-quartile players |
| — FL | 0.518 | far cam, weak |
| — **FR** | **0.063** | far cam, near-total miss ⚠️ |
| — NL | 0.817 | near cam, strong |
| — NR | 0.760 | near cam |

## What this tells the first training run
1. **Player generalizes** (78.7 mAP@50 cross-game) — RF-DETR-S @1280 should push higher.
2. **Referee is broken cross-game** (0.7 mAP@50). Root cause likely too few/!consistent
   ref labels (e6 train = 475 ref). **Add referee data** + check player/ref confusion.
3. **Small / far-endline players are the headline gap**: AP_small=0 and far-cam band
   recall is 6% (FR) / 52% (FL). This is the quantified justification for **SAHI +
   far-endline-ROI tiling + targeted endline data** (docs/04). The FR=6% vs FL=52%
   asymmetry warrants a look (FR coverage/calibration vs genuine geometry).
4. ball=0 everywhere (no labels) — unmeasurable until annotated.

> Targets for run 1 (RF-DETR-S @1280, e6→c2a): beat player mAP@50 0.787 and — the
> figure of merit — lift far-endline band recall (esp. FL/FR) above the nano
> baseline; referee is a data problem, not a model-size problem.
