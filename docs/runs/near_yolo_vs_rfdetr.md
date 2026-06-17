# Near-angle ball+hoop detector: YOLO vs RF-DETR

Comparing the legacy near detector (YOLO11n, AGPL) against the new RF-DETR-Small
(Apache) on the **same held-out near test split** (`Training_frameworks/Uball Near
Angle/data/yolo_split/test`, 486 imgs; both detectors trained with this split held
out → fair). Identical metric (IoU@0.3 matching) via
`uball_shot_detection_dual_fusion_v2/near_v0/eval_detector.py` (YOLO) and
`scripts/eval_detector_rfdetr.py` (RF-DETR, same logic, RGB input).

## Detector metrics (conf 0.25, imgsz/resolution 1280)
| metric | YOLO v1 | YOLO v2 | **RF-DETR** |
|---|---|---|---|
| HOOP recall | 1.000 | 1.000 | **1.000** |
| HOOP precision | 0.998 | 1.000 | 0.996 |
| BALL recall | 0.786 | 0.776 | **0.858** |
| BALL precision | 0.885 | 0.939 | 0.869 |
| **BALL recall @ rim moment** (decides make/miss) | 0.825 | 0.799 | **0.902** |

RF-DETR overall test mAP (from training run_test): mAP@50-95 **0.847**, mAP@50
**0.940**, hoop AP 0.994, ball AP 0.700.

**Verdict:** RF-DETR detects **more balls, especially at the rim moment (+7.7 pts,
0.825→0.902)** — the frames that drive shot detection — at a small precision cost
(more FPs). Hoop is perfect for both.

## 5-min visual confirmation (game e6fba750 NR, UNSEEN by the near model)
1800 frames @ 6 fps: **hoop detected 100%** (avg conf 0.98), **ball detected 70.7%**
(avg conf 0.71). Annotated video: `runs/viz/e6_NR_rfdetr_annotated.mp4`.

## How this relates to the "90-93%" headline
The ~90-93% is **make/miss accuracy (`e2e_acc`) on spotted shots**, decided by a
SEPARATE ResNet18 classifier (`classifier_all17.pt`) — NOT the detector. The
detector drives **`spot_recall`** (how many shots are found) via ball-at-rim
detection + rim localization. So the +7.7 pt rim-moment ball-recall gain above
predicts **higher spot_recall** in the full pipeline; `e2e_acc` is classifier-bound
and should move only modestly (fed more / better-centered shots).

YOLO pipeline baseline (documented, 4 frozen games): `e2e_acc` mean **0.930**,
`spot_recall` ~0.82–0.92.

## Next
- Pipeline-level e2e: run RF-DETR through `near_v0/test_end_to_end.py` (RF-DETR
  detector adapter, classifier fixed) on the 4 frozen games → compare `spot_recall`
  + `e2e_acc` to the YOLO baseline. (Streams ~20-min NR windows from S3; GPU-bound.)
- After the far RF-DETR run finishes: far-angle detector comparison, then fusion.
