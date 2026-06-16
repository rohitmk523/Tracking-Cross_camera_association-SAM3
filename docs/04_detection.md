# 04 · Detection

## Model
**RF-DETR-Small @ imgsz 1280**, classes: `player`, `referee`, `ball`. (Court keypoints are a
separate model — see [03](03_cameras_and_calibration.md).)

Why RF-DETR over YOLO11:
- **License**: Apache-2.0 (YOLO11 is AGPL — excluded from a commercial ship).
- **Small objects / few-shot**: DINOv2 backbone → strong transfer; RF-DETR-S beats YOLO11-XL
  on COCO at ~⅓ latency. (Caveat: RF-DETR's AP-*small* isn't independently published — we
  validate on *our* far-endline crops, and keep a YOLO11-S+P2 internal A/B baseline.)

## The hard case: small players at the far endline
Players at the far baseline are tiny even for the near SuperView camera. Research-backed
levers, **ranked by impact** (do the top ones):
1. **SAHI sliced fine-tuning + sliced inference** — biggest win (+12–15 AP-small). At
   inference, apply SAHI **only to the fixed far-endline ROI band** (e.g. 512×512 tiles @
   0.2 overlap), not the whole frame — keeps latency sane.
2. **High input resolution** (1280; 640 is a non-starter — ball ≈13 px). >1536 not worth the
   Jetson latency.
3. **Targeted data** — chips/crops of small endline players; copy-paste & mosaic aug.
4. **P2 small-object head** (cheap, ~+1 mAP) if using the YOLO baseline.
5. **Bigger backbone** — **weakest lever for AP-small.** A small model at high-res tiles beats
   a big model at low res. Resist upsizing.

> Answer to "do we need more images / bigger model / RF-DETR vs YOLO": **RF-DETR-S + SAHI +
> high-res + targeted endline data** — not a bigger model. More *targeted* data (endline
> small players) matters; raw model size does not.

## Data (Phase 1: reuse what we have)
We **reuse the operator's already-annotated images in `Training_frameworks`** (the YOLO
training sets — Far Angle, Near Angle, 4Cam, E6) by converting to RF-DETR/COCO format. This
avoids a cold-start annotation effort.
- Consolidate all annotated player/ref/ball frames across the existing frameworks into one
  multi-court dataset.
- **Cross-game / cross-court splits** (train on some games, validate on a held-out court) —
  never random-frame splits (avoids leakage; the prior work confirmed whole-game splits
  generalize).
- **Expand later**: add far-endline small-player crops as the priority next batch
  (operator-annotated). Target eventually ~1,000–1,500 frames / ~10k small instances for the
  endline class, but start with existing data.

See [10_data_and_licensing](10_data_and_licensing.md) for license hygiene (operator's own
footage/annotations are clean; verify any third-party Roboflow subsets before shipping).

## Training
- AWS GPU (RF-DETR does **not** train on Apple MPS — confirmed; CUDA only). g5/g6 instance,
  bundle dataset → S3 → train → pull weights. (Port the `aws_train.py` pattern.)
- RF-DETR-Nano/Small tier; pin resolution 1280 (position-embedding grid must match at load).
- Early stopping on cross-court validation mAP.

## Outputs & contract
Per camera, per frame: `[{box_xyxy, score, class}]`. Foot point = bottom-center (refined by
pose ankles in [07](07_pose.md)). Handed to tracking ([05](05_tracking_and_masks.md)).

## Acceptance (v1)
- Player recall on a **held-out court**, reported overall **and for the far-endline band**
  specifically (the metric that matters). Baseline = current models; target set in
  [13_evaluation](13_evaluation.md).
