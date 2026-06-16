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

## Data — OUR footage only (hard rule)
**Train exclusively on our own 4-fixed-camera footage + operator annotations. Do NOT mix in
COCO, NBA/broadcast, or any third-party sports dataset for training.** Our fixed-camera,
specific-court, specific-angle domain is too different from broadcast/COCO imagery — generic
data *hurts* transfer here, it doesn't help. (Caveat that is **not** a violation: the RF-DETR
**DINOv2 backbone** is pretrained on general self-supervised visual features — that's generic
*visual representation*, not sports/NBA data, and is unavoidable for any modern detector. The
**detection training data is ours only.**)

Phase 1 (reuse what we have):
- **Reuse the operator's already-annotated images in `Training_frameworks`** (Far Angle, Near
  Angle, 4Cam, E6 — our footage) converted to RF-DETR/COCO *format* (format ≠ dataset). If any
  third-party Roboflow/NBA subset was ever mixed into those folders, **exclude it** — ours only.
- Consolidate all annotated player/ref/ball frames into one multi-court dataset.
- **Cross-game / cross-court splits** (held-out court for validation; see
  [14_games_and_clips](14_games_and_clips.md)) — never random-frame splits.
- **Expand later**: operator-annotated far-endline small-player crops as the priority next
  batch (~1,000–1,500 frames / ~10k small instances eventually); start with existing data.

See [10_data_and_licensing](10_data_and_licensing.md).

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
