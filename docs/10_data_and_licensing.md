# 10 · Data & Licensing

## Two rules
1. **Train on OUR footage only** (operator decision). No COCO, no NBA/broadcast, no
   third-party sports datasets in *training* — our fixed-camera domain is too different and
   generic data hurts transfer. (Model **backbones** pretrained on general visual features —
   DINOv2 for RF-DETR, etc. — are fine and unavoidable; that's representation pretraining, not
   sports training data.)
2. **Ship-clean**: weights ship only if trained on commercially-usable data. Our own footage
   qualifies. Non-commercial/NDA datasets are **benchmark/method-validation only** — never in
   shipped weights, and per rule 1 not in training at all.

## Train (OUR data only)
| Source | What | License | Use |
|---|---|---|---|
| **Our 4-camera game footage + operator annotations** (`Training_frameworks/*` + new) | player/ref/ball boxes, court keypoints, across our games/courts | **ours** | The ONLY detection/keypoint training data |
| **Our jersey-number crops** | digit crops from our footage | ours | Jersey-ResNet (reuse + improve) |
| **Our player crops** (from our tracks) | identity crops | ours | ReID fine-tune (on our players) |

> If any third-party Roboflow/NBA subset was ever mixed into the existing
> `Training_frameworks` folders, **exclude it** before training — ours only.

## Benchmark / method-validation only (NOT shipped, NOT trained on)
These validate methods and let us read SOTA numbers; per rule 1 they are **never used to
train our models** and per rule 2 never in shipped weights.
| Source | What | License |
|---|---|---|
| **TrackID3x3** | indoor fixed-cam 3×3 basketball tracking (closest to our rig) | Apache-2.0 (data terms vary) |
| **SportsMOT** (basketball split) | tracking boxes + IDs | CC BY-NC 4.0 |
| **DeepSportradar instance-seg** | basketball masks + occlusion metric | CC BY-NC-ND 4.0 |
| **SoccerNet** (tracking/reid/jersey) | methods + volume | NDA/research |
| **DanceTrack** | same-uniform robustness proxy | CC BY-NC |
| MOT17/20 | crowding | CC BY-NC-SA |

## Annotation (operator is sole annotator → must be production-grade)
A comprehensive, resumable annotation workflow (port + harden the DEMO_UBALL FastAPI tool):
- **Pre-labeling**: run the current best detector/jersey models to pre-fill boxes + numbers,
  so the operator *corrects* rather than draws from scratch (proven 10× speed-up in prior work).
- **Classes/tasks**: player, referee, ball; jersey-number boxes + digit; court keypoints (for
  calibration). One tool, task-switchable.
- **Resumable state** (`review_state.json`), per-image approve, keyboard-driven (drag=new box,
  1/2/3 reclassify, E edit digit, D delete, U undo).
- **Quality gates**: per-batch QA frames (overlay) to catch courtside/bench false positives;
  cross-court coverage targets; explicit far-endline small-player batches.
- **Priority batches**: (1) far-endline small players, (2) new courts for generalization,
  (3) jersey numbers for the general OCR, (4) occlusion-heavy windows.
- **Versioning**: datasets versioned (e.g. DVC or dated S3 prefixes); every trained model
  records its dataset version for reproducibility.

## License hygiene checklist (before any ship)
- [ ] No AGPL/GPL code in the shippable path (YOLO11, BoxMOT, sn-gamestate excluded).
- [ ] No NC/NDA data in shipped weights.
- [ ] Third-party Roboflow subset licenses verified.
- [ ] VLM = paid API or Apache-2.0 on-prem model (no copyleft).
- [ ] Model cards record training-data provenance + license per model.
