# 10 · Data & Licensing

## Principle: segregate "train/ship" from "benchmark/research"
Because we ship commercially, **weights that ship may only be trained on commercially-usable
data.** Non-commercial / NDA datasets are powerful but **benchmark-and-method-validation only**
— never in shipped weights.

## Train (ship-clean)
| Source | What | License | Use |
|---|---|---|---|
| **Operator's own footage + annotations** (`Training_frameworks/*`) | player/ref/ball boxes across games/courts (Far Angle, Near Angle, 4Cam, E6) | **ours** | Primary Phase-1 detection data (convert YOLO→RF-DETR/COCO) |
| **Our jersey-number crops** | digit crops | ours | Jersey-ResNet (reuse + improve) |
| Roboflow basketball sets | player/ball/hoop, court keypoints | **CC BY 4.0** (verify each) | Augment detection + court-keypoint model |
| SpaceJam | action clips + 2D pose | MIT | Optional pose/action aux |
| DeepSportradar-ReID | basketball player crops/IDs | Apache-2.0 (repo) | ReID fine-tune |

> Action item: **verify the license of any third-party Roboflow subset** already mixed into
> the existing `Training_frameworks` data before it enters a shipped model. The operator's own
> game footage/annotations are clean.

## Benchmark / research only (NOT shipped)
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
