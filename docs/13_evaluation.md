# 13 · Evaluation

Every stage has a metric and an acceptance bar. Measure on a **held-out court** (never trained
on) to prove generalization, not overfit.

## Detection ([04](04_detection.md))
- **mAP@50, mAP@50-95** per class (player/ref/ball), cross-court.
- **Player recall — overall AND far-endline band** (the metric that matters most; report the
  endline ROI separately).
- Baseline: current models. Targets to set after the first RF-DETR run; directionally, beat
  the existing per-class recall on the held-out court, with the endline band as the headline.

## Per-camera tracking ([05](05_tracking_and_masks.md))
- **MOTA, IDF1, ID switches** per camera (benchmark on SportsMOT basketball split / TrackID3x3
  — research-license, eval only).
- Jersey-read precision (must be high — it's the identity authority); team-classification
  accuracy.

## Cross-camera fusion ([06](06_cross_camera_fusion.md)) — the headline
- **Cross-camera ID-consistency**: fraction of players that keep **one** global ID across all
  cameras for the whole clip (custom metric; the demo's failures were here).
- **Global IDF1 / ID switches** at the fused level.
- **Occlusion-recovery rate**: after a player is occluded in a view, does the global ID persist
  and re-attach correctly.
- **Duplicate-ID rate**: two on-court dots never share a label; one `(team, number)` → one
  track. (The demo's "#1 is two people" / "#25 single-angle" bugs become explicit metrics.)

## Pose ([07](07_pose.md))
- 2D keypoint coverage per tracked player (fraction with a confident skeleton).
- (Follow-on) 3D triangulation reprojection error per camera.

## World model ([08](08_world_model.md))
- **Event detection** precision/recall vs hand-labeled events (shots, make/miss, possession
  changes, blocks) on a held-out game.
- State faithfulness spot-checks (positions/possession correct).

## VLM narration ([09](09_vlm_narration.md))
- **Factual correctness** of narration vs ground-truth events (did the right player do the
  right thing?) — human rubric on a sample of possessions.
- **Hallucination rate** (claims not supported by the world model/frames).
- **Frame-fetch efficiency**: % of events resolved from state alone vs needing a frame
  (lower frame-fetch = the architecture working).
- Latency/cost per possession.

## Acceptance for v1
- End-to-end run on a **previously-unseen court** producing: correct cross-camera global IDs
  through occlusions, a populated event stream, and descriptive narration that is factually
  correct on the majority of possessions — with frame fetches only on the expected event types.

## Regression discipline
- Fixed eval set (held-out court + game) versioned alongside datasets.
- Every model/pipeline change re-runs the harness; metrics tracked in `runs/` with the
  dataset + model versions recorded (model cards).
