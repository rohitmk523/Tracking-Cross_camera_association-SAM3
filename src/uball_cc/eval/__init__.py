"""Evaluation harness (see docs/13_evaluation.md).

Implemented now:
  - detection: mAP@50 / mAP@50-95 per class, overall AND on the far-endline ROI
    band (the headline small-player metric).

Stubs (interfaces + acceptance bars; filled as later stages land):
  - tracking: MOTA / IDF1 / ID-switches per camera.
  - cross-camera: ID-consistency, global IDF1, occlusion-recovery, duplicate-ID.
"""
