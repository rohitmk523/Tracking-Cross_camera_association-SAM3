"""Dataset tooling: audit + consolidation of OUR detection annotations.

OUR-FOOTAGE-ONLY policy (docs/04, docs/10): only our 4-fixed-camera footage +
operator annotations are used for training. Third-party / NBA / COCO data is
never trained on. This package audits the existing `Training_frameworks/*`
annotations and consolidates the player/referee/ball subset into one RF-DETR
(YOLO-format) dataset with cross-court whole-game splits.
"""
