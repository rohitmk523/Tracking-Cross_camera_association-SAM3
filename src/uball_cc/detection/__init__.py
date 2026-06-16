"""Detection stage: predictor interface + adapters.

The eval harness consumes any object implementing `Detector.predict(image_bgr)`
-> list of `Detection(box_xyxy, score, class_id)`. Adapters:
  - DummyDetector  : deterministic, dependency-free; proves the harness runs.
  - RFDETRDetector : loads our trained RF-DETR weights (optional `baseline` deps).
See docs/04_detection.md for the model and output contract.
"""
