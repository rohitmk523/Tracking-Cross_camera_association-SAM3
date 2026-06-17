"""Human-in-the-loop annotation tool for the Model-A player/referee/ball dataset.

Ported + simplified from the DEMO_UBALL / Training_frameworks FastAPI annotator:
flat pool (data/annotate_pool/images + labels), 3 classes (0=player, 1=referee,
2=ball), resumable per-image approval (review_state.json), keyboard-driven, YOLO
in/out (RF-DETR reads YOLO directly). Pre-labels come from scripts/prelabel.py;
the operator CORRECTS rather than draws (~10x faster).
"""
