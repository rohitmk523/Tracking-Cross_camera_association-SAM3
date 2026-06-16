"""Detection evaluation: COCO mAP (overall + per-class + area) and the
far-endline ROI-band player recall (docs/13). Ground truth is the per-split
COCO JSON emitted by the consolidation step.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np

from ..detection.base import Detection, Detector
from ..data.provenance import ANGLES


def _angle_of(file_name: str) -> str | None:
    for tok in Path(file_name).stem.split("_"):
        if tok in ANGLES:
            return tok
    return None


def _iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _in_band(box_xyxy, h: int, band: tuple[float, float]) -> bool:
    foot_y = box_xyxy[3]                       # bottom of box = foot point
    return band[0] * h <= foot_y <= band[1] * h


def run_predictions(detector: Detector, coco_gt, images_dir: Path,
                    max_images: int | None = None):
    """Run the detector over GT images. Returns (coco_predictions, per_image_dets)."""
    import cv2
    preds: list[dict] = []
    per_image: dict[int, list[Detection]] = {}
    img_ids = sorted(coco_gt.getImgIds())
    if max_images:
        img_ids = img_ids[:max_images]
    for img_id in img_ids:
        info = coco_gt.loadImgs(img_id)[0]
        img = cv2.imread(str(images_dir / info["file_name"]))
        if img is None:
            per_image[img_id] = []
            continue
        dets = detector.predict(img)
        per_image[img_id] = dets
        for d in dets:
            x1, y1, x2, y2 = d.box_xyxy
            preds.append({"image_id": img_id, "category_id": d.class_id + 1,
                          "bbox": [x1, y1, x2 - x1, y2 - y1], "score": d.score})
    return preds, per_image, img_ids


def _coco_summarize(coco_gt, preds, img_ids, cat_ids):
    from pycocotools.cocoeval import COCOeval
    if not preds:
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.imgIds = img_ids
        if cat_ids is not None:
            ev.params.catIds = cat_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev.stats  # [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]


def evaluate_detection(detector: Detector, coco_gt_path: Path, images_dir: Path,
                       roi_cfg: dict, max_images: int | None = None) -> dict:
    from pycocotools.coco import COCO
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(coco_gt_path))
    cats = {c["id"]: c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())}

    preds, per_image, img_ids = run_predictions(detector, coco_gt, images_dir, max_images)

    overall = _coco_summarize(coco_gt, preds, img_ids, None)
    result: dict = {
        "detector": getattr(detector, "name", type(detector).__name__),
        "gt": str(coco_gt_path), "n_images": len(img_ids), "n_predictions": len(preds),
        "overall": _stats_dict(overall),
        "per_class": {},
    }
    for cid, name in cats.items():
        s = _coco_summarize(coco_gt, preds, img_ids, [cid])
        result["per_class"][name] = _stats_dict(s)

    result["far_endline_band"] = _roi_band_recall(
        coco_gt, per_image, img_ids, roi_cfg, player_cat_id=_player_cat(cats))
    return result


def _player_cat(cats: dict[int, str]) -> int:
    for cid, name in cats.items():
        if name == "player":
            return cid
    return min(cats)            # fallback: first category


def _stats_dict(stats) -> dict:
    if stats is None:
        return {"mAP_50_95": None, "mAP_50": None, "AP_small": None,
                "AR_100": None, "note": "no predictions"}
    return {
        "mAP_50_95": round(float(stats[0]), 4),
        "mAP_50": round(float(stats[1]), 4),
        "AP_small": round(float(stats[3]), 4),
        "AP_medium": round(float(stats[4]), 4),
        "AP_large": round(float(stats[5]), 4),
        "AR_100": round(float(stats[8]), 4),
        "AR_small": round(float(stats[9]), 4),
    }


def _band_threshold(coco_gt, img_ids, roi_cfg, player_cat_id) -> tuple[str, float | None]:
    """Resolve the small-player band threshold (px box-height) for the run."""
    mode = roi_cfg.get("band_mode", "small_box_percentile")
    if mode == "small_box_px":
        return mode, float(roi_cfg.get("small_box_h_px", 130))
    if mode == "small_box_percentile":
        heights = []
        for img_id in img_ids:
            for a in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id)):
                if a["category_id"] == player_cat_id:
                    heights.append(a["bbox"][3])
        if not heights:
            return mode, None
        p = float(roi_cfg.get("small_box_percentile", 25))
        return mode, float(np.percentile(heights, p))
    return mode, None        # y_band handled separately


def _gt_in_band(box_xyxy, h: int, angle: str, mode: str, thr: float | None,
                roi_cfg: dict) -> bool:
    if mode == "y_band":
        yb = roi_cfg.get("y_band", {})
        band = tuple(yb.get("per_angle", {}).get(angle, yb.get("default", [0.0, 0.45])))
        return _in_band(box_xyxy, h, band)
    if thr is None:
        return False
    return (box_xyxy[3] - box_xyxy[1]) <= thr        # box height <= small threshold


def _roi_band_recall(coco_gt, per_image, img_ids, roi_cfg, player_cat_id) -> dict:
    """Greedy IoU@0.5 player recall on the far/small-player band.

    Default band = smallest `small_box_percentile`% of player boxes by height
    (size-based, court-agnostic). See configs/eval_roi.yaml.
    """
    mode, thr = _band_threshold(coco_gt, img_ids, roi_cfg, player_cat_id)
    gt_total = matched = 0
    by_angle: dict[str, dict] = {}
    for img_id in img_ids:
        info = coco_gt.loadImgs(img_id)[0]
        h = info["height"]
        angle = _angle_of(info["file_name"]) or "NA"
        gt_boxes = [_xywh_to_xyxy(a["bbox"])
                    for a in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
                    if a["category_id"] == player_cat_id
                    and _gt_in_band(_xywh_to_xyxy(a["bbox"]), h, angle, mode, thr, roi_cfg)]
        preds = sorted([d for d in per_image.get(img_id, [])
                        if d.class_id + 1 == player_cat_id
                        and _gt_in_band(d.box_xyxy, h, angle, mode, thr, roi_cfg)],
                       key=lambda d: -d.score)
        used = [False] * len(gt_boxes)
        m = 0
        for d in preds:
            best, bj = 0.5, -1
            for j, g in enumerate(gt_boxes):
                if used[j]:
                    continue
                iou = _iou_xyxy(d.box_xyxy, g)
                if iou >= best:
                    best, bj = iou, j
            if bj >= 0:
                used[bj] = True
                m += 1
        gt_total += len(gt_boxes)
        matched += m
        a = by_angle.setdefault(angle, {"gt": 0, "matched": 0})
        a["gt"] += len(gt_boxes)
        a["matched"] += m
    for a in by_angle.values():
        a["recall@0.5"] = round(a["matched"] / a["gt"], 4) if a["gt"] else None
    return {
        "band_mode": mode,
        "small_box_h_px_threshold": round(thr, 1) if thr else None,
        "player_gt_in_band": gt_total,
        "player_matched": matched,
        "recall@0.5": round(matched / gt_total, 4) if gt_total else None,
        "per_angle": by_angle,
        "note": "size-based proxy for far/small players; swap to calibrated court ROI (docs/03)",
    }


def _xywh_to_xyxy(b):
    return (b[0], b[1], b[0] + b[2], b[1] + b[3])
