#!/usr/bin/env python3
"""Tag every jersey-anchor event with the kit shade (B dark / W bright).

Solves the dual-number problem (two teams wearing the same number): identity becomes
number x kit. Shade = torso brightness (pose-guided), z-normalized per camera; the
per-number split threshold is accepted only when clusters are kit-opposed (the
validated rule from the fine-tune dataset build, 89-100% label accuracy vs GT).

Rewrites runs/anchors/{key}.jersey_anchors.json in place, adding "kit": "B"|"W"|null
per event (null = number not dual or shade unavailable).

  python scripts/annotate_anchor_kits.py --game c2a354fe --tag 300_60 --dual-numbers 1,3,5
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")


def jersey_shade(crop, kpts, kscores):
    sh = [i for i in (5, 6) if kscores[i] >= 0.3]
    hp = [i for i in (11, 12) if kscores[i] >= 0.3]
    h, w = crop.shape[:2]
    if sh and hp:
        y1 = int(max(0, min(kpts[i][1] for i in sh)))
        y2 = int(min(h, max(kpts[i][1] for i in hp)))
        x1 = int(max(0, min(kpts[i][0] for i in sh + hp) - 5))
        x2 = int(min(w, max(kpts[i][0] for i in sh + hp) + 5))
    else:
        y1, y2, x1, x2 = int(0.2 * h), int(0.5 * h), int(0.25 * w), int(0.75 * w)
    if y2 <= y1 or x2 <= x1:
        return None
    hsv = cv2.cvtColor(crop[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    return float(np.median(hsv[:, :, 2]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dual-numbers", required=True, help="roster numbers worn by both teams")
    a = ap.parse_args()
    key = f"{a.game}_{a.tag}"
    dual = {int(x) for x in a.dual_numbers.split(",") if x}
    ap_path = REPO / f"runs/anchors/{key}.jersey_anchors.json"
    doc = json.loads(ap_path.read_text())
    offs = doc["offsets"]

    pose = {}
    for ang in ANGLES:
        p = np.load(REPO / f"runs/pose_cache/{a.game}_{ang}_{a.tag}.pose.npz")
        pose[ang] = {(int(f), int(d)): (k, s) for f, d, k, s in
                     zip(p["frame_idx"], p["det_idx"], p["kpts"], p["kscores"])}
    dets = {}
    for ang in ANGLES:
        z = np.load(REPO / f"runs/dets_cache/{a.game}_{ang}_{a.tag}_small_1280_t0.25.dets.npz")
        m = defaultdict(list)
        for di, (b, c, f) in enumerate(zip(z["boxes"], z["classes"], z["frame_idx"])):
            m[int(f)].append((di, [float(v) for v in b]))
        dets[ang] = m

    def iou(x, y):
        ix1, iy1 = max(x[0], y[0]), max(x[1], y[1])
        ix2, iy2 = min(x[2], y[2]), min(x[3], y[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        return inter / ((x[2]-x[0])*(x[3]-x[1]) + (y[2]-y[0])*(y[3]-y[1]) - inter)

    caps = {ang: cv2.VideoCapture(str(REPO / f"data/clips/{a.game}_{ang}_{a.tag}.mp4")) for ang in ANGLES}
    cache = {}

    def read_frame(ang, cf):
        if (ang, cf) not in cache:
            caps[ang].set(cv2.CAP_PROP_POS_FRAMES, cf)
            ok, img = caps[ang].read()
            cache[(ang, cf)] = img if ok else None
            if len(cache) > 30:
                cache.pop(next(iter(cache)))
        return cache[(ang, cf)]

    # pass 1: shade per dual-number event
    shades = []                                   # (idx, num, z-shade later)
    raw = []
    order = sorted(range(len(doc["anchors"])),
                   key=lambda i: (doc["anchors"][i]["cam"], doc["anchors"][i]["frame"]))
    for i in order:
        ev = doc["anchors"][i]
        if int(ev["number"]) not in dual:
            continue
        ang = ev["cam"]
        cf = ev["frame"] + offs[ang]
        img = read_frame(ang, cf)
        if img is None:
            continue
        best, bd = 0.0, None
        for di, b in dets[ang].get(cf, []):
            v = iou(b, ev["box"])
            if v > best:
                best, bd = v, (di, b)
        if bd is None or best < 0.5:
            continue
        di, b = bd
        kp = pose[ang].get((cf, di))
        if kp is None:
            continue
        ih, iw = img.shape[:2]
        x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
        x2, y2 = min(iw, int(b[2])), min(ih, int(b[3]))
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        k, ks = kp
        sh = jersey_shade(crop, k - [x1, y1], ks)
        if sh is not None:
            raw.append((i, int(ev["number"]), ang, sh))
    for c in caps.values():
        c.release()

    by_cam = defaultdict(list)
    for _, _, ang, sh in raw:
        by_cam[ang].append(sh)
    stats = {ang: (float(np.mean(v)), float(np.std(v)) + 1e-6) for ang, v in by_cam.items()}
    zrows = [(i, num, (sh - stats[ang][0]) / stats[ang][1]) for i, num, ang, sh in raw]

    n_tagged = 0
    for num in sorted(dual):
        zs = np.sort(np.array([z for _, n, z in zrows if n == num]))
        if len(zs) < 60:
            print(f"#{num}: too few shaded reads ({len(zs)}) — left untagged")
            continue
        c0, c1 = zs.min(), zs.max()
        for _ in range(30):
            assign = np.abs(zs - c0) < np.abs(zs - c1)
            c0, c1 = zs[assign].mean(), zs[~assign].mean()
        lo, hi = min(c0, c1), max(c0, c1)
        if not (lo < -0.3 and hi > 0.2):
            print(f"#{num}: clusters not kit-opposed ({lo:.2f},{hi:.2f}) — left untagged")
            continue
        thr = float((lo + hi) / 2)
        for i, n, z in zrows:
            if n == num:
                doc["anchors"][i]["kit"] = "W" if z >= thr else "B"
                n_tagged += 1
        print(f"#{num}: split at z={thr:.2f}, tagged")
    ap_path.write_text(json.dumps(doc))
    print(f"tagged {n_tagged} events -> {ap_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
