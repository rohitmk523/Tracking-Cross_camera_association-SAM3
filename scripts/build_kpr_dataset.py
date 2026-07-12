#!/usr/bin/env python3
"""Export a KPR fine-tuning dataset from jersey-confirmed crops (no human labels).

Source: c2a354fe_300_60 (training game) — keeps the e6 evaluation uncontaminated.
Identity = jersey number x kit shade (both teams wear 1/2/3/4/5/7; the two #3s are
different people, so each number splits into B/W identities by jersey brightness).
Every crop ships with its RTMPose keypoints (positive prompt) and other-person
keypoints inside the crop (negative prompts) — the training signal KPR expects.

Output: data/kpr_finetune/{images,keypoints,negatives}/{pid}/{pid}_c{cam}_f{frame}.{jpg,npy}
        + manifest.json (per-identity counts, split hints)

  python scripts/build_kpr_dataset.py --game c2a354fe --tag 300_60
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
CAMID = {a: i for i, a in enumerate(ANGLES)}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def jersey_shade(crop, kpts, kscores):
    """Median V (brightness) of the torso region (shoulders->hips) — separates B/W kits."""
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
    ap.add_argument("--game", default="c2a354fe")
    ap.add_argument("--tag", default="300_60")
    ap.add_argument("--min-conf", type=float, default=0.9)
    ap.add_argument("--min-h", type=int, default=90)
    ap.add_argument("--max-per-id-cam", type=int, default=120)
    ap.add_argument("--out", default="data/kpr_finetune")
    ap.add_argument("--pid-prefix", default="", help="namespace identities across games (e.g. c2a_)")
    ap.add_argument("--append", action="store_true", help="add to an existing --out (multi-game)")
    ap.add_argument("--dual-numbers", default="1,3,5",
                    help="numbers worn by BOTH teams (from roster) — only these split into B/W")
    a = ap.parse_args()
    key = f"{a.game}_{a.tag}"
    adoc = json.loads((REPO / f"runs/anchors/{key}.jersey_anchors.json").read_text())
    offs = adoc["offsets"]

    dets, pose = {}, {}
    for ang in ANGLES:
        z = np.load(REPO / f"runs/dets_cache/{a.game}_{ang}_{a.tag}_small_1280_t0.25.dets.npz")
        m = defaultdict(list)
        for di, (b, c, f) in enumerate(zip(z["boxes"], z["classes"], z["frame_idx"])):
            m[int(f)].append((di, [float(v) for v in b], int(c)))
        dets[ang] = m
        p = np.load(REPO / f"runs/pose_cache/{a.game}_{ang}_{a.tag}.pose.npz")
        pose[ang] = {(int(f), int(d)): (k, s) for f, d, k, s in
                     zip(p["frame_idx"], p["det_idx"], p["kpts"], p["kscores"])}

    # 1. confident reads -> (num, ang, cf, di, box)
    confirmed = []
    for ev in adoc["anchors"]:
        if ev.get("conf", 0) < a.min_conf:
            continue
        ang = ev["cam"]
        cf = ev["frame"] + offs[ang]
        for di, b, c in dets[ang].get(cf, []):
            if c == 0 and iou(b, ev["box"]) >= 0.6 and b[3] - b[1] >= a.min_h:
                confirmed.append((int(ev["number"]), ang, cf, di, b))
                break
    print(f"confident-read crops: {len(confirmed)}")

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

    # 2. per number: split into two kit-shade identities (global 2-means on shade)
    rows = []                                     # (num, shade, ang, cf, di, box)
    for num, ang, cf, di, b in sorted(confirmed, key=lambda r: (r[1], r[2])):
        img = read_frame(ang, cf)
        if img is None:
            continue
        ih, iw = img.shape[:2]
        x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
        x2, y2 = min(iw, int(b[2])), min(ih, int(b[3]))
        crop = img[y1:y2, x1:x2]
        kp = pose[ang].get((cf, di))
        if crop.size == 0 or kp is None:
            continue
        k, ks = kp
        shade = jersey_shade(crop, k - [x1, y1], ks)
        if shade is None:
            continue
        rows.append((num, shade, ang, cf, di, b))
    dual = {int(x) for x in a.dual_numbers.split(",") if x}
    # per-camera exposure normalization, then a gap split ONLY for roster-dual numbers
    by_cam = defaultdict(list)
    for r in rows:
        by_cam[r[2]].append(r[1])
    cam_stats = {ang: (float(np.mean(v)), float(np.std(v)) + 1e-6) for ang, v in by_cam.items()}
    zrows = [(num, (sh - cam_stats[ang][0]) / cam_stats[ang][1], ang, cf, di, b)
             for num, sh, ang, cf, di, b in rows]
    split_thr = {}
    for num in sorted({r[0] for r in zrows}):
        if num not in dual:
            continue
        zs = np.sort(np.array([r[1] for r in zrows if r[0] == num]))
        if len(zs) < 100:
            continue
        c0, c1 = zs.min(), zs.max()                # 1-D 2-means
        for _ in range(30):
            assign = np.abs(zs - c0) < np.abs(zs - c1)
            c0, c1 = zs[assign].mean(), zs[~assign].mean()
        lo, hi = min(c0, c1), max(c0, c1)
        # genuine B/W split = kit-opposed clusters (dark AND bright vs camera baseline);
        # both-dark or both-bright = one player under varying light (the #5 trap)
        if lo < -0.3 and hi > 0.2:
            split_thr[num] = float((lo + hi) / 2)
        else:
            print(f"#{num}: clusters not kit-opposed ({lo:.2f},{hi:.2f}) — keeping single identity")
    print(f"dual-number split thresholds (z-shade): {split_thr}")
    rows = zrows

    # 3. write crops + keypoints, capped per (identity, camera)
    outd = REPO / a.out
    for sub in ("images", "keypoints", "negatives"):
        (outd / sub).mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    kept = defaultdict(int)
    rng = np.random.RandomState(0)
    rng.shuffle(rows)
    for num, shade, ang, cf, di, b in rows:
        if num in split_thr:
            pid = f"{a.pid_prefix}{num}{'W' if shade >= split_thr[num] else 'B'}"
        else:
            # single-team number, or dual whose reads come ~entirely from one kit
            # (kit-opposed test failed => one player dominates; minor label noise accepted)
            pid = f"{a.pid_prefix}{num}"
        if kept[(pid, ang)] >= a.max_per_id_cam:
            continue
        img = read_frame(ang, cf)
        if img is None:
            continue
        ih, iw = img.shape[:2]
        x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
        x2, y2 = min(iw, int(b[2])), min(ih, int(b[3]))
        crop = img[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        k, ks = pose[ang][(cf, di)]

        def to_crop(kk, kks):
            kc = kk - [x1, y1]
            inside = ((kc[:, 0] >= 0) & (kc[:, 0] <= cw - 1)
                      & (kc[:, 1] >= 0) & (kc[:, 1] <= ch - 1))
            kc[:, 0] = np.clip(kc[:, 0], 0, cw - 1)
            kc[:, 1] = np.clip(kc[:, 1], 0, ch - 1)
            return np.concatenate([kc, (kks * inside)[:, None]], axis=1), inside

        kxyc, _ = to_crop(k, ks)
        negs = []
        for dj, b2, c2 in dets[ang].get(cf, []):
            if dj == di or c2 != 0:
                continue
            kp2 = pose[ang].get((cf, dj))
            if kp2 is None:
                continue
            nxyc, inside = to_crop(*kp2)
            if inside.sum() >= 3:
                negs.append(nxyc)
        stem = f"{pid}_c{CAMID[ang]}_f{cf}_d{di}"
        for sub in ("images", "keypoints", "negatives"):
            (outd / sub / pid).mkdir(exist_ok=True)
        cv2.imwrite(str(outd / "images" / pid / f"{stem}.jpg"), crop)
        np.save(outd / "keypoints" / pid / f"{stem}.npy", kxyc.astype(np.float32))
        np.save(outd / "negatives" / pid / f"{stem}.npy",
                (np.stack(negs) if negs else np.zeros((0, 17, 3))).astype(np.float32))
        kept[(pid, ang)] += 1
        counts[pid] += 1
    for c in caps.values():
        c.release()
    mpath = outd / "manifest.json"
    games = []
    if a.append and mpath.exists():
        old = json.loads(mpath.read_text())
        games = old.get("games", [])
        for k, v in old.get("identities", {}).items():
            counts[k] = counts.get(k, 0) + v if k in counts else v
    games.append({"game": a.game, "tag": a.tag, "prefix": a.pid_prefix, "split": split_thr})
    manifest = {"games": games, "identities": dict(sorted(counts.items())),
                "total": int(sum(counts.values()))}
    mpath.write_text(json.dumps(manifest, indent=1))
    print(f"dataset: {manifest['total']} crops, {len(counts)} identities -> {outd}")
    print("per identity:", dict(sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
