#!/usr/bin/env python3
"""KPR-assisted cross-camera correction: appearance breaks pile-up ties.

The xcam correction picks the detection NEAREST the interpolated truth. In pile-ups
several bodies sit inside the gate and nearest-wins grabs teammates (the 78->88 gap).
Here, whenever >=2 candidates are in the gate, KPR part-based appearance (prompted
with RTMPose keypoints) votes: score = kpr_distance + w_d * court_dist / gate.

Production-real: galleries are built from jersey-CONFIRMED crops (confident read
attached to a detection), NOT from GT. GT only grades the output. Memory-bounded:
batch-8 inference, LRU frame cache (30), embeddings streamed to disk first.

Run with the KPR env:  /tmp/kpr/.venv310/bin/python scripts/kpr_assist_xcam.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
KPR = Path("/tmp/kpr")
sys.path.insert(0, str(KPR))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

ANGLES = ("FL", "FR", "NL", "NR")
OFFS = {"FL": 0, "FR": -11, "NL": -1, "NR": -1}
ZONE = {"FL": 0.6, "FR": 0.6, "NL": 1.0, "NR": 1.0}
GAME, TAG = "e6fba750", "44_60"
KEY = f"{GAME}_{TAG}"
PLAYERS = {"#11": 11, "#22": 22, "#43": 43, "#6": 6}
# kit-aware override, e.g. PLAYERS_JSON='{"#11":11,"#22":22,"#43":43,"#6B":6}' —
# a trailing B/W on the name restricts that player's anchors to that kit shade
import json as _json
import os as _os
if _os.environ.get("PLAYERS_JSON"):
    PLAYERS = _json.loads(_os.environ["PLAYERS_JSON"])
KIT_OF = {pl: (pl[-1] if pl[-1] in ("B", "W") else None) for pl in PLAYERS}


def kit_ok(pl, akit):
    k = KIT_OF[pl]
    return k is None or akit in (None, k)
import os as _os
SAM3_DIR = _os.environ.get("SAM3_DIR", "runs/hybrid_e6")                 # SAM3-free track streams
GATE_CM = float(_os.environ.get("GATE_CM", 180.0))
MAX_INTERP = int(_os.environ.get("MAX_INTERP", 150))
SOFT_GATE = float(_os.environ.get("SOFT_GATE", 0))   # fallback: nearest within this if none in gate
MIN_H = 90
GALLERY_N = 12
W_D = 0.3                                   # court-distance weight in the tie-break
IOU_HIT = 0.3
EMB_CACHE = REPO / f"runs/pose_cache/{KEY}.kpr_embs{_os.environ.get('KPR_TAG', '')}.npz"


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def main() -> int:
    from kpr_pilot import restricted_torch_load
    import os
    import torch

    # ---------- data ----------
    dets, pose_by = {}, {}
    for ang in ANGLES:
        z = np.load(REPO / f"runs/dets_cache/{GAME}_{ang}_{TAG}_small_1280_t0.25.dets.npz")
        m = defaultdict(list)
        for di, (b, c, f) in enumerate(zip(z["boxes"], z["classes"], z["frame_idx"])):
            if int(c) in (0, 1):
                m[int(f)].append((di, [float(v) for v in b]))
        dets[ang] = m
        p = np.load(REPO / f"runs/pose_cache/{GAME}_{ang}_{TAG}.pose.npz")
        pose_by[ang] = {(int(f), int(d)): (k, s) for f, d, k, s in
                        zip(p["frame_idx"], p["det_idx"], p["kpts"], p["kscores"])}
    # GT det-index references were made against the ORIGINAL detector's lists; when
    # the pipeline runs a different detector (Level-2 gate), GT_DETS_DIR points at
    # the original caches so grading stays valid.
    _gtd = _os.environ.get("GT_DETS_DIR")
    gt_dets = None
    if _gtd:
        gt_dets = {}
        for ang in ANGLES:
            z = np.load(REPO / _gtd / f"{GAME}_{ang}_{TAG}_small_1280_t0.25.dets.npz")
            m = defaultdict(list)
            for b, c, f in zip(z["boxes"], z["classes"], z["frame_idx"]):
                if int(c) in (0, 1):
                    m[int(f)].append([float(v) for v in b])
            gt_dets[ang] = m

    anchors = defaultdict(list)
    for ev in json.loads((REPO / f"runs/anchors/{KEY}.jersey_anchors.json").read_text())["anchors"]:
        anchors[(ev["cam"], ev["frame"])].append((ev["box"], int(ev["number"]), ev.get("conf", 1.0), ev.get("kit")))
    gt = json.loads((REPO / f"data/gt_players/{KEY}.json").read_text())

    from uball_cc.fusion.homography import load_calib, project_pixels
    calib = {ang: load_calib(str(REPO / f"configs/calib/{ang}.json")) for ang in ANGLES}

    def court(ang, box):
        (x, y), = project_pixels([((box[0] + box[2]) / 2, box[3])], calib[ang])
        return np.array([x, y])

    # ---------- per-player track streams + truth (same recipe as solve_player_xcam) ----------
    sam_by_pl, truth_by_pl, frames_by_pl = {}, {}, {}
    for pl, num in PLAYERS.items():
        safe = pl.replace("#", "n")
        sam = {}
        for ang in ANGLES:
            p = REPO / SAM3_DIR / f"{KEY}__{safe}__{ang}.json"
            if p.exists():
                sam[ang] = {int(f): r for f, r in json.loads(p.read_text())["frames"].items()
                            if r.get("present") and r.get("box")}
        sam_by_pl[pl] = sam
        gpl = gt.get(pl) or gt[pl[:-1] if pl[-1] in ("B", "W") else pl]
        sel = {int(f): s for f, s in gpl["frames"].items() if s and str(f) in gpl.get("approved", {})}
        frames = sorted(sel)
        frames_by_pl[pl] = (frames, sel)
        anchor_pos = {}
        for f in frames:
            pts, ws = [], []
            for ang in ANGLES:
                sr = sam.get(ang, {}).get(f + OFFS[ang])
                if not sr:
                    continue
                for abox, anum, aconf, akit in anchors.get((ang, f), []):
                    if anum == num and iou(sr["box"], abox) >= 0.3 and kit_ok(pl, akit):
                        pts.append(court(ang, sr["box"])); ws.append(ZONE[ang] * aconf)
                        break
            if pts:
                anchor_pos[f] = np.average(pts, axis=0, weights=ws)
        tr = dict(anchor_pos)
        ak = sorted(anchor_pos)
        for a0, a1 in zip(ak, ak[1:]):
            g = a1 - a0
            if 1 < g <= MAX_INTERP:
                for f in range(a0 + 1, a1):
                    tr[f] = anchor_pos[a0] + (anchor_pos[a1] - anchor_pos[a0]) * ((f - a0) / g)
        truth_by_pl[pl] = tr

    # ---------- collect crops needing embeddings ----------
    # galleries: jersey-CONFIRMED detections (production identity, no GT)
    gal_specs = {pl: [] for pl in PLAYERS}          # (ang, cf, di, box)
    for (ang, f), evs in anchors.items():
        for abox, anum, aconf, akit in evs:
            pl = next((p for p, n in PLAYERS.items()
                       if n == anum and kit_ok(p, akit)), None)
            if pl is None or aconf < 0.95:
                continue
            cf = f + OFFS[ang]
            for di, b in dets[ang].get(cf, []):
                if iou(b, abox) >= 0.6 and b[3] - b[1] >= MIN_H:
                    gal_specs[pl].append((ang, cf, di, tuple(b)))
                    break
    rng = np.random.RandomState(0)
    for pl in gal_specs:
        gs = gal_specs[pl]
        gal_specs[pl] = [gs[i] for i in rng.choice(len(gs), min(GALLERY_N, len(gs)), replace=False)]

    # ---------- embeddings (streamed, batch 8, LRU frames) ----------
    if EMB_CACHE.exists():
        zc = np.load(EMB_CACHE, allow_pickle=True)
        emb_map = {tuple(k): (e, v) for k, e, v in zip(zc["keys"], zc["embs"], zc["vis"])}
    else:
        emb_map = {}

    _ext = {}

    def embed_missing(need_map):
        missing = [k for k in need_map if k not in emb_map]
        print(f"embeddings: {len(emb_map)} cached, {len(missing)} to compute", flush=True)
        if missing:
            _embed(missing, need_map)

    def _embed(missing, need_map):
        if "x" not in _ext:
            _orig = torch.load
            torch.load = restricted_torch_load()
            from torchreid.scripts.builder import build_config
            from torchreid.tools.feature_extractor import KPRFeatureExtractor
            cwd = os.getcwd()
            os.chdir(KPR)
            try:
                cfg = build_config(config_path=str(KPR / __import__("os").environ.get("KPR_CFG", "configs/kpr/imagenet/kpr_occ_posetrack_test.yaml")))
                cfg.use_gpu = False
                cfg.test.batch_size = 8
                _ext["x"] = KPRFeatureExtractor(cfg)
            finally:
                os.chdir(cwd)
                torch.load = _orig
        extractor = _ext["x"]

        caps = {ang: cv2.VideoCapture(str(REPO / f"data/clips/{GAME}_{ang}_{TAG}.mp4")) for ang in ANGLES}
        cache = {}

        def read_frame(ang, cf):
            if (ang, cf) not in cache:
                caps[ang].set(cv2.CAP_PROP_POS_FRAMES, cf)
                ok, img = caps[ang].read()
                cache[(ang, cf)] = img if ok else None
                if len(cache) > 30:
                    cache.pop(next(iter(cache)))
            return cache[(ang, cf)]

        def build_sample(ang, cf, di, box):
            img = read_frame(ang, cf)
            if img is None:
                return None
            ih, iw = img.shape[:2]
            x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
            x2, y2 = min(iw, int(box[2])), min(ih, int(box[3]))
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            ch, cw = crop.shape[:2]

            def to_crop(k, ks):
                kc = k - [x1, y1]
                inside = ((kc[:, 0] >= 0) & (kc[:, 0] <= cw - 1)
                          & (kc[:, 1] >= 0) & (kc[:, 1] <= ch - 1))
                kc[:, 0] = np.clip(kc[:, 0], 0, cw - 1)
                kc[:, 1] = np.clip(kc[:, 1], 0, ch - 1)
                return np.concatenate([kc, (ks * inside)[:, None]], axis=1), inside

            smp = {"image": crop}
            kp = pose_by[ang].get((cf, di))
            if kp is not None:
                kxyc, _ = to_crop(*kp)
                smp["keypoints_xyc"] = kxyc
            negs = []
            for dj, b2 in dets[ang].get(cf, []):
                if dj == di:
                    continue
                kp2 = pose_by[ang].get((cf, dj))
                if kp2 is None:
                    continue
                nxyc, inside = to_crop(*kp2)
                if inside.sum() >= 3:
                    negs.append(nxyc)
            smp["negative_kps"] = np.array(negs) if negs else np.zeros((0, 17, 3))
            return smp

        # sort by (ang, cf) for sequential-ish decode; embed in chunks of 64 samples
        missing.sort()
        CH = 64
        for s0 in range(0, len(missing), CH):
            chunk = missing[s0:s0 + CH]
            samples, keys = [], []
            for k in chunk:
                smp = build_sample(k[0], k[1], k[2], need_map[k])
                if smp is not None:
                    samples.append(smp); keys.append(k)
            if not samples:
                continue
            import torch as _t
            with _t.no_grad():
                _, embs, vis, _ = extractor(samples)
            embs = embs.detach().cpu().numpy() if hasattr(embs, "cpu") else np.asarray(embs)
            vis = vis.detach().cpu().numpy() if hasattr(vis, "cpu") else np.asarray(vis)
            for k, e, v in zip(keys, embs, vis):
                emb_map[k] = (e, v)
            print(f"  embedded {min(s0+CH, len(missing))}/{len(missing)}", flush=True)
        for c in caps.values():
            c.release()
        ks = list(emb_map)
        np.savez_compressed(EMB_CACHE, keys=np.array(ks),
                            embs=np.stack([emb_map[k][0] for k in ks]),
                            vis=np.stack([emb_map[k][1] for k in ks]))
        print(f"embeddings cached -> {EMB_CACHE}")

    # ---------- KPR distance helper ----------
    from torchreid.metrics.distance import compute_distance_matrix_using_bp_features
    import torch as T

    def kpr_dist(q_key, gal_keys):
        if q_key not in emb_map:
            return None
        gk = [k for k in gal_keys if k in emb_map]
        if not gk:
            return None
        qe = T.tensor(np.stack([emb_map[q_key][0]]))
        qv = T.tensor(np.stack([emb_map[q_key][1]]))
        ge = T.tensor(np.stack([emb_map[k][0] for k in gk]))
        gv = T.tensor(np.stack([emb_map[k][1] for k in gk]))
        D, _ = compute_distance_matrix_using_bp_features(qe, ge, qv, gv,
                                                         use_gpu=False, use_logger=False)
        d = np.sort(D.cpu().numpy()[0])
        return float(d[:3].mean())


    # ---- gallery embeddings + threshold calibration (leave-one-out) ----
    gal_need = {}
    for pl, specs in gal_specs.items():
        for ang, cf, di, box in specs:
            gal_need[(ang, cf, di)] = list(box)
    embed_missing(gal_need)

    from torchreid.metrics.distance import compute_distance_matrix_using_bp_features as _cdm
    import torch as _T

    def kdist_many(q_keys, g_keys):
        qk = [k for k in q_keys if k in emb_map]
        gk = [k for k in g_keys if k in emb_map]
        if not qk or not gk:
            return None, None, None
        qe = _T.tensor(np.stack([emb_map[k][0] for k in qk]))
        qv = _T.tensor(np.stack([emb_map[k][1] for k in qk]))
        ge = _T.tensor(np.stack([emb_map[k][0] for k in gk]))
        gv = _T.tensor(np.stack([emb_map[k][1] for k in gk]))
        D, _ = _cdm(qe, ge, qv, gv, use_gpu=False, use_logger=False)
        return D.cpu().numpy(), qk, gk

    gal_keys_all = {pl: [(a, c, d) for a, c, d, _ in gal_specs[pl]] for pl in PLAYERS}
    same_d, diff_d = [], []
    for pl in PLAYERS:
        D, qk, gk = kdist_many(gal_keys_all[pl], gal_keys_all[pl])
        if D is not None and len(qk) > 3:
            for i in range(len(qk)):
                row = np.delete(D[i], i)
                same_d.append(np.sort(row)[:3].mean())
        for pl2 in PLAYERS:
            if pl2 == pl:
                continue
            D, qk, gk = kdist_many(gal_keys_all[pl], gal_keys_all[pl2])
            if D is not None:
                for i in range(len(qk)):
                    diff_d.append(np.sort(D[i])[:3].mean())
    same_d, diff_d = np.array(same_d), np.array(diff_d)
    REACQ_TH = float(np.percentile(same_d, 75))
    print(f"KPR threshold calib: same-id median {np.median(same_d):.3f}, "
          f"diff-id median {np.median(diff_d):.3f} -> reacq threshold {REACQ_TH:.3f}")

    # ---- appearance-based RE-ACQUISITION over long unanchored stretches ----
    REACQ_GAP = int(_os.environ.get("REACQ_GAP", 240))    # >8s without truth
    REACQ_STEP = 15
    scan_need, scan_plan = {}, defaultdict(list)          # pl -> [(f, ang, cf, di, box)]
    for pl in PLAYERS:
        frames, sel = frames_by_pl[pl]
        tr = truth_by_pl[pl]
        have = sorted(tr)
        gaps = []
        if have:
            if have[0] - frames[0] > REACQ_GAP:
                gaps.append((frames[0], have[0]))
            for a, b in zip(have, have[1:]):
                if b - a > REACQ_GAP:
                    gaps.append((a, b))
            if frames[-1] - have[-1] > REACQ_GAP:
                gaps.append((have[-1], frames[-1]))
        for g0, g1 in gaps:
            for f in range(g0 + REACQ_STEP, g1, REACQ_STEP):
                if f not in sel:
                    continue
                for ang in ANGLES:
                    cf = f + OFFS[ang]
                    for di, b in dets[ang].get(cf, []):
                        if b[3] - b[1] >= MIN_H:
                            scan_need[(ang, cf, di)] = b
                            scan_plan[pl].append((f, ang, cf, di, b))
    print(f"re-acquisition scan: {sum(len(v) for v in scan_plan.values())} crop-checks "
          f"({len(scan_need)} unique crops)")
    embed_missing(scan_need)

    n_reacq = 0
    for pl, checks in scan_plan.items():
        byf = defaultdict(list)
        for f, ang, cf, di, b in checks:
            byf[f].append((ang, cf, di, b))
        for f, items in sorted(byf.items()):
            best = None                                   # (score, margin, ang, box)
            for ang, cf, di, b in items:
                q = (ang, cf, di)
                d_own = kdist_many([q], gal_keys_all[pl])
                if d_own[0] is None:
                    continue
                own = np.sort(d_own[0][0])[:3].mean()
                others = []
                for pl2 in PLAYERS:
                    if pl2 == pl:
                        continue
                    d_o = kdist_many([q], gal_keys_all[pl2])
                    if d_o[0] is not None:
                        others.append(np.sort(d_o[0][0])[:3].mean())
                margin = (min(others) - own) if others else 0.0
                if own < REACQ_TH and margin > 0.02:
                    if best is None or own < best[0]:
                        best = (own, margin, ang, b)
            if best is not None:
                truth_by_pl[pl][f] = court(best[2], best[3])
                n_reacq += 1
        # re-interpolate including provisional anchors
        tr = truth_by_pl[pl]
        ak = sorted(tr)
        for a0, a1 in zip(ak, ak[1:]):
            g = a1 - a0
            if 1 < g <= MAX_INTERP:
                for f in range(a0 + 1, a1):
                    if f not in tr:
                        tr[f] = tr[a0] + (tr[a1] - tr[a0]) * ((f - a0) / g)
    print(f"re-acquisition: {n_reacq} provisional anchors placed")

    # ambiguous instances: >=2 candidates in gate at a truth frame
    amb = []                                        # (pl, ang, f, [(di, box, dist_cm), ...])
    for pl in PLAYERS:
        frames, sel = frames_by_pl[pl]
        tr = truth_by_pl[pl]
        for f in frames:
            tp = tr.get(f)
            if tp is None:
                continue
            is_anchor_f = False
            for ang in ANGLES:
                cf = f + OFFS[ang]
                sr = sam_by_pl[pl].get(ang, {}).get(cf)
                if sr and any(anum == PLAYERS[pl] and iou(sr["box"], abox) >= 0.3
                              and kit_ok(pl, akit)
                              for abox, anum, _, akit in anchors.get((ang, f), [])):
                    is_anchor_f = True
            for ang in ANGLES:
                cf = f + OFFS[ang]
                cands = []
                for di, b in dets[ang].get(cf, []):
                    d = float(np.linalg.norm(court(ang, b) - tp))
                    if d <= GATE_CM and b[3] - b[1] >= MIN_H:
                        cands.append((di, b, d))
                if len(cands) >= 2:
                    ds = sorted(c[2] for c in cands)
                    if ds[1] - ds[0] < 60.0:       # genuine tie only: runner-up within 60cm
                        amb.append((pl, ang, f, cands))
    n_crops = len({(ang, f + OFFS[ang], di) for pl, ang, f, cs in amb for di, _, _ in cs})
    print(f"galleries: { {p: len(g) for p, g in gal_specs.items()} } | "
          f"ambiguous instances: {len(amb)} | unique candidate crops: {n_crops}")
    amb_need = {}
    for pl, ang, f, cands in amb:
        for di, b, _ in cands:
            amb_need[(ang, f + OFFS[ang], di)] = b
    embed_missing(amb_need)

    gal_keys = {pl: [(a, c, d) for a, c, d, _ in gal_specs[pl]] for pl in PLAYERS}
    amb_lookup = {(pl, ang, f): cands for pl, ang, f, cands in amb}

    # ---------- corrected choice + scoring, baseline vs KPR-assisted ----------
    report = {}
    for pl, num in PLAYERS.items():
        frames, sel = frames_by_pl[pl]
        tr = truth_by_pl[pl]
        sam = sam_by_pl[pl]
        chosen = {"base": {}, "kpr": {}, "kprseg": {}}
        n_ties = n_flipped = 0
        seg_cand = defaultdict(list)   # (ang) -> [(f, [(di,b,d,kd), ...])] for segment pass
        for f in frames:
            tp = tr.get(f)
            for ang in ANGLES:
                cf = f + OFFS[ang]
                sr = sam.get(ang, {}).get(cf)
                is_anchor = sr and any(anum == num and iou(sr["box"], abox) >= 0.3
                                       and kit_ok(pl, akit)
                                       for abox, anum, _, akit in anchors.get((ang, f), []))
                if is_anchor:
                    for m in chosen:
                        chosen[m][(ang, f)] = sr["box"]
                    continue
                if tp is None:
                    if sr:
                        for m in chosen:
                            chosen[m][(ang, f)] = sr["box"]
                    continue
                cands_all = [(di, b, float(np.linalg.norm(court(ang, b) - tp)))
                             for di, b in dets[ang].get(cf, [])]
                in_gate = [(di, b, d) for di, b, d in cands_all if d <= GATE_CM]
                if not in_gate:
                    if sr:
                        for m in chosen:
                            chosen[m][(ang, f)] = sr["box"]
                    elif SOFT_GATE > 0 and cands_all:
                        di_s, b_s, d_s = min(cands_all, key=lambda x: x[2])
                        if d_s <= SOFT_GATE:
                            for m in chosen:
                                chosen[m][(ang, f)] = b_s
                    continue
                nearest = min(in_gate, key=lambda x: x[2])
                chosen["base"][(ang, f)] = nearest[1]
                pick = nearest
                key_c = (pl, ang, f)
                if key_c in amb_lookup and len(in_gate) >= 2:
                    n_ties += 1
                    scored = []
                    for di, b, d in in_gate:
                        kd = kpr_dist((ang, cf, di), gal_keys[pl])
                        if kd is not None:
                            scored.append((kd + W_D * d / GATE_CM, di, b, d))
                    if scored:
                        best = min(scored, key=lambda x: x[0])
                        if best[1] != nearest[0]:
                            n_flipped += 1
                        pick = (best[1], best[2], best[3])
                        seg_cand[ang].append((f, [(di2, b2, d2, sc - W_D * d2 / GATE_CM)
                                                  for sc, di2, b2, d2 in scored]))
                chosen["kpr"][(ang, f)] = pick[1]
                chosen["kprseg"][(ang, f)] = pick[1]   # overwritten below where a segment decides

        # ---- segment-level voting: one decision per contiguous tie run ----
        n_segs = 0
        for ang, entries in seg_cand.items():
            entries.sort()
            groups, cur = [], [entries[0]]
            for e in entries[1:]:
                if e[0] - cur[-1][0] <= 5:
                    cur.append(e)
                else:
                    groups.append(cur); cur = [e]
            groups.append(cur)
            for grp in groups:
                if len(grp) < 3:
                    continue                      # too short to aggregate; keep frame picks
                n_segs += 1
                chains = []                        # {"last": box, "frames": {f: box}, "kds": [], "ds": []}
                for f, cands in grp:
                    for di, b, d, kd in cands:
                        best_c, best_v = None, 0.4
                        for c in chains:
                            v = iou(c["last"], b)
                            if v > best_v:
                                best_c, best_v = c, v
                        if best_c is None:
                            chains.append({"last": b, "frames": {f: b}, "kds": [kd], "ds": [d]})
                        else:
                            best_c["last"] = b
                            best_c["frames"][f] = b
                            best_c["kds"].append(kd); best_c["ds"].append(d)
                cov = max(len(c["frames"]) for c in chains)
                elig = [c for c in chains if len(c["frames"]) >= max(3, 0.3 * len(grp))]
                if not elig:
                    continue
                win = min(elig, key=lambda c: np.mean(c["kds"]) + W_D * np.mean(c["ds"]) / GATE_CM)
                for f, _ in grp:
                    if f in win["frames"]:
                        chosen["kprseg"][(ang, f)] = win["frames"][f]

        # error decomposition for base mode: no-pick vs wrong-pick
        n_none = n_wrong = n_vis = 0
        for f in frames:
            for ang in ANGLES:
                if ang not in sel[f]:
                    continue
                cf = f + OFFS[ang]
                gb = gt_dets[ang].get(cf, []) if gt_dets else [b for _, b in dets[ang].get(cf, [])]
                if sel[f][ang] >= len(gb):
                    continue
                n_vis += 1
                cb = chosen["base"].get((ang, f))
                if cb is None:
                    n_none += 1
                elif iou(gb[sel[f][ang]], cb) < IOU_HIT:
                    n_wrong += 1
        # FUSED coverage (>=1 visible camera correct) — the product metric
        fused_hit = fused_n = 0
        for f in frames:
            vis_any = hit_any = False
            for ang in ANGLES:
                if ang not in sel[f]:
                    continue
                cf = f + OFFS[ang]
                src = gt_dets[ang] if gt_dets else None
                gb = (src.get(cf, []) if src is not None
                      else [b for _, b in dets[ang].get(cf, [])])
                if sel[f][ang] >= len(gb):
                    continue
                vis_any = True
                cb = chosen["kpr"].get((ang, f))
                if cb is not None and iou(gb[sel[f][ang]], cb) >= IOU_HIT:
                    hit_any = True
            if vis_any:
                fused_n += 1
                fused_hit += int(hit_any)
        fused = fused_hit / max(1, fused_n)
        print(f"{pl}: FUSED coverage {fused:.0%} ({fused_hit}/{fused_n})")
        row = {"ties": n_ties, "flipped": n_flipped, "segments": n_segs,
               "err_none": n_none, "err_wrong": n_wrong, "n_vis": n_vis,
               "fused": round(fused, 3)}
        for m in ("base", "kpr", "kprseg"):
            pc = {}
            for ang in ANGLES:
                nv = nh = 0
                for f in frames:
                    if ang not in sel[f]:
                        continue
                    cf = f + OFFS[ang]
                    gb = gt_dets[ang].get(cf, []) if gt_dets else [b for _, b in dets[ang].get(cf, [])]
                    if sel[f][ang] >= len(gb):
                        continue
                    nv += 1
                    cb = chosen[m].get((ang, f))
                    if cb and iou(gb[sel[f][ang]], cb) >= IOU_HIT:
                        nh += 1
                if nv >= 20:
                    pc[ang] = nh / nv
            row[m] = {"per_cam": {k: round(v, 3) for k, v in pc.items()},
                      "all_angles": round(sum(pc.values()) / len(pc), 3) if pc else None}
        report[pl] = row
        dump = _os.environ.get("DUMP_DIR")
        if dump:
            dd = REPO / dump
            dd.mkdir(parents=True, exist_ok=True)
            safe = pl.replace("#", "n")
            for ang in ANGLES:
                fj = {}
                for f in frames:
                    cb = chosen["kpr"].get((ang, f))
                    if cb:
                        fj[str(f + OFFS[ang])] = {"box": [round(x, 1) for x in cb],
                                                  "present": True, "score": 1.0}
                (dd / f"{KEY}__{safe}__{ang}.json").write_text(
                    json.dumps({"player": pl, "cam": ang, "frames": fj}))
        print(f"{pl}: MISSES: no-pick {row['err_none']} vs wrong-pick {row['err_wrong']} of {row['n_vis']} vis")
        print(f"{pl}: ties {n_ties}, flips {n_flipped}, segments {row['segments']} | "
              f"base {row['base']['all_angles']:.0%} -> KPR {row['kpr']['all_angles']:.0%} "
              f"-> KPR-SEG {row['kprseg']['all_angles']:.0%}")

    means = {m: np.mean([r[m]["all_angles"] for r in report.values()])
             for m in ("base", "kpr", "kprseg")}
    print(f"\nMEAN strict all-angles: base {means['base']:.1%} -> frame-KPR {means['kpr']:.1%} "
          f"-> SEGMENT-KPR {means['kprseg']:.1%}   [SAM3+xcam was 88%]")
    fused_mean = float(np.mean([r["fused"] for r in report.values()]))
    print(f"MEAN FUSED (>=1 cam, kpr mode): {fused_mean:.1%}")
    (REPO / f"runs/tracking/ledger/kprxcam_{KEY}.json").write_text(json.dumps(
        {"window": KEY, "method": "hybrid xcam + KPR tie-break (no SAM3)",
         "players": report, "mean": {m: round(float(v), 3) for m, v in means.items()},
         "mean_fused": round(fused_mean, 3)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
