#!/usr/bin/env python3
"""OFFLINE IDENTITY SOLVER v2.2 — micro-tracklets + anchor-interval assignment.

Design settled by measurement: fused tracklets are contaminated (v1 failed), raw
per-frame assignment flaps (v2 failed). v2.2:
  A. MICRO-TRACKLETS: stitch fused observations only across <=2-frame gaps at tight
     court radius — segments are pure by construction.
  B. LABEL: attach dense jersey anchors to micro-tracklets; resolve each NUMBER into
     one or two real players by temporal CO-OCCURRENCE of labeled segments (two #3s
     on court at once = two people; team letters too noisy to be trusted for this).
  C. SIDE-SPLIT co-occurring numbers by appearance clustering of their segments.
  D. ASSIGN: labeled segments form each identity's scaffold (conflict-free by anchor
     weight); unlabeled segments fill gaps by continuity + appearance, greedy over
     global link costs with temporal-overlap vetoes.

  python scripts/solve_identities_mt.py --game e6fba750 --tag 44_60 \
      --worldstate runs/tracking/e6fba750_44_60_worldstate_vSAM3.json --version vMT
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")
FPS = 29.97
OFFS = {"e6fba750_44_60": {"FL": 0, "FR": -11, "NL": -1, "NR": -1},
        "c2a354fe_300_60": {"FL": 0, "FR": -4, "NL": -3, "NR": -4}}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--worldstate", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--stitch-cm", type=float, default=80.0)
    ap.add_argument("--min-num-anchors", type=int, default=15)
    ap.add_argument("--link-max-gap-s", type=float, default=25.0)
    ap.add_argument("--link-speed", type=float, default=600.0)
    a = ap.parse_args()
    from scipy.optimize import linear_sum_assignment

    key = f"{a.game}_{a.tag}"
    offs = OFFS[key]
    ws = json.loads(Path(a.worldstate).read_text())
    teams_rows = {}
    for ang in ANGLES:
        m = defaultdict(dict)
        for r in json.loads((REPO / f"runs/tracking/{a.game}_{ang}_{a.tag}_teams.json").read_text())["tracks"]:
            m[r["frame"]][r["track_id"]] = r
        teams_rows[ang] = m
    reid = {}
    for ang in ANGLES:
        p = REPO / f"runs/tracking/{a.game}_{ang}_{a.tag}_reid.npz"
        if p.exists():
            z = np.load(p)
            for i, v in zip(z["ids"], z["emb"]):
                vv = np.asarray(v, float)
                reid[(ang, int(i))] = vv / (np.linalg.norm(vv) + 1e-8)

    # ---- observations ----
    OBS = {}
    for fr in ws["frames"]:
        f = fr["frame"]
        row = []
        for t in fr["tracks"]:
            if t.get("coasting"):
                continue
            cams = {}
            vecs = []
            for ang, lid in (t.get("members") or {}).items():
                rr = teams_rows[ang].get(f + offs[ang], {}).get(lid)
                if rr is None:
                    continue
                cams[ang] = (lid, rr["box_xyxy"])
                if (ang, lid) in reid:
                    vecs.append(reid[(ang, lid)])
            v = None
            if vecs:
                v = np.mean(vecs, axis=0)
                v = v / (np.linalg.norm(v) + 1e-8)
            row.append({"xy": np.array(t["court_xy"], float), "cams": cams, "reid": v})
        OBS[f] = row

    frames_sorted = sorted(OBS)

    # ---- A. micro-tracklets ----
    mts = []                                  # each: dict(frames=[(f, oi)], ...)
    active = []                               # (mt_index, last_frame)
    for f in frames_sorted:
        row = OBS[f]
        cand_mt = [(mi, lf) for mi, lf in active if f - lf <= 2]
        C = np.full((len(row), len(cand_mt)), 1e6)
        for oi, o in enumerate(row):
            for ci, (mi, lf) in enumerate(cand_mt):
                last_xy = mts[mi]["xy"][-1]
                d = float(np.linalg.norm(o["xy"] - last_xy))
                lim = a.stitch_cm * (f - lf)
                if d <= lim:
                    C[oi, ci] = d
        ri, ci_ = linear_sum_assignment(C)
        matched_obs, matched_mt = set(), {}
        for oi, ci in zip(ri, ci_):
            if C[oi, ci] < 1e5:
                matched_obs.add(oi)
                matched_mt[cand_mt[ci][0]] = oi
        new_active = []
        for mi, lf in active:
            if mi in matched_mt:
                oi = matched_mt[mi]
                o = row[oi]
                mts[mi]["frames"].append((f, oi))
                mts[mi]["xy"].append(o["xy"])
                if o["reid"] is not None:
                    mts[mi]["reid_sum"] += o["reid"]
                    mts[mi]["n_reid"] += 1
                new_active.append((mi, f))
            elif f - lf <= 2:
                new_active.append((mi, lf))
        for oi, o in enumerate(row):
            if oi not in matched_obs:
                mts.append({"frames": [(f, oi)], "xy": [o["xy"]],
                            "reid_sum": (o["reid"].copy() if o["reid"] is not None
                                         else np.zeros(512)),
                            "n_reid": 1 if o["reid"] is not None else 0,
                            "anchor": Counter()})
                new_active.append((len(mts) - 1, f))
        active = new_active
    for mt in mts:
        mt["start"] = mt["frames"][0][0]
        mt["end"] = mt["frames"][-1][0]
        mt["reid"] = None
        if mt["n_reid"]:
            v = mt["reid_sum"] / mt["n_reid"]
            mt["reid"] = v / (np.linalg.norm(v) + 1e-8)

    obs_to_mt = {}
    for mi, mt in enumerate(mts):
        for f, oi in mt["frames"]:
            obs_to_mt[(f, oi)] = mi

    # ---- B. anchors -> micro-tracklets ----
    anchors_raw = json.loads((REPO / f"runs/anchors/{key}.jersey_anchors.json").read_text())
    n_att = 0
    for ev in anchors_raw["anchors"]:
        row = OBS.get(ev["frame"])
        if not row:
            continue
        best, bi = 0.35, None
        for oi, o in enumerate(row):
            got = o["cams"].get(ev["cam"])
            if got is None:
                continue
            v = iou(got[1], ev["box"])
            if v > best:
                best, bi = v, oi
        if bi is None:
            continue
        mi = obs_to_mt.get((ev["frame"], bi))
        if mi is not None:
            mts[mi]["anchor"][int(ev["number"])] += 1
            n_att += 1

    # a segment's number label: dominant, needs >=2 reads and >=70% share
    for mt in mts:
        mt["num"] = None
        if mt["anchor"]:
            num, c = mt["anchor"].most_common(1)[0]
            if c >= 2 and c / sum(mt["anchor"].values()) >= 0.7:
                mt["num"] = num

    # ---- number -> 1 or 2 players by co-occurrence of labeled segments ----
    by_num = defaultdict(list)
    for mi, mt in enumerate(mts):
        if mt["num"] is not None:
            by_num[mt["num"]].append(mi)
    identities = []                            # list of dicts(label, member mts)
    for num, mis in sorted(by_num.items(), key=lambda kv: -len(kv[1])):
        total_anchor = sum(sum(mts[m]["anchor"].values()) for m in mis)
        if total_anchor < a.min_num_anchors:
            for m in mis:
                mts[m]["num"] = None
            continue
        overlaps = 0
        for i in range(len(mis)):
            for j in range(i + 1, len(mis)):
                A, B = mts[mis[i]], mts[mis[j]]
                if not (A["end"] < B["start"] or B["end"] < A["start"]):
                    overlaps += 1
        if overlaps >= 2:                      # same number visible twice concurrently
            vecs = [mts[m]["reid"] for m in mis if mts[m]["reid"] is not None]
            if len(vecs) >= 2:
                X = np.stack(vecs)
                from sklearn.cluster import KMeans
                lab = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
                groups = {0: [], 1: []}
                vi = 0
                for m in mis:
                    if mts[m]["reid"] is not None:
                        groups[int(lab[vi])].append(m)
                        vi += 1
                    else:
                        groups[0].append(m)
                identities.append({"label": f"#{num}a", "mts": set(groups[0])})
                identities.append({"label": f"#{num}b", "mts": set(groups[1])})
            else:
                identities.append({"label": f"#{num}", "mts": set(mis)})
        else:
            identities.append({"label": f"#{num}", "mts": set(mis)})

    # conflict-free scaffold: within an identity, overlapping members keep the
    # anchor-heavier one
    for ident in identities:
        keep = []
        for m in sorted(ident["mts"], key=lambda m: -sum(mts[m]["anchor"].values())):
            if all(mts[m]["end"] < mts[k]["start"] or mts[m]["start"] > mts[k]["end"]
                   for k in keep):
                keep.append(m)
        ident["mts"] = set(keep)
    assigned = {}
    for ii, ident in enumerate(identities):
        for m in ident["mts"]:
            assigned[m] = ii

    # ---- D. fill gaps with unlabeled segments (global greedy, overlap vetoes) ----
    def spans(ident):
        return sorted((mts[m]["start"], mts[m]["end"]) for m in ident["mts"])

    def link_cost(m, ident):
        mt = mts[m]
        if any(not (mt["end"] < s0 or mt["start"] > s1) for s0, s1 in spans(ident)):
            return None
        best = None
        for k in ident["mts"]:
            e = mts[k]
            if mt["start"] >= e["end"]:
                gap = (mt["start"] - e["end"]) / FPS
                d = float(np.linalg.norm(mt["xy"][0] - e["xy"][-1]))
            else:
                gap = (e["start"] - mt["end"]) / FPS
                d = float(np.linalg.norm(e["xy"][0] - mt["xy"][-1]))
            if gap > a.link_max_gap_s or d > max(120.0, a.link_speed * gap):
                continue
            cos = 0.5
            if mt["reid"] is not None and e["reid"] is not None:
                cos = float(np.dot(mt["reid"], e["reid"]))
                if cos < 0.4:
                    continue
            c = gap * 0.12 + d / 900.0 + (1 - cos) * 0.8
            best = c if best is None or c < best else best
        return best

    unl = [m for m in range(len(mts)) if m not in assigned]
    for _ in range(8):
        cands = []
        for m in unl:
            for ii, ident in enumerate(identities):
                c = link_cost(m, ident)
                if c is not None:
                    cands.append((c, m, ii))
        if not cands:
            break
        cands.sort()
        progress = False
        placed = set()
        for c, m, ii in cands:
            if m in placed or m in assigned:
                continue
            if link_cost(m, identities[ii]) is None:
                continue
            identities[ii]["mts"].add(m)
            assigned[m] = ii
            placed.add(m)
            progress = True
        unl = [m for m in unl if m not in assigned]
        if not progress:
            break

    # ---- emit ----
    n_named = len(identities)
    pool = {}
    out_frames = []
    for fr in ws["frames"]:
        f = fr["frame"]
        row = OBS.get(f, [])
        tracks = []
        seen = set()
        oi = 0
        for t in fr["tracks"]:
            if t.get("coasting"):
                continue
            mi = obs_to_mt.get((f, oi))
            oi += 1
            if mi is None:
                continue
            if mi in assigned:
                iid = assigned[mi]
            else:
                iid = pool.setdefault(mi, n_named + len(pool))
            if iid in seen:
                continue
            seen.add(iid)
            t2 = dict(t)
            t2["global_id"] = int(iid)
            tracks.append(t2)
        out_frames.append({"frame": f, "tracks": tracks})
    players = []
    for ii, ident in enumerate(identities):
        digits = "".join(ch for ch in ident["label"] if ch.isdigit())
        players.append({"global_id": ii, "team": None,
                        "jersey": int(digits) if digits else None, "court_xy": None})
    solved = {"n_global_ids": n_named + len(pool), "players": players,
              "frames": out_frames, "ref_angle": ws.get("ref_angle"),
              "angles": ws.get("angles"), "fps": FPS,
              "solver_mt": {"micro_tracklets": len(mts), "anchors_attached": n_att,
                            "identities": [i["label"] for i in identities],
                            "labeled_mts": len(assigned), "pool_mts": len(pool)}}
    out = Path(a.worldstate).parent / f"{a.game}_{a.tag}_worldstate_{a.version}.json"
    out.write_text(json.dumps(solved))
    print(f"mts {len(mts)} | anchors attached {n_att} | identities "
          f"{[i['label'] for i in identities]}")
    print(f"solved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
