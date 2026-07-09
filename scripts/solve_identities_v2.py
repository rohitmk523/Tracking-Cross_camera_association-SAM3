#!/usr/bin/env python3
"""OFFLINE IDENTITY SOLVER v2 — frame-level anchored re-tracking.

v1 failed because fused tracklets are internally contaminated. v2 ignores every
upstream identity decision: it takes per-frame fused OBSERVATIONS (court position +
per-camera member boxes + ReID) and re-assigns identities offline with the three
advantages no online tracker has:
  1. DENSE ANCHORS — 4k+ per-window jersey reads; an anchored observation is
     force-assigned to its (team, number) identity, and contradicting assignments
     are forbidden.
  2. BOTH DIRECTIONS — a forward pass and a backward pass; disagreements resolved
     by whichever pass is closer to its supporting anchor (identity flows outward
     from every clamp).
  3. ROSTER — only real numbers may exist; misreads can't create people.

  python scripts/solve_identities_v2.py --game e6fba750 --tag 44_60 \
      --worldstate runs/tracking/e6fba750_44_60_worldstate_vSAM3.json --version vS2
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


def load_all(a, key):
    ws = json.loads(Path(a.worldstate).read_text())
    boxes = {}
    teams_rows = {}
    for ang in ANGLES:
        p = REPO / f"runs/tracking/{a.game}_{ang}_{a.tag}_teams.json"
        m = defaultdict(dict)
        for r in json.loads(p.read_text())["tracks"]:
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
    anchors_raw = json.loads((REPO / f"runs/anchors/{key}.jersey_anchors.json").read_text())
    return ws, teams_rows, reid, anchors_raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--worldstate", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--min-anchor-count", type=int, default=20)
    ap.add_argument("--gate-cm", type=float, default=420.0)
    ap.add_argument("--w-reid", type=float, default=1.2)
    a = ap.parse_args()
    from scipy.optimize import linear_sum_assignment

    key = f"{a.game}_{a.tag}"
    offs = OFFS[key]
    ws, teams_rows, reid, anchors_raw = load_all(a, key)

    # ---- observations per frame (ignore upstream gids) ----
    # obs = {frame: [ {xy, cams:{ang:(lid,box,team)}, reid, obs_team} ]}
    OBS = {}
    for fr in ws["frames"]:
        f = fr["frame"]
        row = []
        for t in fr["tracks"]:
            if t.get("coasting"):
                continue
            cams = {}
            vecs = []
            tvotes = Counter()
            for ang, lid in (t.get("members") or {}).items():
                rr = teams_rows[ang].get(f + offs[ang], {}).get(lid)
                if rr is None:
                    continue
                cams[ang] = (lid, rr["box_xyxy"], rr.get("team"))
                if rr.get("team") in ("A", "B"):
                    tvotes[rr["team"]] += 1
                if (ang, lid) in reid:
                    vecs.append(reid[(ang, lid)])
            v = None
            if vecs:
                v = np.mean(vecs, axis=0)
                v = v / (np.linalg.norm(v) + 1e-8)
            row.append({"xy": np.array(t["court_xy"], float), "cams": cams, "reid": v,
                        "team": tvotes.most_common(1)[0][0] if tvotes else None,
                        "anchor": None})
        OBS[f] = row

    # ---- attach anchors to observations (same cam, box IoU) ----
    n_attached = 0
    roster = Counter()
    by_frame_anchor = defaultdict(list)
    for ev in anchors_raw["anchors"]:
        by_frame_anchor[ev["frame"]].append(ev)
    for f, evs in by_frame_anchor.items():
        row = OBS.get(f)
        if not row:
            continue
        for ev in evs:
            best, bi = 0.35, None
            for i, o in enumerate(row):
                got = o["cams"].get(ev["cam"])
                if got is None:
                    continue
                v = iou(got[1], ev["box"])
                if v > best:
                    best, bi = v, i
            if bi is not None:
                o = row[bi]
                akey = (o["team"], int(ev["number"]))
                if o["anchor"] is None:
                    o["anchor"] = Counter()
                o["anchor"][akey] += 1
                n_attached += 1
                if o["team"] in ("A", "B"):
                    roster[akey] += 1

    # ---- GLOBAL number->team resolution: obs team letters are noisy; a number's
    # true team(s) come from its anchor-weighted distribution. Only genuinely
    # bimodal numbers (>=25% minority, both sides substantial) are two identities.
    by_num = defaultdict(Counter)
    for (tm, num), c in roster.items():
        if tm in ("A", "B"):
            by_num[num][tm] += c
    resolved = {}                                   # num -> list of teams
    remap_anchor = {}                               # (obs_team, num) -> canonical (team, num)
    for num, dist in by_num.items():
        total = sum(dist.values())
        if total < a.min_anchor_count:
            continue
        maj, majc = dist.most_common(1)[0]
        minority = total - majc
        if minority >= 0.25 * total and minority >= 30:
            resolved[num] = ["A", "B"]              # real both-team number (e6: 11, 22)
            remap_anchor[("A", num)] = ("A", num)
            remap_anchor[("B", num)] = ("B", num)
        else:
            resolved[num] = [maj]
            remap_anchor[("A", num)] = (maj, num)
            remap_anchor[("B", num)] = (maj, num)
    # re-key every attached anchor through the resolution
    for f, row in OBS.items():
        for o in row:
            if o["anchor"]:
                na = Counter()
                for k, c in o["anchor"].items():
                    nk = remap_anchor.get(k)
                    if nk:
                        na[nk] += c
                o["anchor"] = na if na else None
    idents = [(tm, num) for num, tms in resolved.items() for tm in tms]
    idents.sort(key=lambda k: -by_num[k[1]][k[0]])
    NAMED = {k: i for i, k in enumerate(idents)}
    n_named = len(NAMED)
    print(f"anchors attached: {n_attached}; identities from roster: "
          f"{[f'{t}#{n}({roster[(t, n)]})' for t, n in idents]}")

    def solve_pass(frames_order):
        state = {}
        for k, i in NAMED.items():
            state[i] = {"xy": None, "v": np.zeros(2), "last": None, "reid": None,
                        "team": k[0], "num": k[1]}
        next_pool = n_named
        assign_out = {}
        pool_meta = {}
        for f in frames_order:
            row = OBS.get(f, [])
            if not row:
                continue
            ids = list(state.keys())
            C = np.full((len(row), len(ids)), 1e6)
            for oi, o in enumerate(row):
                akey = acount = None
                if o["anchor"]:
                    akey, acount = o["anchor"].most_common(1)[0]
                for ii, iid in enumerate(ids):
                    s = state[iid]
                    # anchor logic: FORCE only on multi-read agreement; single reads
                    # are a bonus, not a command (one misread must not yank identity)
                    if akey is not None and akey in NAMED:
                        if iid == NAMED[akey]:
                            if acount >= 2:
                                C[oi, ii] = -100.0
                                continue
                            bonus = -2.5
                        else:
                            bonus = 0.0
                            if acount >= 2 and s.get("num") is not None:
                                continue                  # strongly-anchored obs: no other named
                    else:
                        bonus = 0.0
                    if s["xy"] is None:
                        base = 3.5                        # unseen identity: acquisition cost
                        d = 0.0
                    else:
                        gap = max(1, f - s["last"]) if s["last"] is not None else 1
                        pred = s["xy"] + s["v"] * min(gap, 15)
                        d = float(np.linalg.norm(o["xy"] - pred))
                        if d > a.gate_cm + 25 * min(gap, 40):
                            continue
                        base = d / 300.0 + min(gap, 60) / 45.0
                    cos_term = 0.0
                    if o["reid"] is not None and s["reid"] is not None:
                        cos_term = a.w_reid * (1.0 - float(np.dot(o["reid"], s["reid"])))
                    if o["team"] and s["team"] and o["team"] != s["team"]:
                        base += 4.0
                    C[oi, ii] = base + cos_term + bonus
            ri, ci = linear_sum_assignment(C)
            used = set()
            out_f = {}
            for oi, ii in zip(ri, ci):
                if C[oi, ii] >= 1e5:
                    continue
                iid = ids[ii]
                out_f[oi] = iid
                used.add(oi)
                s = state[iid]
                o = row[oi]
                if s["xy"] is not None and s["last"] is not None and f != s["last"]:
                    s["v"] = 0.7 * s["v"] + 0.3 * (o["xy"] - s["xy"]) / (f - s["last"])
                s["xy"], s["last"] = o["xy"], f
                if o["reid"] is not None:
                    s["reid"] = o["reid"] if s["reid"] is None else \
                        0.9 * s["reid"] + 0.1 * o["reid"]
                    s["reid"] /= (np.linalg.norm(s["reid"]) + 1e-8)
                if o["team"] and s["team"] is None:
                    s["team"] = o["team"]
            for oi, o in enumerate(row):
                if oi in used:
                    continue
                iid = next_pool
                next_pool += 1
                state[iid] = {"xy": o["xy"], "v": np.zeros(2), "last": f,
                              "reid": o["reid"], "team": o["team"], "num": None}
                out_f[oi] = iid
            assign_out[f] = out_f
        return assign_out

    frames_sorted = sorted(OBS)
    fwd = solve_pass(frames_sorted)
    bwd = solve_pass(list(reversed(frames_sorted)))

    # ---- reconcile: named agreement wins; disagreement -> nearest-anchor pass ----
    # precompute per-frame distance to nearest anchor-attached obs frame (global)
    anchor_frames = sorted({f for f, evs in by_frame_anchor.items() if OBS.get(f)})
    af = np.array(anchor_frames) if anchor_frames else np.array([0])

    final = {}
    for f in frames_sorted:
        row = OBS.get(f, [])
        out_f = {}
        for oi in range(len(row)):
            a_id = fwd.get(f, {}).get(oi)
            b_id = bwd.get(f, {}).get(oi)
            a_named = a_id is not None and a_id < n_named
            b_named = b_id is not None and b_id < n_named
            if a_named and b_named:
                if a_id == b_id:
                    out_f[oi] = a_id
                else:
                    # nearest anchor arbitration: forward pass carries identity from the
                    # PAST anchor, backward from the FUTURE one — trust the closer clamp
                    past = af[af <= f]
                    futr = af[af >= f]
                    dp = f - past[-1] if len(past) else 1e9
                    dfu = futr[0] - f if len(futr) else 1e9
                    out_f[oi] = a_id if dp <= dfu else b_id
            elif a_named:
                out_f[oi] = a_id
            elif b_named:
                out_f[oi] = b_id
            else:
                out_f[oi] = n_named + 500 + (a_id if a_id is not None else 0)
        final[f] = out_f

    # ---- emit worldstate ----
    label = {i: f"{k[0]}#{k[1]}" for k, i in NAMED.items()}
    out_frames = []
    for fr in ws["frames"]:
        f = fr["frame"]
        row = OBS.get(f, [])
        assigned = final.get(f, {})
        tracks = []
        seen = set()
        oi = 0
        for t in fr["tracks"]:
            if t.get("coasting"):
                continue
            iid = assigned.get(oi)
            oi += 1
            if iid is None or iid in seen:
                continue
            seen.add(iid)
            t2 = dict(t)
            t2["global_id"] = int(iid)
            tracks.append(t2)
        out_frames.append({"frame": f, "tracks": tracks})
    players = [{"global_id": i, "team": k[0], "jersey": k[1], "court_xy": None}
               for k, i in NAMED.items()]
    solved = {"n_global_ids": len(players), "players": players, "frames": out_frames,
              "ref_angle": ws.get("ref_angle"), "angles": ws.get("angles"), "fps": FPS,
              "solver_v2": {"anchors_attached": n_attached, "named": [label[i] for i in label]}}
    out = Path(a.worldstate).parent / f"{a.game}_{a.tag}_worldstate_{a.version}.json"
    out.write_text(json.dumps(solved))
    print(f"solved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
