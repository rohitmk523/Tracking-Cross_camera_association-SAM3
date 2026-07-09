#!/usr/bin/env python3
"""OFFLINE IDENTITY SOLVER v1 — the batch-architecture keystone.

Input: a fused worldstate (any version). The fused global-ids are treated as
TRACKLETS (intervals), not identities. The solver assigns tracklets to physical
IDENTITIES over the whole window:

  1. ANCHORS: jersey reads are near-absolute evidence — every tracklet with a
     committed (team, number) seeds/joins that identity.
  2. CHAINING: remaining tracklets attach by minimal (time-gap, court-distance,
     appearance, team-compatibility) cost to identities whose timeline they extend —
     greedy over globally sorted link costs, hard vetoes on temporal overlap and
     team contradiction.
  3. Every switch/fragment the tracker made becomes repairable: identity flows
     backward and forward from each anchor.

Output: worldstate with tracks remapped to solved identities (+ solver report).

  python scripts/solve_identities.py --game e6fba750 --tag 44_60 \
      --worldstate runs/tracking/e6fba750_44_60_worldstate_vSAM3.json \
      --version vSAM3_solved
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--worldstate", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--max-gap-s", type=float, default=20.0)
    ap.add_argument("--max-speed-cms", type=float, default=650.0,
                    help="link plausibility: court cm per second (fast break ~600)")
    ap.add_argument("--min-reid-cos", type=float, default=0.45)
    a = ap.parse_args()

    ws = json.loads(Path(a.worldstate).read_text())

    # --- per-cam jersey evidence + reid vectors via members ---
    jersey_rows: dict[tuple, Counter] = defaultdict(Counter)   # (cam, lid) -> Counter(num)
    for ang in ANGLES:
        p = REPO / f"runs/tracking/{a.game}_{ang}_{a.tag}_teams.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text())["tracks"]:
            if r.get("jersey") is not None:
                jersey_rows[(ang, r["track_id"])][int(r["jersey"])] += 1
    reid_vecs: dict[tuple, np.ndarray] = {}
    for ang in ANGLES:
        p = REPO / f"runs/tracking/{a.game}_{ang}_{a.tag}_reid.npz"
        if p.exists():
            z = np.load(p)
            for i, v in zip(z["ids"], z["emb"]):
                reid_vecs[(ang, int(i))] = np.asarray(v, float)

    # --- build tracklets from fused gids ---
    T: dict[int, dict] = {}
    for fr in ws["frames"]:
        f = fr["frame"]
        for t in fr["tracks"]:
            g = t["global_id"]
            d = T.setdefault(g, {"frames": [], "xy": [], "members": Counter(),
                                 "team": Counter(), "jersey": Counter()})
            d["frames"].append(f)
            d["xy"].append(t["court_xy"])
            if t.get("team"):
                d["team"][t["team"]] += 1
            for cam, lid in (t.get("members") or {}).items():
                d["members"][(cam, lid)] += 1
    for g, d in T.items():
        d["start"], d["end"] = min(d["frames"]), max(d["frames"])
        d["start_xy"], d["end_xy"] = d["xy"][0], d["xy"][-1]
        for (cam, lid), n in d["members"].items():
            for num, c in jersey_rows.get((cam, lid), {}).items():
                d["jersey"][num] += c * min(1.0, n / 30.0)
        vs = [reid_vecs[k] for k in d["members"] if k in reid_vecs]
        d["reid"] = None
        if vs:
            v = np.mean(vs, axis=0)
            d["reid"] = v / (np.linalg.norm(v) + 1e-8)
        d["team_maj"] = d["team"].most_common(1)[0][0] if d["team"] else None
        d["num"] = (d["jersey"].most_common(1)[0][0]
                    if d["jersey"] and d["jersey"].most_common(1)[0][1] >= 2.0 else None)

    # --- identities: seed from (team, number) anchors ---
    ident_of: dict[int, int] = {}
    idents: dict[int, dict] = {}
    next_id = 1
    anchor_key_to_ident: dict[tuple, int] = {}
    for g, d in sorted(T.items(), key=lambda kv: -sum(kv[1]["jersey"].values())):
        if d["num"] is None:
            continue
        key = (d["team_maj"], d["num"])
        if key in anchor_key_to_ident:
            iid = anchor_key_to_ident[key]
            # temporal overlap veto: same identity cannot be in two places at once
            spans = idents[iid]["spans"]
            if any(not (d["end"] < s0 - 3 or d["start"] > s1 + 3) for s0, s1 in spans):
                continue                                # conflicting claim: leave unassigned
            ident_of[g] = iid
            idents[iid]["spans"].append((d["start"], d["end"]))
            idents[iid]["members"].append(g)
        else:
            iid = next_id
            next_id += 1
            anchor_key_to_ident[key] = iid
            idents[iid] = {"label": f"{d['team_maj'] or '?'}#{d['num']}",
                           "spans": [(d["start"], d["end"])], "members": [g]}
            ident_of[g] = iid

    # --- chain the rest: global greedy over link costs ---
    unassigned = [g for g in T if g not in ident_of]
    def link_cost(d, iid):
        best = None
        for gm in idents[iid]["members"]:
            e = T[gm]
            # candidate must extend timeline (either direction)
            if d["start"] >= e["end"]:
                gap = (d["start"] - e["end"]) / FPS
                dist = float(np.hypot(d["start_xy"][0] - e["end_xy"][0],
                                      d["start_xy"][1] - e["end_xy"][1]))
            elif e["start"] >= d["end"]:
                gap = (e["start"] - d["end"]) / FPS
                dist = float(np.hypot(e["start_xy"][0] - d["end_xy"][0],
                                      e["start_xy"][1] - d["end_xy"][1]))
            else:
                continue                                 # temporal overlap with this member
            if gap > a.max_gap_s or dist > max(150.0, a.max_speed_cms * max(gap, 0.2)):
                continue
            cos = 0.5
            if d["reid"] is not None and e["reid"] is not None:
                cos = float(np.dot(d["reid"], e["reid"]))
                if cos < a.min_reid_cos:
                    continue
            c = gap * 0.15 + dist / 800.0 + (1.0 - cos)
            best = c if best is None or c < best else best
        return best

    # iterate: costs change as identities grow — a few greedy rounds
    for _ in range(6):
        cands = []
        for g in unassigned:
            d = T[g]
            for iid in idents:
                lbl_team = idents[iid]["label"][0]
                if d["team_maj"] and lbl_team in "AB" and d["team_maj"] in "AB" \
                        and d["team_maj"] != lbl_team:
                    continue
                if d["num"] is not None:               # numbered tracklets already handled
                    continue
                # full-identity temporal overlap veto
                if any(not (d["end"] < s0 - 3 or d["start"] > s1 + 3)
                       for s0, s1 in idents[iid]["spans"]):
                    continue
                c = link_cost(d, iid)
                if c is not None:
                    cands.append((c, g, iid))
        if not cands:
            break
        cands.sort()
        placed = set()
        progress = False
        for c, g, iid in cands:
            if g in placed or g not in unassigned:
                continue
            if any(not (T[g]["end"] < s0 - 3 or T[g]["start"] > s1 + 3)
                   for s0, s1 in idents[iid]["spans"]):
                continue
            ident_of[g] = iid
            idents[iid]["spans"].append((T[g]["start"], T[g]["end"]))
            idents[iid]["members"].append(g)
            unassigned.remove(g)
            placed.add(g)
            progress = True
        if not progress:
            break

    # leftovers become their own identities (unnamed)
    for g in unassigned:
        iid = next_id
        next_id += 1
        idents[iid] = {"label": f"unk{iid}", "spans": [(T[g]["start"], T[g]["end"])],
                       "members": [g]}
        ident_of[g] = iid

    # --- emit remapped worldstate ---
    out_frames = []
    for fr in ws["frames"]:
        tracks = []
        for t in fr["tracks"]:
            t2 = dict(t)
            t2["global_id"] = ident_of[t["global_id"]]
            tracks.append(t2)
        # a solved identity may briefly appear twice (chained gids overlapping a frame
        # or two at seams): keep the non-coasting one
        seen: dict[int, dict] = {}
        for t2 in tracks:
            k = t2["global_id"]
            if k not in seen or (seen[k].get("coasting") and not t2.get("coasting")):
                seen[k] = t2
        out_frames.append({"frame": fr["frame"], "tracks": list(seen.values())})
    players = []
    for iid, info in idents.items():
        nums = Counter()
        teams = Counter()
        for g in info["members"]:
            nums.update(T[g]["jersey"])
            teams.update(T[g]["team"])
        players.append({"global_id": iid,
                        "team": teams.most_common(1)[0][0] if teams else None,
                        "jersey": nums.most_common(1)[0][0] if nums else None,
                        "court_xy": None})
    solved = {"n_global_ids": len(idents), "players": players, "frames": out_frames,
              "ref_angle": ws.get("ref_angle"), "angles": ws.get("angles"), "fps": FPS,
              "solver": {"n_tracklets": len(T), "n_identities": len(idents),
                         "n_anchored": len(anchor_key_to_ident),
                         "n_chained": len(T) - len([g for g in T if T[g]["num"]]) - len(unassigned),
                         "n_leftover": len(unassigned)}}
    out = Path(a.worldstate).parent / f"{a.game}_{a.tag}_worldstate_{a.version}.json"
    out.write_text(json.dumps(solved))
    print(f"solver: {len(T)} tracklets -> {len(idents)} identities "
          f"({len(anchor_key_to_ident)} jersey-anchored, {len(unassigned)} leftover) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
