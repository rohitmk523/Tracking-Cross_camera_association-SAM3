"""Tracklet-level association (docs/06 follow-up): merge fragmented global IDs.

Per-frame fusion still births a fresh global ID whenever a player drops out longer than
the re-entry memory or re-appears past the gates — the audited runs carried ~20 IDs for
~13 people. Top MTMC systems close exactly this gap with a tracklet post-pass (SoccerNet
GSR 2024 winner: "refines and merges short tracklets into longer trajectories").

Rule: two global tracklets are the SAME person when they
  never overlap in time (a person can't be two IDs at once),
  bridge a short gap (<= max_gap frames),
  line up spatially (endpoint distance <= dist_base + dist_per_frame * gap, capped),
  and are team-compatible (same team, or one side unknown; REF only merges with REF).
Merges are applied greedily, lowest spatial cost first, transitively re-checked.
"""
from __future__ import annotations

import numpy as np

MAX_GAP_FRAMES = 75            # ~2.5s: beyond this, identity is a guess -> leave split
DIST_BASE_CM = 200.0           # endpoint tolerance at gap 0
DIST_PER_FRAME_CM = 6.0        # ~1.8 m/s of drift allowance while unseen
DIST_CAP_CM = 450.0


def _spans(frames: list[dict]) -> dict[int, dict]:
    """gid -> {first, last, first_xy, last_xy, team, n}."""
    out: dict[int, dict] = {}
    for fr in frames:
        for t in fr["tracks"]:
            g = t["global_id"]
            s = out.setdefault(g, {"first": fr["frame"], "last": fr["frame"],
                                   "first_xy": t["court_xy"], "last_xy": t["court_xy"],
                                   "team": None, "n": 0, "_teams": {}})
            s["last"], s["last_xy"], s["n"] = fr["frame"], t["court_xy"], s["n"] + 1
            if t.get("team"):
                s["_teams"][t["team"]] = s["_teams"].get(t["team"], 0) + 1
    for s in out.values():
        s["team"] = max(s["_teams"], key=s["_teams"].get) if s["_teams"] else None
        del s["_teams"]
    return out


def _compatible(a: dict, b: dict) -> bool:
    ta, tb = a["team"], b["team"]
    if ta == "REF" or tb == "REF":
        return ta == tb                              # a ref never merges with a player
    return ta is None or tb is None or ta == tb


def merge_map(frames: list[dict], *, max_gap: int = MAX_GAP_FRAMES,
              dist_base: float = DIST_BASE_CM, dist_per_frame: float = DIST_PER_FRAME_CM,
              dist_cap: float = DIST_CAP_CM) -> dict[int, int]:
    """{old_gid: canonical_gid} for every fragment that should fold into an earlier track."""
    spans = _spans(frames)
    # candidate pairs: a ends, then b starts within the gap, close enough, compatible
    cands = []
    for ga, a in spans.items():
        for gb, b in spans.items():
            if gb == ga or b["first"] <= a["last"]:            # must be strictly later
                continue
            gap = b["first"] - a["last"]
            if gap > max_gap or not _compatible(a, b):
                continue
            d = float(np.hypot(a["last_xy"][0] - b["first_xy"][0],
                               a["last_xy"][1] - b["first_xy"][1]))
            if d <= min(dist_cap, dist_base + dist_per_frame * gap):
                cands.append((d + 0.5 * gap, ga, gb))
    cands.sort()
    root: dict[int, int] = {}

    def find(g: int) -> int:
        while root.get(g, g) != g:
            g = root[g]
        return g

    merged_span = {g: dict(s) for g, s in spans.items()}
    for _, ga, gb in cands:
        ra, rb = find(ga), find(gb)
        if ra == rb:
            continue
        a, b = merged_span[ra], merged_span[rb]
        if b["first"] <= a["last"]:                            # transitive overlap -> unsafe
            continue
        if not _compatible(a, b):
            continue
        root[rb] = ra                                          # fold the later into the earlier
        a["last"], a["last_xy"], a["n"] = b["last"], b["last_xy"], a["n"] + b["n"]
        a["team"] = a["team"] or b["team"]
    return {g: find(g) for g in spans if find(g) != g}


def apply_merges(world_state: dict, mapping: dict[int, int]) -> dict:
    """Return a NEW world-state with fragment IDs folded into their canonical IDs."""
    if not mapping:
        return world_state
    frames = [{**fr, "tracks": [{**t, "global_id": mapping.get(t["global_id"], t["global_id"])}
                                for t in fr["tracks"]]}
              for fr in world_state["frames"]]
    roster: dict[int, dict] = {}
    for p in world_state["players"]:
        g = mapping.get(p["global_id"], p["global_id"])
        cur = roster.get(g)
        if cur is None:
            roster[g] = {**p, "global_id": g}
        else:                                                   # keep the richer record
            cur["team"] = cur["team"] or p.get("team")
            cur["jersey"] = cur["jersey"] if cur["jersey"] is not None else p.get("jersey")
    players = sorted(roster.values(), key=lambda p: p["global_id"])
    return {**world_state, "frames": frames, "players": players, "n_global_ids": len(players)}
