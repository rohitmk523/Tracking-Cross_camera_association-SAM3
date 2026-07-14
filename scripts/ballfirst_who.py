#!/usr/bin/env python3
"""Engine v3 WHO — possession state machine over the full game.

Keeps the proven arc trigger (detection 88-94%); replaces release-instant
attribution with the BALL STORY: shooter = the holder whose HOLD ends in the
flight that reaches the rim. Rules from the prototype: catch-switching after
flight, FLIGHT-END LANDING assignment (trajectory extrapolation names the
catcher), backward fill between anchors.

  .venv/bin/python scripts/ballfirst_who.py --game c2a354fe
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from game_meta import GAME_OFFS, GAME_CHUNKS

FPS = 29.97
ANGLES = ("FL", "FR", "NL", "NR")
HOLD_D = 0.7
SWITCH_K = 6
CATCH_K = 3
FLIGHT_V = 14.0


def holder_timeline(game, tag, offs, tglob):
    """Chunk-wide holder per ref frame via the state machine."""
    ball, tracks = {}, defaultdict(dict)
    for ang in ANGLES:
        z = np.load(REPO / f"runs/ball_cache/{game}_{ang}_{tag}.ball.npz")
        bd = {}
        for b, sc, f, c in zip(z["boxes"], z["scores"], z["frame_idx"], z["classes"]):
            if int(c) != 0:
                continue
            f = int(f)
            if f not in bd or sc > bd[f][1]:
                bd[f] = ([float(v) for v in b], float(sc))
        ball[ang] = bd
    for p in (REPO / tglob.format(tag=tag)).glob(f"{game}_{tag}__n*__*.json"):
        parts = p.stem.split("__")
        d = json.loads(p.read_text())["frames"]
        tracks[parts[1]][parts[2]] = {int(fr): r["box"]
                                      for fr, r in d.items() if r.get("present")}
    if not tracks:
        return {}, {}
    max_f = max(max(v) - offs[a] for a in ANGLES for v in [ball[a]] if v)

    # per-frame observations
    obs = []
    last_pos = {a: None for a in ANGLES}
    for f in range(0, max_f + 1):
        votes = defaultdict(float)
        speeds = []
        flight_vec = None
        for ang in ANGLES:
            cf = f + offs[ang]
            bb = ball[ang].get(cf)
            if not bb:
                continue
            bx, by = (bb[0][0] + bb[0][2]) / 2, (bb[0][1] + bb[0][3]) / 2
            prev = ball[ang].get(cf - 3)
            if prev:
                px, py = (prev[0][0] + prev[0][2]) / 2, (prev[0][1] + prev[0][3]) / 2
                v = np.hypot(bx - px, by - py) / 3
                speeds.append(v)
                if v > FLIGHT_V:
                    flight_vec = (ang, bx, by, (bx - px) / 3, (by - py) / 3)
            for sid, angs in tracks.items():
                box = angs.get(ang, {}).get(cf)
                if box is None or by > box[3]:
                    continue
                bw = max(box[2] - box[0], 1.0)
                dx = abs(bx - (box[0] + box[2]) / 2) / bw
                if dx <= HOLD_D and by >= box[1] - 0.1 * (box[3] - box[1]):
                    votes[sid] += (1.0 - 0.5 * dx)
        v = float(np.median(speeds)) if speeds else None
        state = ("FLIGHT" if (v is not None and v > FLIGHT_V and not votes)
                 else "HOLD" if votes else "GAP")
        obs.append((f, state, dict(votes), flight_vec))

    # state machine with catch + LANDING rule
    holder = {}
    cur, cand, streak, since_flight = None, None, 0, 99
    last_flight = None                      # (ang, x, y, vx, vy, f)
    for f, state, votes, fv in obs:
        if state == "FLIGHT":
            since_flight = 0
            if fv:
                last_flight = (*fv, f)
        else:
            since_flight += 1
        if state == "HOLD" and votes:
            top = max(votes, key=votes.get)
            # LANDING rule: right after flight, prefer the stream whose box
            # contains the extrapolated landing point of that flight
            if since_flight <= 4 and last_flight is not None:
                ang, lx, ly, vx, vy, lf0 = last_flight
                dt = (f + offs[ang]) - (lf0 + offs[ang])
                px, py = lx + vx * dt, ly + vy * dt
                landing = None
                for sid, angs in tracks.items():
                    box = angs.get(ang, {}).get(f + offs[ang])
                    if box is None:
                        continue
                    if box[0] - 10 <= px <= box[2] + 10 and box[1] <= py <= box[3] + 20:
                        if landing is None or votes.get(sid, 0) > votes.get(landing, 0):
                            landing = sid
                if landing is not None and landing in votes:
                    top = landing
            k_need = CATCH_K if since_flight <= 4 else SWITCH_K
            if top != cur:
                streak = streak + 1 if top == cand else 1
                cand = top
                if streak >= k_need:
                    cur = top
                    streak = 0
            else:
                streak = 0
        if cur is not None:
            holder[f] = cur
    return holder, tracks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--plays", default=None)
    ap.add_argument("--tracks-glob", default=None)
    a = ap.parse_args()
    game = a.game
    offs = GAME_OFFS[game]
    tglob = a.tracks_glob or ("runs/events_fg_{tag}" if game == "e6fba750"
                              else f"runs/events_fg_{game[:3]}_{{tag}}")
    playsf = a.plays or f"data/plays/{game}_full.json"
    led = json.loads((REPO / f"runs/tracking/ledger/shots_{game}_full.json").read_text())
    plays = json.loads((REPO / playsf).read_text())["plays"]
    gt = [p for p in plays if ("MAKE" in p["cls"] or "MISS" in p["cls"])]
    roster = json.loads((REPO / f"data/rosters/{game}.json").read_text())
    by_num = defaultdict(list)
    for p in roster["players"]:
        by_num[p["num"]].append(p)

    def name_last(sid):
        core = sid.lstrip("#").lstrip("n")
        kit = core[-1] if core[-1] in ("B", "W") else None
        num = int(core.rstrip("BW"))
        c = by_num.get(num, [])
        if len(c) == 1:
            return c[0]["name"].split()[-1]
        m = [p for p in c if p["team"] == {"B": 1, "W": 2}.get(kit)]
        return m[0]["name"].split()[-1] if len(m) == 1 else "?"

    timelines = {}
    for tag in GAME_CHUNKS[game]:
        timelines[tag], _ = holder_timeline(game, tag, offs, tglob)
        print(f"  [{tag}] holder frames: {len(timelines[tag])}", flush=True)

    base_ok = v3_ok = tot = 0
    rows = []
    for g in gt:
        cand = [o for o in led if abs(o["t"] - g["t"]) <= 1.5 and o.get("rel_f") is not None]
        if not cand:
            continue
        o = min(cand, key=lambda o: (o.get("rq") is None, o.get("rq", 9.9),
                                     abs(o["t"] - g["t"])))
        tot += 1
        gt_last = g["a"].split()[-1]
        base_ok += bool(o["pred_player"]
                        and o["pred_player"].rstrip("?").split()[-1] == gt_last)
        tl = timelines[o["chunk"]]
        arr = o["arrive_f"]
        # v3 WHO: holder at the release moment (HOLD feeding the arc's
        # flight) — a stale last-holder from seconds earlier must not count
        rel = o.get("rel_f") or (arr - 20)
        pick = None
        pf = None
        for f in range(rel + 8, rel - int(1.2 * FPS), -1):
            h = tl.get(f)
            if h is not None:
                pick, pf = h, f
                break
        v3_ok += bool(pick and name_last(pick) == gt_last)
        b_ok = bool(o["pred_player"]
                    and o["pred_player"].rstrip("?").split()[-1] == gt_last)
        v_ok = bool(pick and name_last(pick) == gt_last)
        agree = bool(pick and o["pred_player"]
                     and name_last(pick) == o["pred_player"].rstrip("?").split()[-1])
        rows.append({"b": b_ok, "v": v_ok, "agree": agree,
                     "rq": o.get("rq"), "cls": g["cls"],
                     "dist": o.get("release_dist_cm")})
    print(f"\n{game}: n={tot} | baseline {base_ok}/{tot} ({base_ok/tot:.0%}) "
          f"| v3 {v3_ok}/{tot} ({v3_ok/tot:.0%})")
    def score(rule, label):
        ok = sum((r["v"] if rule(r) else r["b"]) for r in rows)
        print(f"  arbiter [{label}]: {ok}/{tot} ({ok/tot:.0%})")
    score(lambda r: not r["agree"] and r["rq"] is not None and r["rq"] > 0.6,
          "a: disagree & dirty release -> v3")
    score(lambda r: r["rq"] is not None and r["rq"] > 0.6, "b: dirty release -> v3")
    score(lambda r: (r["dist"] or 9e9) < 500, "c: paint shots -> v3")
    score(lambda r: not r["agree"] and (r["dist"] or 9e9) < 600,
          "d: disagree & close -> v3")
    both = sum(1 for r in rows if r["b"] and r["v"])
    print(f"  union {sum(1 for r in rows if r['b'] or r['v'])}/{tot} | "
          f"both {both} | agree-rate {sum(r['agree'] for r in rows)}/{tot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
