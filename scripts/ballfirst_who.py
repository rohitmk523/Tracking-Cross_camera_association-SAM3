#!/usr/bin/env python3
"""Possession state machine — WHO HAS THE BALL at every frame.

THE product. Consumes the smoothed per-camera ball trajectories and the
cross-camera-fused player streams, and emits runs/tracking/ledger/
holders_<game>.json (RLE segments of frame-range -> stream id).

Rules: per-frame cross-camera hold votes, switch hysteresis, catch-after-
flight, flight-end LANDING assignment, and RELEASE (nobody has the ball)
when the ball is airborne or hold evidence has been absent too long.
Scored against annotator GT by scripts/score_holders_gt.py.

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
# RELEASE (nobody has the ball). GT says ~32% of frames have no holder;
# without these the machine holds the last player forever and is wrong on
# every one of them. Ball coverage is only ~50%/cam, so a dropout must NOT
# release immediately — only sustained absence of any hold evidence does.
import os
FLIGHT_RELEASE = int(os.environ.get("UBALL_FLIGHT_RELEASE", 2))
HOLD_GRACE = int(os.environ.get("UBALL_HOLD_GRACE", 10))


def holder_timeline(game, tag, offs, tglob):
    """Chunk-wide holder per ref frame via the state machine."""
    ball, tracks = {}, defaultdict(dict)
    for ang in ANGLES:
        # LAYER 1: smoothed trajectory (Kalman+RTS, gaps bridged) instead of
        # raw per-frame detections — smooth velocity, teleports rejected
        z = np.load(REPO / f"runs/ball_traj/{game}_{ang}_{tag}.traj.npz")
        bd = {}
        for f, cx, cy, vx, vy, imp, cf in zip(
                z["frame_idx"], z["cx"], z["cy"], z["vx"], z["vy"],
                z["imputed"], z["conf"]):
            bd[int(f)] = (float(cx), float(cy), float(np.hypot(vx, vy)),
                          float(vx), float(vy), float(cf), int(imp))
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
        fv_conf = -1.0
        for ang in ANGLES:
            cf = f + offs[ang]
            bb = ball[ang].get(cf)
            if not bb:
                continue
            bx, by, v, vx, vy, cfid, imp = bb
            speeds.append(v)
            if v > FLIGHT_V and cfid > fv_conf:
                # real smoothed velocity — the LANDING rule extrapolates this
                flight_vec = (ang, bx, by, vx, vy)
                fv_conf = cfid
            wmul = 0.5 if imp else 1.0
            for sid, angs in tracks.items():
                box = angs.get(ang, {}).get(cf)
                if box is None or by > box[3]:
                    continue
                bw = max(box[2] - box[0], 1.0)
                dx = abs(bx - (box[0] + box[2]) / 2) / bw
                if dx <= HOLD_D and by >= box[1] - 0.1 * (box[3] - box[1]):
                    votes[sid] += wmul * (1.0 - 0.5 * dx)
        v = float(np.median(speeds)) if speeds else None
        state = ("FLIGHT" if (v is not None and v > FLIGHT_V and not votes)
                 else "HOLD" if votes else "GAP")
        obs.append((f, state, dict(votes), flight_vec))

    # state machine with catch + LANDING rule
    holder = {}
    cur, cand, streak, since_flight = None, None, 0, 99
    since_hold = 0                          # frames since any positive hold vote
    last_flight = None                      # (ang, x, y, vx, vy, f)
    for f, state, votes, fv in obs:
        if state == "FLIGHT":
            since_flight = 0
            if fv:
                last_flight = (*fv, f)
        else:
            since_flight += 1
        if state != "HOLD" or not votes:
            since_hold += 1
        else:
            since_hold = 0
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
        # RELEASE: ball airborne with no candidate, or no hold evidence for
        # long enough that continuing to name a holder is a fabrication.
        if (state == "FLIGHT" and since_hold > FLIGHT_RELEASE) or since_hold > HOLD_GRACE:
            cur, cand, streak = None, None, 0
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
        m = [p for p in c if p["team"] == roster.get("kit_team", {"B": 1, "W": 2}).get(kit)]
        return m[0]["name"].split()[-1] if len(m) == 1 else "?"

    timelines = {}
    for tag in GAME_CHUNKS[game]:
        timelines[tag], _ = holder_timeline(game, tag, offs, tglob)
        print(f"  [{tag}] holder frames: {len(timelines[tag])}", flush=True)
    # cache holder timelines as RLE segments (global frames) for the events layer
    segs = []
    for tag in GAME_CHUNKS[game]:
        base = round(float(tag.split("_")[0]) * FPS)
        for f in sorted(timelines[tag]):
            pl = timelines[tag][f]
            gf = base + f
            if segs and segs[-1][2] == pl and gf - segs[-1][1] <= 3:
                segs[-1][1] = gf
            else:
                segs.append([gf, gf, pl])
    outp = REPO / f"runs/tracking/ledger/holders_{game}.json"
    outp.write_text(json.dumps({"fps": FPS, "segments": segs}))
    print(f"holder cache -> {outp} ({len(segs)} segments)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
