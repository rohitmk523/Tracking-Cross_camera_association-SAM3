#!/usr/bin/env python3
"""Possession-change events (events v2 part B): REBOUND / TURNOVER+STEAL.

Rides on (a) the possession timeline (detect_events' cross-cam vote +
hysteresis + sticky, recomputed here per chunk over the full game) and (b) the
v2 shot events (runs/tracking/ledger/events_v2_{game}.json — arc-triggered,
P3 make/miss).

  REBOUND: after a shot event with a MISS verdict, the first sustained
           possession within [t+0.2, t+5.0]s. Off/def from rebounder's team
           vs shooter's team (GT class is just REBOUND — WHO is what's scored).
  TURNOVER+STEAL: possession changes TEAM with no shot event nearby and both
           sides holding ≥1s -> TURNOVER (loser) + STEAL (gainer).

Scored vs the plays GT (e6: 54 REBOUND, 4 TURNOVER, 1 STEAL).

  python scripts/detect_possession_events.py --game e6fba750
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

ANGLES = ("FL", "FR", "NL", "NR")
from game_meta import GAME_OFFS as OFFS, GAME_CHUNKS
FPS = 29.97

POSSESS_EXPAND = 1.35
HYSTERESIS = 8
REBOUND_WINDOW = (0.2, 5.0)
SUSTAIN_F = 10                  # frames a rebounder must hold within the window
TO_HOLD_S = 1.0                 # each side's hold to call a live-ball team change
TO_SHOT_GUARD_S = 3.0           # no shot event this close to a team change


def box_ball_dist(pb, bb):
    bx, by = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    cx, cy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
    w, h = (pb[2] - pb[0]) * POSSESS_EXPAND / 2, (pb[3] - pb[1]) * POSSESS_EXPAND / 2
    if abs(bx - cx) <= w and abs(by - cy) <= h:
        return 0.0
    return float(np.hypot((bx - cx) / max(w, 1), (by - cy) / max(h, 1)))


def possession_chunk(game, tag, offs):
    """{chunk-local ref frame -> identity} via per-cam nearest + vote +
    hysteresis + sticky (detect_events core, ball class 0 ONLY — the corrected
    caches also carry hoop rows which would fake a permanent rim possession)."""
    ball = {ang: {} for ang in ANGLES}
    for ang in ANGLES:
        p = REPO / f"runs/ball_cache/{game}_{ang}_{tag}.ball.npz"
        if not p.exists():
            return {}
        z = np.load(p)
        cls = z["classes"] if "classes" in z else np.zeros(len(z["scores"]))
        for b, s, f, c in zip(z["boxes"], z["scores"], z["frame_idx"], cls):
            if int(c) != 0:
                continue
            f = int(f)
            if f not in ball[ang] or s > ball[ang][f][1]:
                ball[ang][f] = ([float(v) for v in b], float(s))
    tracks = defaultdict(dict)
    tdir = REPO / (f"runs/events_fg_{tag}" if game == "e6fba750"
                   else f"runs/events_fg_{game[:3]}_{tag}")
    for p in tdir.glob(f"{game}_{tag}__n*__*.json"):
        parts = p.stem.split("__")
        pl, ang = "#" + parts[1][1:], parts[2]
        d = json.loads(p.read_text())["frames"]
        tracks[pl][ang] = {int(f): r["box"] for f, r in d.items() if r.get("present")}
    players = sorted(tracks)
    if not players:
        return {}
    max_f = max(max(v) - offs[ang] for ang in ANGLES for v in [ball[ang]] if v)
    raw = {}
    for f in range(0, max_f + 1):
        votes = Counter()
        for ang in ANGLES:
            bb = ball[ang].get(f + offs[ang])
            if not bb:
                continue
            best, bpl = 2.0, None
            for pl in players:
                pb = tracks[pl].get(ang, {}).get(f + offs[ang])
                if pb is None:
                    continue
                d = box_ball_dist(pb, bb[0])
                if d < best:
                    best, bpl = d, pl
            if bpl is not None:
                votes[bpl] += 2 if best == 0.0 else 1
        if votes:
            raw[f] = votes.most_common(1)[0][0]
    possess, cur, streak, last_cand = {}, None, 0, None
    for f in range(0, max_f + 1):
        cand = raw.get(f)
        if cand is not None and cand != cur:
            streak = streak + 1 if cand == last_cand else 1
            last_cand = cand
            if streak >= HYSTERESIS:
                cur = cand
        elif cand == cur:
            streak = 0
        if cur is not None:
            possess[f] = cur
    return possess


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="e6fba750")
    ap.add_argument("--plays", default="data/plays/e6fba750_full.json")
    a = ap.parse_args()
    offs = OFFS[a.game]
    roster = json.loads((REPO / f"data/rosters/{a.game}.json").read_text())
    by_num = defaultdict(list)
    for pr in roster["players"]:
        by_num[pr["num"]].append(pr)

    def rec_of(pl):
        num = int("".join(c for c in pl if c.isdigit()))
        c = by_num.get(num, [])
        return c[0] if len(c) == 1 else (c[0] if c else None)

    # ---- full-game possession timeline (global seconds), RLE segments ----
    pos_path = REPO / f"runs/tracking/ledger/possession_{a.game}.json"
    if pos_path.exists():
        segs = json.loads(pos_path.read_text())["segments"]
    else:
        glob = {}
        for tag in GAME_CHUNKS[a.game]:
            t0 = float(tag.split("_")[0])
            pc = possession_chunk(a.game, tag, offs)
            for f, pl in pc.items():
                glob[round(t0 * FPS) + f] = pl
            print(f"  [{tag}] possession frames: {len(pc)}", flush=True)
        segs = []
        for f in sorted(glob):
            pl = glob[f]
            if segs and segs[-1][2] == pl and f - segs[-1][1] <= 3:
                segs[-1][1] = f
            else:
                segs.append([f, f, pl])
        segs = [s for s in segs if s[1] - s[0] + 1 >= SUSTAIN_F]
        pos_path.write_text(json.dumps({"fps": FPS, "segments": segs}))
    print(f"possession segments (>= {SUSTAIN_F}f): {len(segs)}")

    ev = json.loads((REPO / f"runs/tracking/ledger/events_v2_{a.game}.json").read_text())["events"]
    shots = [e for e in ev if not e["classification"].endswith("_ATTEMPT")]
    misses = [e for e in ev if e["classification"].endswith("_MISS")]

    # ---- REBOUND: first sustained possession after a missed shot ----
    out = []
    for m in misses:
        lo = (m["t"] + REBOUND_WINDOW[0]) * FPS
        hi = (m["t"] + REBOUND_WINDOW[1]) * FPS
        # the segment must BEGIN inside the window: the sticky possession keeps
        # the SHOOTER as possessor through the ball flight, so his ongoing
        # segment overlaps every post-miss window (measured: WHO 3/43)
        cand = [s for s in segs if lo <= s[0] <= hi
                and min(s[1], hi) - s[0] + 1 >= SUSTAIN_F]
        if not cand:
            continue
        # who SECURED it: longest hold beginning in the window (measured:
        # first-touch WHO 30%, longest-hold 37% — remaining errors are the
        # crowd-possession signal itself, same root cause as FG-paint WHO)
        s = max(cand, key=lambda s: s[1] - s[0])
        r = rec_of(s[2])
        sh_team = next((p["team"] for p in roster["players"]
                        if p["name"] == m.get("player_a")), None)
        kind = ("OFF" if r and sh_team and r["team"] == sh_team else
                "DEF" if r and sh_team else "?")
        out.append({"t": round(max(s[0], lo) / FPS, 1), "classification": "REBOUND",
                    "player_a": r["name"] if r else s[2], "off_def": kind,
                    "after_shot_t": m["t"], "source": "cv", "confidence": 0.6})

    # ---- TURNOVER + STEAL: TEAM-level timeline first (raw player segments
    # flip constantly — 255 emitted vs 5 GT). Merge consecutive same-team
    # blocks across <=1.5s gaps, drop blocks <2s, then a change between
    # adjacent blocks (gap <=2s, no shot event within +-3s) is a live-ball
    # team change: TURNOVER (last holder) + STEAL (first new holder).
    blocks = []                                  # [f0, f1, team, last_pl, first_pl]
    for s in segs:
        r = rec_of(s[2])
        if r is None:
            continue
        if (blocks and blocks[-1][2] == r["team"]
                and s[0] - blocks[-1][1] <= 1.5 * FPS):
            blocks[-1][1] = s[1]
            blocks[-1][3] = r["name"]
        else:
            blocks.append([s[0], s[1], r["team"], r["name"], r["name"]])
    blocks = [b for b in blocks if b[1] - b[0] + 1 >= 2.0 * FPS]
    for b0, b1 in zip(blocks, blocks[1:]):
        t_change = b1[0] / FPS
        if b0[2] == b1[2] or b1[0] - b0[1] > 2.0 * FPS:
            continue
        if any(abs(e["t"] - t_change) <= TO_SHOT_GUARD_S for e in shots):
            continue
        out.append({"t": round(t_change, 1), "classification": "TURNOVER",
                    "player_a": b0[3], "source": "cv", "confidence": 0.4})
        out.append({"t": round(t_change, 1), "classification": "STEAL",
                    "player_a": b1[4], "source": "cv", "confidence": 0.4})
    out.sort(key=lambda e: e["t"])
    outp = REPO / f"runs/tracking/ledger/possession_events_{a.game}.json"
    outp.write_text(json.dumps({"game": a.game, "events": out}, indent=1))

    # ---- score vs GT ----
    plays_doc = json.loads((REPO / a.plays).read_text())
    for cls, tol in (("REBOUND", 2.5), ("TURNOVER", 3.0), ("STEAL", 3.0)):
        gt = [p for p in plays_doc["plays"] if p["cls"] == cls]
        cv = [e for e in out if e["classification"] == cls]
        det = who = 0
        for g in gt:
            cand = [e for e in cv if abs(e["t"] - g["t"]) <= tol]
            if not cand:
                continue
            det += 1
            e = min(cand, key=lambda e: abs(e["t"] - g["t"]))
            who += e["player_a"].split()[-1] == g["a"].split()[-1]
        fp = sum(1 for e in cv if not any(abs(e["t"] - g["t"]) <= tol for g in gt))
        print(f"{cls}: GT {len(gt)} | detected {det} ({det/max(len(gt),1):.0%}) | "
              f"WHO {who}/{det} | CV emitted {len(cv)} (unmatched {fp})")
    print(f"-> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
