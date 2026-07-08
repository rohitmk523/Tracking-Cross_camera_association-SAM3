#!/usr/bin/env python3
"""Event ANCHORS for possession inference (plan D2-4): human-logged plays (shots,
rebounds, steals, turnovers — with player name + jersey number) mapped onto a clip's
frame timeline and resolved to our fused global identities via jersey numbers.

Anchors constrain the holder decoder: the shooter HELD the ball just before the shot;
a defensive rebound flips possession; a STEAL is a turnover with a known actor. A
rebound followed by a different player's shot forces >=1 pass in between.

Timeline: plays timestamps are on the production video timeline our clips are cut
from (validated on e6: plays 46.5s rebound == action at 47s-start clip open), so
clip_frame = (play_ts - clip_start) * FPS. --validate prints each anchor with its
in-clip time for a human spot-check.

  python scripts/build_anchors.py --game e6fba750 --tag 44_60 --start 44 \
      --worldstate runs/tracking/e6fba750_44_60_worldstate.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FPS = 29.97

# classification -> anchor semantics
SHOT = {"FG_MAKE", "FG_MISS", "3PT_MAKE", "3PT_MISS", "4PT_MAKE", "4PT_MISS"}
POSSESSION_FLIP = {"REBOUND"}          # rebounder holds next
DIRECT_TURNOVER = {"STEAL", "TURNOVER"}
IGNORE = {"FOUL", "TIPOFF", "OTHER", "BLOCK", "FREE_THROW_MAKE", "FREE_THROW_MISS"}


def load_plays(game: str) -> list[dict]:
    """Local snapshot first (data/plays/*.json); full-game exports drop in the same
    format as they land."""
    rows: list[dict] = []
    for p in sorted((REPO / "data/plays").glob("*.json")):
        d = json.loads(p.read_text())
        rows += [r for r in d.get("rows", []) if r.get("gid8") == game]
    # dedupe on (ts, classification)
    seen = set()
    out = []
    for r in sorted(rows, key=lambda r: r["ts"]):
        k = (round(r["ts"], 1), r["classification"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--start", type=float, required=True, help="clip start second on the game video")
    ap.add_argument("--dur", type=float, default=None, help="clip duration s (default: parse from tag)")
    ap.add_argument("--worldstate", required=True)
    ap.add_argument("--pre-window", type=float, default=1.0,
                    help="seconds before a shot in which the shooter must be the holder")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dur = a.dur if a.dur is not None else float(a.tag.split("_")[-1])
    ws = json.loads(Path(a.worldstate).read_text())
    # jersey -> gid (fused roster). Same number can exist on both teams; keep both and
    # let the decoder disambiguate by team when the anchor's team is known.
    by_jersey: dict[int, list] = {}
    for p in ws["players"]:
        if p.get("jersey") is not None:
            by_jersey.setdefault(int(p["jersey"]), []).append(
                {"gid": p["global_id"], "team": p.get("team")})

    anchors = []
    n_unresolved = 0
    for r in load_plays(a.game):
        t_clip = r["ts"] - a.start
        if not (-2.0 <= t_clip <= dur + 2.0):
            continue
        cls = r["classification"]
        if cls in IGNORE:
            continue
        frame = int(round(t_clip * FPS))
        jer = r.get("jersey_a")
        cands = by_jersey.get(int(jer), []) if jer is not None else []
        kind = ("shot" if cls in SHOT else
                "possession_flip" if cls in POSSESSION_FLIP else
                "turnover" if cls in DIRECT_TURNOVER else None)
        if kind is None:
            continue
        if not cands:
            n_unresolved += 1
        anchors.append({
            "frame": frame, "t_clip": round(t_clip, 2), "kind": kind,
            "classification": cls, "player": r.get("player_a"),
            "jersey": jer, "gid_candidates": cands,
            "assist_jersey": r.get("jersey_b"),
            "pre_frames": int(a.pre_window * FPS) if kind == "shot" else 0,
        })

    out = Path(a.out or REPO / f"runs/anchors/{a.game}_{a.tag}.anchors.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"game": a.game, "tag": a.tag, "clip_start": a.start,
                               "fps": FPS, "anchors": anchors}, indent=1))
    print(f"{a.game} {a.tag}: {len(anchors)} anchors "
          f"({sum(1 for x in anchors if x['kind'] == 'shot')} shots, "
          f"{sum(1 for x in anchors if x['kind'] == 'possession_flip')} rebounds, "
          f"{sum(1 for x in anchors if x['kind'] == 'turnover')} turnovers; "
          f"{n_unresolved} without a named gid) -> {out}")
    if a.validate:
        for x in anchors:
            g = ",".join(f"gid{c['gid']}({c['team']})" for c in x["gid_candidates"]) or "UNRESOLVED"
            print(f"  {x['t_clip']:6.1f}s f{x['frame']:5d} {x['classification']:<16} "
                  f"#{x['jersey']} {x['player']} -> {g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
