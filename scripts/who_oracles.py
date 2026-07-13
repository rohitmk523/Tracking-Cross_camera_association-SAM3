#!/usr/bin/env python3
"""E0 — oracle ceilings for crowd/paint WHO (docs/PAINT_WHO_PLAN.md).

For every GT shot matched to a detected event, measure what a PERFECT version
of each signal family could recover:
  oracle-track   GT shooter's track exists near the ball in the release window
                 (arc cam / ANY cam) -> the pure-logic ceiling
  oracle-cam     a single-frame under-ball pick on ANY camera at release = GT
                 -> the cross-cam-vote (E2) ceiling
  oracle-window  any single frame within +-0.3s of release on the arc cam
                 picks GT -> the timing ceiling

  .venv/bin/python scripts/who_oracles.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")
from game_meta import GAME_OFFS, GAME_CHUNKS
FPS = 29.97


def load_chunk(game, tag):
    ball = {a: {} for a in ANGLES}
    for a in ANGLES:
        z = np.load(REPO / f"runs/ball_cache/{game}_{a}_{tag}.ball.npz")
        cls = z["classes"]
        for b, s, f, c in zip(z["boxes"], z["scores"], z["frame_idx"], cls):
            if int(c) != 0:
                continue
            f = int(f)
            if f not in ball[a] or s > ball[a][f][1]:
                ball[a][f] = ([float(v) for v in b], float(s))
    tr = defaultdict(dict)
    tdir = REPO / (f"runs/events_fg_{tag}" if game == "e6fba750"
                   else f"runs/events_fg_{game[:3]}_{tag}")
    for p in tdir.glob(f"{game}_{tag}__n*__*.json"):
        parts = p.stem.split("__")
        pl, a = "#" + parts[1][1:], parts[2]
        d = json.loads(p.read_text())["frames"]
        tr[pl][a] = {int(f): r["box"] for f, r in d.items() if r.get("present")}
    return ball, tr


def pick(ball, tr, ang, f):
    """Single-frame under-ball candidate (same gates as production)."""
    bb = ball[ang].get(f)
    if not bb:
        return None
    bx, by = (bb[0][0] + bb[0][2]) / 2, (bb[0][1] + bb[0][3]) / 2
    best = None
    for pl in tr:
        box = tr[pl].get(ang, {}).get(f)
        if box is None:
            continue
        bw = max(box[2] - box[0], 1.0)
        if not (box[0] - 0.6 * bw <= bx <= box[2] + 0.6 * bw) or by > box[3]:
            continue
        if by > box[1] + 0.6 * (box[3] - box[1]):
            continue
        d = np.hypot(bx - (box[0] + box[2]) / 2, by - box[1]) / bw
        if best is None or d < best[0]:
            best = (d, pl)
    return best[1] if best else None


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="e6fba750")
    a = ap.parse_args()
    game = a.game
    OFFS = GAME_OFFS[game]
    globals()["OFFS"] = OFFS
    led = json.loads((REPO / f"runs/tracking/ledger/shots_{game}_full.json").read_text())
    plays = json.loads((REPO / f"data/plays/{game}_full.json").read_text())["plays"]
    gt = [p for p in plays if ("MAKE" in p["cls"] or "MISS" in p["cls"])]
    roster = json.loads((REPO / f"data/rosters/{game}.json").read_text())
    name_num = {}
    for pr in roster["players"]:
        name_num.setdefault(pr["name"].split()[-1], pr["num"])
    data = {tag: load_chunk(game, tag) for tag in GAME_CHUNKS[game]}

    tot = cur_ok = o_trk_arc = o_trk_any = o_cam = o_win = o_num = 0
    per = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    for g in gt:
        cand = [o for o in led if abs(o["t"] - g["t"]) <= 1.5 and o.get("rel_f") is not None]
        if not cand:
            continue
        o = min(cand, key=lambda o: abs(o["t"] - g["t"]))
        ball, tr = data[o["chunk"]]
        num = name_num.get(g["a"].split()[-1])
        rec = next((p for p in roster["players"]
                    if p["name"].split()[-1] == g["a"].split()[-1]), None)
        dual = sum(1 for p in roster["players"] if p["num"] == num) > 1
        want_kit = ("B" if rec["team"] == 1 else "W") if (rec and dual) else None
        num_ids, gt_ids = set(), set()
        for pl in tr:
            core = pl.lstrip("#")
            kitc = core[-1] if core[-1] in ("B", "W") else None
            dg = core.rstrip("BW")
            if not dg.isdigit() or int(dg) != num:
                continue
            num_ids.add(pl)
            if want_kit is None or kitc in (None, want_kit):
                gt_ids.add(pl)
        if not gt_ids:
            continue
        tot += 1
        k = g["cls"].split("_")[0] if "FREE" not in g["cls"] else "FT"
        row = per[k]
        row[0] += 1
        wok = bool(o["pred_player"] and o["pred_player"].rstrip("?").split()[-1]
                   == g["a"].split()[-1])
        cur_ok += wok
        row[1] += wok
        rel = o["rel_f"]

        def near_gt(ang, f):
            bb = ball[ang].get(f)
            if not bb:
                return False
            bx, by = (bb[0][0] + bb[0][2]) / 2, (bb[0][1] + bb[0][3]) / 2
            for pl in gt_ids:
                box = tr[pl].get(ang, {}).get(f)
                if box is None:
                    continue
                bw = max(box[2] - box[0], 1.0)
                if box[0] - 1.5 * bw <= bx <= box[2] + 1.5 * bw and by <= box[3]:
                    return True
            return False

        win = range(rel - 9, rel + 10)                    # +-0.3s
        ta = any(near_gt(o["cam"], f) for f in win)
        o_trk_arc += ta
        row[2] += ta
        tn = ta or any(near_gt(a2, rel - OFFS[o["cam"]] + OFFS[a2] + d)
                       for a2 in ANGLES for d in (-9, -4, 0, 4, 9))
        o_trk_any += tn
        row[3] += tn
        hold = gt_ids
        gt_ids = num_ids                 # number-level presence (any kit)
        o_num += any(near_gt(a2, rel - OFFS[o["cam"]] + OFFS[a2] + d)
                     for a2 in ANGLES for d in (-9, -4, 0, 4, 9))
        gt_ids = hold
        names = {pl: pl for pl in tr}
        def is_gt(pl):
            return pl in gt_ids
        oc = any(is_gt(pick(ball, tr, a, rel - OFFS[o["cam"]] + OFFS[a]))
                 for a in ANGLES)
        o_cam += oc
        row[4] += oc
        ow = any(is_gt(pick(ball, tr, o["cam"], f)) for f in win)
        o_win += ow
        row[5] += ow

    def pc(x):
        return f"{x}/{tot} ({x/tot:.0%})"
    print(f"harness n={tot} | CURRENT WHO {pc(cur_ok)}")
    print(f"oracle-track arc-cam {pc(o_trk_arc)} | ANY cam {pc(o_trk_any)}")
    print(f"oracle-NUM-track any cam (any kit) {pc(o_num)}")
    print(f"oracle-cam  (E2 bound) {pc(o_cam)}")
    print(f"oracle-win  (+-0.3s, arc cam) {pc(o_win)}")
    print(f"\n{'class':<5} {'n':>3} {'cur':>4} {'trkA':>5} {'trk*':>5} {'cam':>4} {'win':>4}")
    for k, r in sorted(per.items()):
        print(f"{k:<5} {r[0]:>3} {r[1]:>4} {r[2]:>5} {r[3]:>5} {r[4]:>4} {r[5]:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
