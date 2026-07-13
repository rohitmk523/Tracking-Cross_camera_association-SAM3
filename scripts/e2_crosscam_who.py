#!/usr/bin/env python3
"""E2 — cross-camera release vote (docs/PAINT_WHO_PLAN.md).

The production picker (integrated under-ball holder over [rel-0.8s, rel]) runs
on the ARC camera only. E2 runs the SAME picker independently in all four
views at the release instant and takes a margin-weighted vote — a defender
occludes one ray, rarely four. Same 134-shot harness.

  .venv/bin/python scripts/e2_crosscam_who.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")
OFFS = {"FL": 0, "FR": -11, "NL": -1, "NR": -1}
FPS = 29.97
CHUNKS = ("0_600", "600_600", "1200_600", "1800_600", "2400_600", "3000_345")


def load_chunk(game, tag):
    ball = {a: {} for a in ANGLES}
    for a in ANGLES:
        z = np.load(REPO / f"runs/ball_cache/{game}_{a}_{tag}.ball.npz")
        for b, s, f, c in zip(z["boxes"], z["scores"], z["frame_idx"], z["classes"]):
            if int(c) != 0:
                continue
            f = int(f)
            if f not in ball[a] or s > ball[a][f][1]:
                ball[a][f] = ([float(v) for v in b], float(s))
    tr = defaultdict(dict)
    for p in (REPO / f"runs/events_fg_{tag}").glob(f"{game}_{tag}__n*__*.json"):
        parts = p.stem.split("__")
        pl, a = "#" + parts[1][1:], parts[2]
        d = json.loads(p.read_text())["frames"]
        tr[pl][a] = {int(f): r["box"] for f, r in d.items() if r.get("present")}
    return ball, tr


def holder_scores(ball, tr, ang, f0, f1):
    score = defaultdict(float)
    for f in range(f0, f1 + 1):
        bb = ball[ang].get(f)
        if not bb:
            continue
        bx, by = (bb[0][0] + bb[0][2]) / 2, (bb[0][1] + bb[0][3]) / 2
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
            score[pl] += np.exp(-d)
    return score


def main() -> int:
    game = "e6fba750"
    led = json.loads((REPO / f"runs/tracking/ledger/shots_{game}_full.json").read_text())
    plays = json.loads((REPO / f"data/plays/{game}_full.json").read_text())["plays"]
    gt = [p for p in plays if ("MAKE" in p["cls"] or "MISS" in p["cls"])]
    roster = json.loads((REPO / f"data/rosters/{game}.json").read_text())
    name_num = {}
    for pr in roster["players"]:
        name_num.setdefault(pr["name"].split()[-1], pr["num"])
    data = {tag: load_chunk(game, tag) for tag in CHUNKS}

    tot = base_ok = vote_ok = 0
    per = defaultdict(lambda: [0, 0, 0])
    for g in gt:
        cand = [o for o in led if abs(o["t"] - g["t"]) <= 1.5 and o.get("rel_f") is not None]
        if not cand:
            continue
        o = min(cand, key=lambda o: abs(o["t"] - g["t"]))
        ball, tr = data[o["chunk"]]
        rel = o["rel_f"]
        num = name_num.get(g["a"].split()[-1])
        gt_ids = {pl for pl in tr if pl.lstrip("#").rstrip("BW").isdigit()
                  and int(pl.lstrip("#").rstrip("BW")) == num}
        if not gt_ids:
            continue
        tot += 1
        k = g["cls"].split("_")[0] if "FREE" not in g["cls"] else "FT"
        per[k][0] += 1
        wok0 = bool(o["pred_player"] and o["pred_player"].rstrip("?").split()[-1]
                    == g["a"].split()[-1])
        base_ok += wok0
        per[k][1] += wok0

        votes = defaultdict(float)
        for ang in ANGLES:
            lf = rel - OFFS[o["cam"]] + OFFS[ang]
            sc = holder_scores(ball, tr, ang, lf - int(0.8 * FPS), lf)
            if not sc:
                continue
            ranked = sorted(sc.values(), reverse=True)
            margin = ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)
            top = max(sc.items(), key=lambda kv: kv[1])[0]
            w = margin / (ranked[0] + 1e-6)          # confidence in [0,1]
            votes[top] += (2.0 if ang == o["cam"] else 1.0) * w
        pick = max(votes.items(), key=lambda kv: kv[1])[0] if votes else None
        wok1 = pick in gt_ids if pick else wok0
        vote_ok += wok1
        per[k][2] += wok1

    print(f"harness n={tot}")
    print(f"BASELINE (arc cam only): {base_ok}/{tot} ({base_ok/tot:.0%})")
    print(f"E2 cross-cam vote:       {vote_ok}/{tot} ({vote_ok/tot:.0%})")
    print(f"\n{'class':<5} {'n':>3} {'base':>5} {'e2':>4}")
    for k, (n, b, w) in sorted(per.items()):
        print(f"{k:<5} {n:>3} {b:>5} {w:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
