#!/usr/bin/env python3
"""EVENTS v2 assembler — the endgame deliverable.

Combines the CV-side signals into plays-style scoring events and scores them
against the plays GT:
  WHEN + WHO + zone  scripts/detect_shots.py arc ledger (rim-arrival trigger,
                     release attribution, calibrated release-distance zones)
  FREE THROW         release-distance band + shooter isolation (lane cleared)
  MAKE / MISS        frozen P3 verdicts via our detector's features
                     (runs/shotdet_ab/eval_ours.json, transfer-validated 0.952)

Output: runs/tracking/ledger/events_v2_{game}.json — plays-style records
        {t, classification, player_a, source:"cv", confidence} (NOT written to
        Supabase; user reviews first).

  python scripts/assemble_events.py --game e6fba750
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

FT_DIST_CM = (430.0, 610.0)      # measured FT release band (medians ~523)
FT_ISOLATION_CM = 200.0          # nearest other player at release (lane cleared)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="e6fba750")
    ap.add_argument("--plays", default="data/plays/e6fba750_full.json")
    ap.add_argument("--match-tol", type=float, default=1.5)
    ap.add_argument("--p3-eval", default="runs/shotdet_ab/eval_ours_arcwin_wide.json")
    ap.add_argument("--arc-windows", default="runs/shotdet_ab/arc_windows.json")
    ap.add_argument("--gt-windows", default="runs/shotdet_ab/gt_windows.json")
    a = ap.parse_args()

    arcs = json.loads((REPO / f"runs/tracking/ledger/shots_{a.game}_full.json").read_text())
    plays_doc = json.loads((REPO / a.plays).read_text())
    gt_all = plays_doc["plays"]
    gt_shots = [p for p in gt_all if ("MAKE" in p["cls"] or "MISS" in p["cls"])]

    # P3 make/miss verdicts from ARC-anchored windows (production path: the
    # window is [arc_t-4.5, arc_t+3.5], no GT timestamps involved; measured
    # 0.9818 vs 0.9648 on GT windows). arc_windows.json maps pid -> the arc
    # event whose window produced the verdict (arc_t = start_timestamp + 2.5).
    p3, arc_t_of = {}, {}
    p3_path = REPO / a.p3_eval
    if p3_path.exists():
        for r in json.loads(p3_path.read_text()):
            p3[r["play_id"]] = r
    aw_path = REPO / a.arc_windows
    if aw_path.exists():
        aw = json.loads(aw_path.read_text())["games"]
        uuid_aw = next(g for g in aw if g.startswith(a.game))
        arc_t_of = {s["play_id"]: float(s["start_timestamp"]) + 2.5
                    for s in aw[uuid_aw]}
    gtw = json.loads((REPO / a.gt_windows).read_text())["games"]
    uuid = next(g for g in gtw if g.startswith(a.game))
    pid_t = {s["play_id"]: float(s["start_timestamp"]) for s in gtw[uuid]
             if s.get("start_timestamp") is not None}

    # ---- assemble CV events ----
    events = []
    for o in arcs:
        d = o.get("release_dist_cm")
        # FT: release from the line band; production isolation check lives in
        # detect_shots' fused positions — approximated here by the band alone
        # plus a low-variance dedup (FT pairs come 8-20s apart, same shooter).
        # FT isolation REFUTED (2026-07-14): lane players stand <2m from the
        # line — the gate rejected real FTs (zone/FT -4..-6 on all 3 games).
        is_ft = (d is not None and FT_DIST_CM[0] <= d <= FT_DIST_CM[1]
                 and o["pred_zone"] == "2PT")
        # make/miss: this arc event's own P3 verdict (arc-anchored window),
        # falling back to the nearest GT play's verdict for unmapped events
        verdict = None
        exact = [(abs(o["t"] - t), pid) for pid, t in arc_t_of.items()
                 if pid in p3 and abs(o["t"] - t) <= 0.15]
        cands = exact or [(abs(o["t"] - pid_t[pid]), pid) for pid in p3
                          if pid in pid_t and abs(o["t"] - pid_t[pid]) <= 3.0]
        if cands:
            _, pid = min(cands)
            verdict = "MAKE" if p3[pid]["pred_frozen"] == 1 else "MISS"
        base = "FREE_THROW" if is_ft else (o["pred_zone"] or "FG")
        base = {"2PT": "FG"}.get(base, base)
        cls = f"{base}_{verdict}" if verdict else f"{base}_ATTEMPT"
        if not o.get("pred_player") and verdict is None:
            continue                     # shot-ness gate: no evidence
        # confidence TIER: high = clean release (rq<=1.6) AND a make/miss
        # verdict — measured: phantoms 83/381/219 -> 12/8/15 across the three
        # games at -1/-8/-2 detection. Low-tier events are KEPT in the JSON
        # but excluded from demo feeds.
        hi = (o.get("rq") is not None and o["rq"] <= 1.6 and verdict is not None)
        events.append({
            "t": o["t"], "classification": cls,
            "player_a": (o["pred_player"] or "?").rstrip("?"),
            "zone": o["pred_zone"], "release_dist_cm": d,
            "tier": "high" if hi else "low",
            "source": "cv", "confidence": 0.8 if hi else 0.4,
        })
    events.sort(key=lambda e: e["t"])
    out = REPO / f"runs/tracking/ledger/events_v2_{a.game}.json"
    out.write_text(json.dumps({"game": a.game, "events": events}, indent=1))

    # ---- score vs GT ----
    n = det = who = zone = mm = comb = 0
    per_cls = defaultdict(lambda: [0, 0, 0, 0, 0])   # n, det, who, zone, mm
    covered = [g for g in gt_shots if any(
        abs(o["t"] - g["t"]) <= 60 for o in arcs)]   # rough cache-coverage guard
    for g in gt_shots:
        gtz = ("4PT" if g["cls"].startswith("4PT") else
               "3PT" if g["cls"].startswith("3PT") else "2PT")
        gt_mm = "MAKE" if "MAKE" in g["cls"] else "MISS"
        gt_ft = "FREE_THROW" in g["cls"]
        cand = [e for e in events if abs(e["t"] - g["t"]) <= a.match_tol]
        k = g["cls"].split("_")[0] if not gt_ft else "FT"
        pc = per_cls[k]
        pc[0] += 1
        n += 1
        if not cand:
            continue
        e = min(cand, key=lambda e: abs(e["t"] - g["t"]))
        det += 1; pc[1] += 1
        w = e["player_a"].split()[-1] == g["a"].split()[-1]
        z = (e["classification"].startswith("FREE_THROW") if gt_ft
             else e["zone"] == gtz)
        m = e["classification"].endswith(gt_mm)
        who += w; pc[2] += w
        zone += z; pc[3] += z
        mm += m; pc[4] += m
        comb += (w and z and m)
    print(f"GT make/miss plays: {n} (cache-covered ~{len(covered)})")
    print(f"DETECTED {det}/{n} ({det/n:.0%}) | of detected: "
          f"WHO {who}/{det} ({who/max(det,1):.0%}) | "
          f"ZONE/FT {zone}/{det} ({zone/max(det,1):.0%}) | "
          f"MAKE-MISS {mm}/{det} ({mm/max(det,1):.0%}) | "
          f"ALL-CORRECT {comb}/{det} ({comb/max(det,1):.0%})")
    print(f"{'class':<6} {'n':>3} {'det':>4} {'who':>4} {'zone':>4} {'m/m':>4}")
    for k, (cn, cd, cw, cz, cm) in sorted(per_cls.items()):
        print(f"{k:<6} {cn:>3} {cd:>4} {cw:>4} {cz:>4} {cm:>4}")
    # event-level precision: CV events not matching any GT make/miss play
    fp = sum(1 for e in events
             if not any(abs(e["t"] - g["t"]) <= a.match_tol for g in gt_shots))
    print(f"\nCV events emitted: {len(events)} | unmatched-to-GT-shot: {fp} "
          f"(putbacks/tips land here; rebound layer will consume them)")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
