#!/usr/bin/env python3
"""EVENT ATTRIBUTION v1 — WHO + FROM WHERE for scoring plays (events workstream).

Two-step design from the tracking-data literature: (1) frame-level POSSESSION
model — ball-to-player proximity per camera, cross-camera vote, hysteresis;
(2) event attribution at shot timestamps — the possessor just before the shot is
the shooter (WHO); his court position at release maps through the zone config
(configs/court_zones_court-a.json) to 2/3/4PT (WHERE/points).

v1 uses GT play timestamps as triggers (tests attribution only; fusion-v2 supplies
timestamps in production). Output: plays-style events JSON + per-play comparison.

  python scripts/detect_events.py --game e6fba750 --tag 44_60 \
      --tracks-dir runs/demo_e6_events --plays data/plays/e6fba750_44_60.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO / "src"))

ANGLES = ("FL", "FR", "NL", "NR")
OFFS = {"e6fba750": {"FL": 0, "FR": -11, "NL": -1, "NR": -1}}
FPS = 29.97
POSSESS_EXPAND = 1.35          # player box expansion when testing ball containment
HYSTERESIS = 8                 # frames a new possessor must hold to take possession
SHOT_LOOKBACK = (0.2, 4.0)     # GT timestamps lag the release; look back wide
REBOUND_WINDOW = (0.5, 5.0)    # possession window after a miss


def box_ball_dist(pb, bb):
    """0 if ball center inside expanded player box, else normalized center distance."""
    bx, by = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    cx, cy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
    w, h = (pb[2] - pb[0]) * POSSESS_EXPAND / 2, (pb[3] - pb[1]) * POSSESS_EXPAND / 2
    if abs(bx - cx) <= w and abs(by - cy) <= h:
        return 0.0
    return float(np.hypot((bx - cx) / max(w, 1), (by - cy) / max(h, 1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="e6fba750")
    ap.add_argument("--tag", default="44_60")
    ap.add_argument("--tracks-dir", required=True,
                    help="corrected per-player masklets ({key}__nX__{ang}.json)")
    ap.add_argument("--plays", required=True, help="GT plays json (triggers + truth)")
    ap.add_argument("--dets-dir", default="runs/dets_cache")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from uball_cc.fusion.homography import load_calib, project_pixels
    key = f"{a.game}_{a.tag}"
    offs = OFFS[a.game]
    zones = json.loads((REPO / "configs/court_zones_court-a.json").read_text())
    roster = json.loads((REPO / f"data/rosters/{a.game}.json").read_text())
    plays_doc = json.loads((REPO / a.plays).read_text())
    clip_t0 = float(plays_doc.get("clip_t0", 0))
    calib = {ang: load_calib(str(REPO / f"configs/calib/{ang}.json")) for ang in ANGLES}

    def court(ang, box):
        (x, y), = project_pixels([((box[0] + box[2]) / 2, box[3])], calib[ang])
        return np.array([x, y])

    # ---- ball detections per cam/frame: low-conf ball cache if present ----
    ball = {ang: {} for ang in ANGLES}
    for ang in ANGLES:
        bp = REPO / f"runs/ball_cache/{a.game}_{ang}_{a.tag}.ball.npz"
        if bp.exists():
            z = np.load(bp)
            for b, s, f in zip(z["boxes"], z["scores"], z["frame_idx"]):
                f = int(f)
                if f not in ball[ang] or s > ball[ang][f][1]:
                    ball[ang][f] = ([float(v) for v in b], float(s))
        else:
            z = np.load(REPO / a.dets_dir / f"{key.replace('_', f'_{ang}_', 1)}_small_1280_t0.25.dets.npz")
            for b, s, c, f in zip(z["boxes"], z["scores"], z["classes"], z["frame_idx"]):
                if int(c) == 2:
                    f = int(f)
                    if f not in ball[ang] or s > ball[ang][f][1]:
                        ball[ang][f] = ([float(v) for v in b], float(s))

    # ---- ball trajectory linking: bridge sparse dets (<=1.5s gaps, plausible motion) ----
    # MEASURED (sandbox, 9 GT plays): with 3-12% true ball coverage, interpolated
    # ball evidence HURTS (v1.2 1.5s-gap: 2/9->1/9; v1.3 confirm-only 0.4s: 0/9).
    # Linking stays OFF until ball detection recall improves (more labels+retrain).
    MAX_GAP, MAX_V = 0, 60.0
    for ang in ANGLES:
        fs = sorted(ball[ang])
        for f0, f1 in zip(fs, fs[1:]):
            gap = f1 - f0
            if not (1 < gap <= MAX_GAP):
                continue
            b0, s0 = ball[ang][f0]
            b1, s1 = ball[ang][f1]
            c0 = np.array([(b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2])
            c1 = np.array([(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2])
            if np.linalg.norm(c1 - c0) / gap > MAX_V:
                continue
            for f in range(f0 + 1, f1):
                w = (f - f0) / gap
                ib = [b0[i] + (b1[i] - b0[i]) * w for i in range(4)]
                ball[ang][f] = (ib, min(s0, s1), True)   # interpolated flag
        print(f"{ang}: ball frames {len(fs)} -> {len(ball[ang])} after linking", flush=True)

    # ---- identity boxes per cam/frame from corrected masklets ----
    tracks = defaultdict(dict)                     # pl -> ang -> {cf: box}
    for p in (REPO / a.tracks_dir).glob(f"{key}__n*__*.json"):
        parts = p.stem.split("__")
        pl, ang = "#" + parts[1][1:], parts[2]
        d = json.loads(p.read_text())["frames"]
        tracks[pl][ang] = {int(f): r["box"] for f, r in d.items() if r.get("present")}
    players = sorted(tracks)
    print(f"identities: {players} | ball dets/cam: "
          f"{ {ang: len(ball[ang]) for ang in ANGLES} }")

    # ---- possession per reference frame: per-cam nearest, cross-cam vote ----
    max_f = max(max(v) - offs[ang] for ang in ANGLES for v in [ball[ang]] if v)
    raw_pos = {}
    for f in range(0, max_f + 1):
        votes = Counter()
        for ang in ANGLES:
            bb = ball[ang].get(f + offs[ang])
            if not bb:
                continue
            interp = len(bb) > 2 and bb[2]
            best, bpl = 2.0, None
            for pl in players:
                pb = tracks[pl].get(ang, {}).get(f + offs[ang])
                if pb is None:
                    continue
                d = box_ball_dist(pb, bb[0])
                if d < best:
                    best, bpl = d, pl
            if bpl is None:
                continue
            if interp:
                if best == 0.0:
                    votes[bpl] += 1        # interpolated: confirm only
            else:
                votes[bpl] += 2 if best == 0.0 else 1
        if votes:
            raw_pos[f] = votes.most_common(1)[0][0]
    # hysteresis + STICKY: absence of ball evidence keeps the current possessor
    possess, cur, streak, last_cand = {}, None, 0, None
    for f in range(0, max_f + 1):
        cand = raw_pos.get(f)
        if cand is not None and cand != cur:
            streak = streak + 1 if cand == last_cand else 1
            last_cand = cand
            if streak >= HYSTERESIS:
                cur = cand
        elif cand == cur:
            streak = 0
        if cur is not None:
            possess[f] = cur

    # ---- name <-> identity mapping ----
    by_num = defaultdict(list)
    for pr in roster["players"]:
        by_num[pr["num"]].append(pr)

    def name_of(pl):
        num = int("".join(c for c in pl if c.isdigit()))
        c = by_num.get(num, [])
        return c[0]["name"] if len(c) == 1 else (c[0]["name"] + "?" if c else pl)

    # ---- attribution at play timestamps ----
    out_events, table = [], []
    for play in plays_doc["plays"]:
        t_clip = play["t"] - clip_t0
        f_shot = int(round(t_clip * FPS))
        cls = play["cls"]
        is_shot = any(k in cls for k in ("MAKE", "MISS"))
        if cls in ("REBOUND", "STEAL"):
            # possession lands AT/AFTER these events — look forward
            window = range(f_shot, f_shot + int(2.5 * FPS))
        else:
            window = range(max(0, f_shot - int(SHOT_LOOKBACK[1] * FPS)),
                           max(1, f_shot - int(SHOT_LOOKBACK[0] * FPS)))
        who = Counter(possess[f] for f in window if f in possess)
        pred_pl = who.most_common(1)[0][0] if who else None
        pred_name, pred_zone = None, None
        if pred_pl:
            pred_name = name_of(pred_pl)
            recent = [f for f in window if possess.get(f) == pred_pl]
            # release position: freshest possession within 1.5s of the timestamp
            # (older frames misplace deep shooters -> 4PT under-zoning)
            fresh = [f for f in recent if abs(f - f_shot) <= int(1.5 * FPS)]
            f_rel = max(fresh) if fresh else (max(recent) if recent else None)
            # zone from the shooter's position AT the logged timestamp when his
            # track covers it (deep shooters hold the release spot; old possession
            # frames under-range 4PT)
            if is_shot and any((f_shot + offs[ang]) in tracks[pred_pl].get(ang, {})
                               for ang in ANGLES):
                f_rel = f_shot
            if f_rel is not None:
                pts = [court(ang, tracks[pred_pl][ang][f_rel + offs[ang]])
                       for ang in ANGLES
                       if (f_rel + offs[ang]) in tracks[pred_pl].get(ang, {})]
                if pts:
                    pos = np.mean(pts, axis=0)
                    dL = np.linalg.norm(pos - np.array(zones["baskets"]["L"]))
                    dR = np.linalg.norm(pos - np.array(zones["baskets"]["R"]))
                    side = "L" if dL < dR else "R"
                    d = min(dL, dR)
                    pred_zone = ("2PT" if d < zones["three_pt_r_cm"]
                                 else "3PT" if d < zones["four_pt_r_cm"][side]
                                 else "4PT")
        ev = {"label": cls, "t": play["t"],
              "pred_player": pred_name, "pred_zone": pred_zone if is_shot else None,
              "gt_player": play.get("a"), "confidence": 0.9, "source": "cv"}
        out_events.append(ev)
        who_ok = (pred_name is not None and play.get("a") is not None
                  and pred_name.rstrip("?").split()[-1] == play["a"].split()[-1])
        zone_ok = None
        if is_shot and pred_zone:
            gt_zone = ("4PT" if cls.startswith("4PT") else
                       "3PT" if cls.startswith("3PT") else "2PT")
            zone_ok = pred_zone == gt_zone
        table.append((play["t"], cls, play.get("a"), pred_name, pred_zone, who_ok, zone_ok))

    print(f"\n{'t':>7} {'GT class':<16} {'GT player':<22} {'pred WHO':<22} {'zone':<5} who zone")
    n_who = n_zone = n_shot = 0
    for t, cls, gta, pn, pz, wok, zok in table:
        print(f"{t:7.1f} {cls:<16} {str(gta):<22} {str(pn):<22} {str(pz):<5} "
              f"{'OK' if wok else 'X ':<3} {('OK' if zok else 'X') if zok is not None else '-'}")
        n_who += bool(wok)
        if zok is not None:
            n_shot += 1
            n_zone += bool(zok)
    print(f"\nWHO correct: {n_who}/{len(table)} | zone correct: {n_zone}/{n_shot} shots")
    outp = REPO / (a.out or f"runs/tracking/ledger/events_{key}.json")
    outp.write_text(json.dumps({"window": key, "events": out_events,
                                "who_acc": f"{n_who}/{len(table)}",
                                "zone_acc": f"{n_zone}/{n_shot}"}, indent=1))
    print(f"-> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
