#!/usr/bin/env python3
"""Self-contained shot detection + shooter attribution (events v2).

No triangulation (side-angle depth is unreliable — shot-det team's verdict). Uses
OUR ball+hoop specialist cache (far cams see the hoop ~100%). Per far camera:

  1. SHOT EVENT: ball rises and its trajectory peaks near the hoop, then descends
     (an arc terminating at the rim) -> a shot, timestamped at the apex.
  2. RELEASE: ~0.4-0.7s before apex, the ball is at the shooter's hands. At that
     instant the ball sits ON the shooter in the image -> shooter = the identity
     whose box contains / is directly under the ball (image space, no 3-D).
  3. POINTS: the shooter's feet -> court zone (2/3/4PT), reused from detect_events.

Validated against the plays GT (which plays are shots; classification=make/miss;
player_a=shooter). Make/miss itself is scored vs GT here; production verdict later
= the shot-det trained rim model.

  python scripts/detect_shots.py --game e6fba750 --tag 44_60 \
      --tracks-dir runs/events_fg_0_600 --plays data/plays/e6fba750_44_60.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO / "src"))

ANGLES = ("FL", "FR", "NL", "NR")
FAR = ("FL", "FR")                      # far cams see the hoop + whole arc
OFFS = {"e6fba750": {"FL": 0, "FR": -11, "NL": -1, "NR": -1}}
FPS = 29.97


def load_ballhoop(game, tag):
    ball = {ang: {} for ang in ANGLES}
    hoop = {}
    for ang in ANGLES:
        p = REPO / f"runs/ball_cache/{game}_{ang}_{tag}.ball.npz"
        if not p.exists():
            continue
        z = np.load(p)
        cls = z["classes"] if "classes" in z else np.zeros(len(z["scores"]))
        hoops = []
        for b, s, f, c in zip(z["boxes"], z["scores"], z["frame_idx"], cls):
            if int(c) == 1:
                hoops.append(b)
            else:
                f = int(f)
                cy = (b[1] + b[3]) / 2
                if f not in ball[ang] or s > ball[ang][f][1]:
                    ball[ang][f] = ([float(v) for v in b], float(s))
        if hoops:
            hoop[ang] = np.median(np.stack(hoops), axis=0)     # static rim
    return ball, hoop


def find_shots(ball_ang, rim, min_rise=70, near_rim_px=110):
    """Ball-arc apexes near the rim. Returns [(apex_frame, apex_xy)]."""
    if rim is None or not ball_ang:
        return []
    rx, ry = (rim[0] + rim[2]) / 2, (rim[1] + rim[3]) / 2
    fs = sorted(ball_ang)
    cy = {f: (ball_ang[f][0][1] + ball_ang[f][0][3]) / 2 for f in fs}
    cx = {f: (ball_ang[f][0][0] + ball_ang[f][0][2]) / 2 for f in fs}
    shots, i = [], 0
    while i < len(fs):
        f = fs[i]
        # candidate apex: local min of cy (highest point), near rim x/y
        win = [g for g in fs if abs(g - f) <= 8]
        if (cy[f] == min(cy[g] for g in win) and abs(cx[f] - rx) < near_rim_px
                and abs(cy[f] - ry) < near_rim_px):
            # require a real rise before (ball came up) — arc, not a bounce
            before = [g for g in fs if f - 20 <= g < f]
            if before and max(cy[g] for g in before) - cy[f] >= min_rise:
                if not shots or f - shots[-1][0] > 15:      # dedup within 0.5s
                    shots.append((f, (cx[f], cy[f])))
        i += 1
    return shots


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="e6fba750")
    ap.add_argument("--tag", default="44_60")
    ap.add_argument("--tracks-dir", required=True)
    ap.add_argument("--plays", required=True)
    ap.add_argument("--dets-dir", default="runs/dets_cache")
    ap.add_argument("--release-lead", type=float, default=0.5, help="s before apex = release")
    a = ap.parse_args()
    from uball_cc.fusion.homography import load_calib, project_pixels
    key = f"{a.game}_{a.tag}"
    offs = OFFS[a.game]
    zones = json.loads((REPO / "configs/court_zones_court-a.json").read_text())
    roster = json.loads((REPO / f"data/rosters/{a.game}.json").read_text())
    plays_doc = json.loads((REPO / a.plays).read_text())
    clip_t0 = float(plays_doc.get("clip_t0", 0))
    calib = {ang: load_calib(str(REPO / f"configs/calib/{ang}.json")) for ang in ANGLES}
    by_num = defaultdict(list)
    for pr in roster["players"]:
        by_num[pr["num"]].append(pr)

    def name_of(pl):
        n = int("".join(c for c in pl if c.isdigit()))
        c = by_num.get(n, [])
        return c[0]["name"] if len(c) == 1 else (c[0]["name"] + "?" if c else pl)

    ball, hoop = load_ballhoop(a.game, a.tag)
    tracks = defaultdict(dict)
    for p in (REPO / a.tracks_dir).glob(f"{key}__n*__*.json"):
        parts = p.stem.split("__")
        pl, ang = "#" + parts[1][1:], parts[2]
        d = json.loads(p.read_text())["frames"]
        tracks[pl][ang] = {int(f): r["box"] for f, r in d.items() if r.get("present")}
    players = sorted(tracks)

    def court(ang, box):
        (x, y), = project_pixels([((box[0] + box[2]) / 2, box[3])], calib[ang])
        return np.array([x, y])

    # ---- detect shots per far camera ----
    shots = []                              # (ref_frame, ang, apex_xy)
    for ang in FAR:
        for af, axy in find_shots(ball[ang], hoop.get(ang)):
            shots.append((af - offs[ang], ang, af, axy))
    shots.sort()
    print(f"detected {len(shots)} shot-arc events across far cams")

    # ---- attribute each detected shot ----
    def shooter_at(ref_f, ang, apex_f):
        """Shooter = player whose raised hands (top of box) are nearest the ball,
        scanned across the release window [apex-0.8s .. apex-0.1s]. At release the
        ball is at the top of the shooter, so we score proximity to his TOP-center
        and require the ball to sit near his upper body (not a defender's mid-body)."""
        lo = apex_f - int(0.8 * FPS)
        hi = apex_f - int(0.1 * FPS)
        best = None
        for cand in range(lo, hi + 1):
            bb = ball[ang].get(cand)
            if not bb:
                continue
            bx = (bb[0][0] + bb[0][2]) / 2
            by = (bb[0][1] + bb[0][3]) / 2
            for pl in players:
                box = tracks[pl].get(ang, {}).get(cand)
                if box is None:
                    continue
                bw = box[2] - box[0]
                topx, topy = (box[0] + box[2]) / 2, box[1]
                # ball must be within ~1 box-width horizontally and near the top
                # third of the player (hands), i.e. not far below his head
                if abs(bx - topx) > bw and not (box[0] <= bx <= box[2]):
                    continue
                if by > box[1] + 0.55 * (box[3] - box[1]):    # ball below mid-body: skip
                    continue
                d = np.hypot(bx - topx, by - topy)
                if best is None or d < best[0]:
                    best = (d, pl, cand)
        return best

    out = []
    for ref_f, ang, apex_f, apex_xy in shots:
        s = shooter_at(ref_f, ang, apex_f)
        pl = s[1] if s else None
        zone = None
        if pl:
            # feet-zone at release
            box = tracks[pl].get(ang, {}).get(s[2])
            if box is not None:
                pos = court(ang, box)
                dL = np.linalg.norm(pos - np.array(zones["baskets"]["L"]))
                dR = np.linalg.norm(pos - np.array(zones["baskets"]["R"]))
                side = "L" if dL < dR else "R"
                d = min(dL, dR)
                zone = ("2PT" if d < zones["three_pt_r_cm"]
                        else "3PT" if d < zones["four_pt_r_cm"][side] else "4PT")
        out.append({"t": round(ref_f / FPS + clip_t0, 1), "cam": ang,
                    "pred_player": name_of(pl) if pl else None, "pred_zone": zone})

    # ---- score vs GT shots ----
    gt_shots = [p for p in plays_doc["plays"] if any(k in p["cls"] for k in ("MAKE", "MISS"))]
    matched = who_ok = zone_ok = 0
    print(f"\n{'GT t':>7} {'GT class':<12} {'GT player':<20} | {'det?':>4} {'pred player':<20} {'zone':<5}")
    for g in gt_shots:
        cand = [o for o in out if abs(o["t"] - g["t"]) <= 1.5]
        gtz = ("4PT" if g["cls"].startswith("4PT") else "3PT" if g["cls"].startswith("3PT") else "2PT")
        if cand:
            matched += 1
            o = min(cand, key=lambda o: abs(o["t"] - g["t"]))
            wok = (o["pred_player"] and o["pred_player"].rstrip("?").split()[-1] == g["a"].split()[-1])
            zok = (o["pred_zone"] == gtz)
            who_ok += bool(wok); zone_ok += bool(zok)
            print(f"{g['t']:7.1f} {g['cls']:<12} {g['a'][:20]:<20} | {'YES':>4} "
                  f"{str(o['pred_player'])[:20]:<20} {str(o['pred_zone']):<5} "
                  f"{'WHO OK' if wok else ''} {'ZONE OK' if zok else ''}")
        else:
            print(f"{g['t']:7.1f} {g['cls']:<12} {g['a'][:20]:<20} | {'--':>4} (shot not detected)")
    print(f"\nSHOT RECALL: {matched}/{len(gt_shots)} | WHO: {who_ok}/{matched} | ZONE: {zone_ok}/{matched}")
    (REPO / f"runs/tracking/ledger/shots_{key}.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
