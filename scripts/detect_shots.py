#!/usr/bin/env python3
"""Self-contained shot detection + shooter attribution (events v2), full game.

No triangulation (side-angle depth is unreliable — shot-det team's verdict). Uses
OUR ball+hoop specialist cache (far cams see the hoop ~100%). Per far camera:

  1. SHOT EVENT: ball rises and its trajectory peaks near the hoop, then descends
     (an arc terminating at the rim) -> a shot, timestamped at the apex.
  2. RELEASE: ~0.4-0.7s before apex, the ball is at the shooter's hands. At that
     instant the ball sits ON the shooter in the image -> shooter = the identity
     whose box contains / is directly under the ball (image space, no 3-D).
  3. POINTS: the shooter's feet -> court zone (2/3/4PT), reused from detect_events.

Validated against the plays GT (which plays are shots; classification=make/miss;
player_a=shooter). Make/miss production verdict = frozen P3 via our adapter
(scripts/shotdet_p1_adapter.py + shotdet_transfer_eval.py: 0.952 on e6).

  python scripts/detect_shots.py --game e6fba750 \
      --plays data/plays/e6fba750_full.json --tracks-glob "runs/events_fg_{tag}"
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
from game_meta import GAME_OFFS as OFFS, GAME_CHUNKS
FPS = 29.97



def load_ballhoop(game, tag):
    """Per-chunk ball dict {ang: {frame: (box, score)}} + static hoop median."""
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
                if f not in ball[ang] or s > ball[ang][f][1]:
                    ball[ang][f] = ([float(v) for v in b], float(s))
        if hoops:
            hoop[ang] = np.median(np.stack(hoops), axis=0)     # static rim
    return ball, hoop


def find_shots(ball_ang, rim, arrive_px=100, min_above_px=60):
    """RIM-ARRIVAL events: the ball descends into the rim vicinity after
    having been clearly above it (an arc terminating at this rim). Catches
    long 3PT/4PT arcs whose APEX is far above the rim (the old apex-near-rim
    gate only ever saw short close-range arcs). Returns [(arrive_f, apex_f)]:
    arrival frame + the arc's highest point in the prior ~1.5s."""
    if rim is None or not ball_ang:
        return []
    rx, ry = (rim[0] + rim[2]) / 2, (rim[1] + rim[3]) / 2
    fs = sorted(ball_ang)
    cy = {f: (ball_ang[f][0][1] + ball_ang[f][0][3]) / 2 for f in fs}
    cx = {f: (ball_ang[f][0][0] + ball_ang[f][0][2]) / 2 for f in fs}
    shots = []
    rw = max(rim[2] - rim[0], 20.0)
    inside_prev = {f: abs(cx[f] - rx) < 0.9 * rw and abs(cy[f] - ry) < 0.9 * rw
                   for f in fs}
    for i, f in enumerate(fs):
        trigger = None
        # (a) ARC ARRIVAL: descending into the rim vicinity after a clear rise
        if np.hypot(cx[f] - rx, cy[f] - ry) <= arrive_px:
            prev = [g for g in fs[max(0, i - 5):i] if f - 6 <= g < f]
            before = [g for g in fs if f - 45 <= g < f]
            if prev and cy[prev[-1]] < cy[f] and before:
                apex_f = min(before, key=lambda g: cy[g])
                if (ry - cy[apex_f]) >= min_above_px:
                    trigger = (f, apex_f)
        # (b) RIM-BOX ENTRY: flat layups/putbacks never rise 60px above the
        # rim (measured: ALL 25 missed FGs had the ball IN the rim box) —
        # the fusion repo's own make cue is this box interaction
        if trigger is None and inside_prev[f]:
            recent = [g for g in fs if f - 15 <= g < f]
            if not any(inside_prev[g] for g in recent):
                before = [g for g in fs if f - 45 <= g < f] or [f]
                trigger = (f, min(before, key=lambda g: cy[g]))
        if trigger and (not shots or trigger[0] - shots[-1][0] > 30):
            shots.append(trigger)
    return shots


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="e6fba750")
    ap.add_argument("--plays", default="data/plays/e6fba750_full.json")
    ap.add_argument("--tracks-glob", default="runs/events_fg_{tag}")
    ap.add_argument("--release-lead", type=float, default=0.5)
    ap.add_argument("--match-tol", type=float, default=1.5, help="s, GT match window")
    a = ap.parse_args()
    from uball_cc.fusion.homography import load_calib, project_pixels
    offs = OFFS[a.game]
    zones = json.loads((REPO / "configs/court_zones_court-a.json").read_text())
    roster = json.loads((REPO / f"data/rosters/{a.game}.json").read_text())
    plays_doc = json.loads((REPO / a.plays).read_text())
    clip_t0 = float(plays_doc.get("clip_t0", 0))
    calib = {ang: load_calib(str(REPO / f"configs/calib/{ang}.json")) for ang in ANGLES}
    by_num = defaultdict(list)
    for pr in roster["players"]:
        by_num[pr["num"]].append(pr)

    kit_team = {"B": 1, "W": 2}          # roster: team1_color Black, team2 White
    def name_of(pl):
        n = int("".join(c for c in pl if c.isdigit()))
        c = by_num.get(n, [])
        if len(c) == 1:
            return c[0]["name"]
        kt = kit_team.get(pl[-1]) if pl and pl[-1] in ("B", "W") else None
        if kt is not None:
            m = [p for p in c if p["team"] == kt]
            if len(m) == 1:
                return m[0]["name"]
        return (c[0]["name"] + "?" if c else pl)

    def court(ang, box):
        (x, y), = project_pixels([((box[0] + box[2]) / 2, box[3])], calib[ang])
        return np.array([x, y])

    # ---- per chunk: detect arcs + attribute, emit global-time events ----
    out = []
    for tag in GAME_CHUNKS[a.game]:
        t0_chunk = float(tag.split("_")[0])
        ball, hoop = load_ballhoop(a.game, tag)
        if not any(ball[ang] for ang in FAR):
            print(f"  [{tag}] no ball cache — skipped")
            continue
        key = f"{a.game}_{tag}"
        tracks = defaultdict(dict)
        tdir = REPO / a.tracks_glob.format(tag=tag)
        for p in tdir.glob(f"{key}__n*__*.json"):
            parts = p.stem.split("__")
            pl, ang = "#" + parts[1][1:], parts[2]
            d = json.loads(p.read_text())["frames"]
            tracks[pl][ang] = {int(f): r["box"] for f, r in d.items() if r.get("present")}
        players = sorted(tracks)

        def release_frame(ang, arrive_f, apex_f):
            """Walk BACK from the apex along the final monotone ascent; the
            walk stops where the rise began — the catch/set point, which is at
            the SHOOTER. (Taking the globally lowest ball in a window instead
            latches onto the incoming PASS at the passer's chest — measured:
            FG WHO 21%.)"""
            lo = apex_f - int(0.9 * FPS)        # flight release->apex < 0.9s
            fs = sorted(f for f in ball[ang] if lo <= f <= apex_f)
            if not fs:
                return None
            cy = {f: (ball[ang][f][0][1] + ball[ang][f][0][3]) / 2 for f in fs}
            rel = fs[-1]
            for f in reversed(fs[:-1]):
                if rel - f > 8:                 # detection gap too big — stop
                    break
                if cy[f] >= cy[rel] - 3.0:      # ball was lower (or flat) before
                    rel = f
                else:
                    break                       # rise ends: earlier ball higher
            return rel

        def attribute(arc_ang, rel_f):
            """WHO = the identity that HELD the ball leading into the release,
            not whoever's hands are at the ball at the release instant (a
            contesting defender reaches the ball exactly then — measured: FG
            37% / 3PT 44% with single-frame attribution). Integrate ball
            proximity on the arc cam over [rel-0.8s, rel]: the shooter
            accumulates over the dribble/set; the contester gets one beat."""
            score = defaultdict(float)
            for ang in [arc_ang] + [a for a in ANGLES if a != arc_ang]:
                lf_rel = rel_f - offs[arc_ang] + offs[ang]
                for f in range(lf_rel - int(0.8 * FPS), lf_rel + 1):
                    bb = ball[ang].get(f)
                    if not bb:
                        continue
                    bx = (bb[0][0] + bb[0][2]) / 2
                    by = (bb[0][1] + bb[0][3]) / 2
                    for pl in players:
                        box = tracks[pl].get(ang, {}).get(f)
                        if box is None:
                            continue
                        bw = max(box[2] - box[0], 1.0)
                        if not (box[0] - 0.6 * bw <= bx <= box[2] + 0.6 * bw):
                            continue
                        if by > box[3]:                  # ball below his feet
                            continue
                        topx, topy = (box[0] + box[2]) / 2, box[1]
                        d = np.hypot(bx - topx, by - topy) / bw
                        score[(pl, ang)] += np.exp(-d)
                if score:                # arc cam produced evidence — use it
                    break
            if not score:
                return None
            (pl, ang), _ = max(score.items(), key=lambda kv: kv[1])
            return (0.0, pl, ang)

        def fused_feet(pl, arc_ang, rel_f):
            """Shooter court position = median over ALL cams tracking him at
            release (near cams project feet accurately; far cams at range do
            not — measured zone acc 94% near vs 39% far)."""
            pts = []
            for ang in ANGLES:
                lf = rel_f - offs[arc_ang] + offs[ang]
                for df in (0, -1, 1, -2, 2):
                    box = tracks[pl].get(ang, {}).get(lf + df)
                    if box is not None:
                        pts.append(court(ang, box))
                        break
            return np.median(np.stack(pts), axis=0) if pts else None

        n_chunk = 0
        for arc_ang in FAR:
            for arrive_f, apex_f in find_shots(ball[arc_ang], hoop.get(arc_ang)):
                ref_f = arrive_f - offs[arc_ang]
                rel_f = release_frame(arc_ang, arrive_f, apex_f)
                s = None
                if rel_f is not None:
                    for nudge in (0, 4, 8):     # if ball too low (gather), a
                        s = attribute(arc_ang, rel_f + nudge)   # beat later it
                        if s:                   # is at the shooter's hands
                            break
                pl, att_ang = (s[1], s[2]) if s else (None, None)
                zone, dist = None, None
                if pl:
                    # farthest fused position across the set/release beats — a
                    # shooter drifts INWARD after release, never outward
                    ds = []
                    for back in (0, 6, 12):
                        pos = fused_feet(pl, arc_ang, rel_f - back)
                        if pos is None:
                            continue
                        dL = np.linalg.norm(pos - np.array(zones["baskets"]["L"]))
                        dR = np.linalg.norm(pos - np.array(zones["baskets"]["R"]))
                        ds.append(min(dL, dR))
                    if ds:
                        dist = float(max(ds))
                        rb = zones.get("release_zone_b_cm")
                        if rb:      # empirically calibrated release boundaries
                            zone = ("2PT" if dist < rb[0]
                                    else "3PT" if dist < rb[1] else "4PT")
                        else:       # geometric court-line radii (compress at range)
                            side = "L" if dL < dR else "R"
                            zone = ("2PT" if dist < zones["three_pt_r_cm"]
                                    else "3PT" if dist < zones["four_pt_r_cm"][side]
                                    else "4PT")
                out.append({"t": round(t0_chunk + ref_f / FPS + clip_t0, 1),
                            "cam": arc_ang, "att_cam": att_ang, "chunk": tag,
                            "arrive_f": int(arrive_f), "apex_f": int(apex_f),
                            "rel_f": int(rel_f) if rel_f is not None else None,
                            "pred_player": name_of(pl) if pl else None,
                            "pred_zone": zone,
                            "release_dist_cm": round(dist, 1) if dist else None})
                n_chunk += 1
        print(f"  [{tag}] {n_chunk} arc events")
    out.sort(key=lambda o: o["t"])
    print(f"detected {len(out)} shot-arc events across far cams (full game)")

    # ---- score vs GT shots ----
    gt_shots = [p for p in plays_doc["plays"] if any(k in p["cls"] for k in ("MAKE", "MISS"))]
    matched = who_ok = zone_ok = 0
    ft_like = 0
    print(f"\n{'GT t':>7} {'GT class':<16} {'GT player':<20} | {'det?':>4} {'pred player':<20} {'zone':<5}")
    for g in gt_shots:
        cand = [o for o in out if abs(o["t"] - g["t"]) <= a.match_tol]
        gtz = ("4PT" if g["cls"].startswith("4PT") else "3PT" if g["cls"].startswith("3PT")
               else "2PT")
        if cand:
            matched += 1
            o = min(cand, key=lambda o: abs(o["t"] - g["t"]))
            wok = (o["pred_player"] and g["a"]
                   and o["pred_player"].rstrip("?").split()[-1] == g["a"].split()[-1])
            zok = (o["pred_zone"] == gtz)
            who_ok += bool(wok); zone_ok += bool(zok)
            flag = "WHO OK" if wok else ""
            print(f"{g['t']:7.1f} {g['cls']:<16} {str(g['a'])[:20]:<20} | {'YES':>4} "
                  f"{str(o['pred_player'])[:20]:<20} {str(o['pred_zone']):<5} "
                  f"{flag} {'ZONE OK' if zok else ''}")
        else:
            ft_like += "FREE_THROW" in g["cls"]
            print(f"{g['t']:7.1f} {g['cls']:<16} {str(g['a'])[:20]:<20} | {'--':>4} (not detected)")
    print(f"\nSHOT RECALL: {matched}/{len(gt_shots)} "
          f"({ft_like} of the misses are free throws) | "
          f"WHO: {who_ok}/{matched} ({who_ok/max(matched,1):.0%}) | "
          f"ZONE: {zone_ok}/{matched} ({zone_ok/max(matched,1):.0%})")
    led = REPO / "runs/tracking/ledger"
    led.mkdir(parents=True, exist_ok=True)
    (led / f"shots_{a.game}_full.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
