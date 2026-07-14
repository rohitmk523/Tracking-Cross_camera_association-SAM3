#!/usr/bin/env python3
"""BALL-FIRST prototype (docs/BALL_FIRST_SPEC.md) on selected hard cases.

Per case window: 4-cam ball detections -> HOLD/FLIGHT/RIM segmentation ->
holder state machine (cross-cam vote + hysteresis) -> event transitions.
Prints the ball STORY and checks the shot's WHO (= last holder) vs GT.
Renders a yellow-holder 4-angle clip per case.

  .venv/bin/python scripts/ballfirst_proto.py --cases 0,4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from game_meta import GAME_OFFS

FPS = 29.97
ANGLES = ("FL", "FR", "NL", "NR")
HOLD_D = 0.7          # ball within this many box-widths of a stream = touching
SWITCH_K = 6          # frames of consistent new holder to switch (hysteresis)
FLIGHT_V = 14.0       # px/frame ball speed = flight
W, H = 1600, 900
CELL_W, CELL_H = 620, 450


def name_of(roster_players, by_num, sid):
    core = sid.lstrip("#").lstrip("n")
    kit = core[-1] if core[-1] in ("B", "W") else None
    num = int(core.rstrip("BW"))
    c = by_num.get(num, [])
    if len(c) == 1:
        return f"#{num} {c[0]['name'].split()[-1]}"
    m = [p for p in c if p["team"] == {"B": 1, "W": 2}.get(kit)]
    return f"#{num} {m[0]['name'].split()[-1]}" if len(m) == 1 else f"#{num}?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="0,4")
    a = ap.parse_args()
    sel = json.loads(Path("/tmp/unsolv_sel.json").read_text())
    for ci in [int(x) for x in a.cases.split(",")]:
        s = sel[ci]
        game, tag, offs = s["game"], s["chunk"], GAME_OFFS[s["game"]]
        chunk0 = float(tag.split("_")[0])
        t_rel = chunk0 + s["rel"] / FPS
        lo_t, hi_t = t_rel - 6.0, t_rel + 2.0
        roster = json.loads((REPO / f"data/rosters/{game}.json").read_text())
        by_num = defaultdict(list)
        for p in roster["players"]:
            by_num[p["num"]].append(p)

        ball, hoop, tracks = {}, {}, defaultdict(dict)
        for ang in ANGLES:
            z = np.load(REPO / f"runs/ball_cache/{game}_{ang}_{tag}.ball.npz")
            bd, hp = {}, []
            for b, sc, f, c in zip(z["boxes"], z["scores"], z["frame_idx"], z["classes"]):
                if int(c) == 1:
                    hp.append(b)
                else:
                    f = int(f)
                    if f not in bd or sc > bd[f][1]:
                        bd[f] = ([float(v) for v in b], float(sc))
            ball[ang] = bd
            hoop[ang] = np.median(np.stack(hp), axis=0) if hp else None
        tglob = s["tglob"]
        for p in (REPO / tglob.format(tag=tag)).glob(f"{game}_{tag}__n*__*.json"):
            parts = p.stem.split("__")
            d = json.loads(p.read_text())["frames"]
            tracks[parts[1]][parts[2]] = {int(fr): r["box"]
                                          for fr, r in d.items() if r.get("present")}

        # ---- per ref-frame observation: cross-cam holder votes + flight ----
        f_lo = round((lo_t - chunk0) * FPS)
        f_hi = round((hi_t - chunk0) * FPS)
        obs = []                       # (state, holder_votes dict, rim_flag)
        for f in range(f_lo, f_hi):
            votes = defaultdict(float)
            speeds, rim_near = [], False
            for ang in ANGLES:
                cf = f + offs[ang]
                bb = ball[ang].get(cf)
                if not bb:
                    continue
                bx, by = (bb[0][0] + bb[0][2]) / 2, (bb[0][1] + bb[0][3]) / 2
                prev = ball[ang].get(cf - 3)
                if prev:
                    px, py = (prev[0][0] + prev[0][2]) / 2, (prev[0][1] + prev[0][3]) / 2
                    speeds.append(np.hypot(bx - px, by - py) / 3)
                hb = hoop[ang]
                if hb is not None and np.hypot(bx - (hb[0]+hb[2])/2, by - (hb[1]+hb[3])/2) < 90:
                    rim_near = True
                for sid, angs in tracks.items():
                    box = angs.get(ang, {}).get(cf)
                    if box is None or by > box[3]:
                        continue
                    bw = max(box[2] - box[0], 1.0)
                    dx = abs(bx - (box[0] + box[2]) / 2) / bw
                    if dx <= HOLD_D and by >= box[1] - 0.1 * (box[3] - box[1]):
                        votes[sid] += (1.0 - 0.5 * dx)
            v = np.median(speeds) if speeds else None
            state = ("RIM" if rim_near else
                     "FLIGHT" if (v is not None and v > FLIGHT_V and not votes) else
                     "HOLD" if votes else "GAP")
            obs.append((f, state, dict(votes)))

        # ---- holder state machine with hysteresis ----
        story, holder, cand, streak = [], None, None, 0
        since_flight = 99
        for f, state, votes in obs:
            since_flight = 0 if state in ("FLIGHT", "GAP", "RIM") else since_flight + 1
            if state == "HOLD" and votes:
                top = max(votes, key=votes.get)
                # CATCH rule: a hold right after flight IS a transition —
                # switch fast (3 frames); mid-hold contests keep slow hysteresis
                k_need = 3 if since_flight <= 4 else SWITCH_K
                if top == holder:
                    streak = 0
                else:
                    streak = streak + 1 if top == cand else 1
                    cand = top
                    if streak >= k_need:
                        if holder is not None:
                            story.append(("PASS/TRANSITION", holder, top, f))
                        holder = top
                        streak = 0
                if not story or story[-1][0] != "HOLD" or story[-1][1] != holder:
                    story.append(("HOLD", holder, None, f))
            elif state == "RIM":
                story.append(("RIM", holder, None, f))
        # collapse story
        print(f"\n===== CASE {ci+1}: {game} t={s['t']} GT: {s['gt_cls']} by {s['gt']} "
              f"(pipeline had said: {s['pred']}) =====")
        last = None
        holder_at_rim = None
        for ev, h, h2, f in story:
            t = chunk0 + f / FPS
            nm = name_of(roster, by_num, h) if h else "?"
            if ev == "RIM" and last != "RIM":
                print(f"  {t:7.1f}s  BALL AT RIM — last holder: {nm}")
                holder_at_rim = h
            elif ev == "PASS/TRANSITION":
                print(f"  {t:7.1f}s  {name_of(roster, by_num, h)} -> {name_of(roster, by_num, h2)}")
            elif ev == "HOLD" and last != "HOLD":
                print(f"  {t:7.1f}s  HOLD {nm}")
            last = ev
        who = name_of(roster, by_num, holder_at_rim) if holder_at_rim else "?"
        gt_last = s["gt"].split()[-1]
        verdict = "CORRECT" if gt_last in who else "WRONG"
        print(f"  => STATE-MACHINE WHO: {who}  [{verdict} vs GT {s['gt']}]")

        # ---- yellow-holder render ----
        holder_by_f = {}
        cur = None
        si = 0
        for f, state, votes in obs:
            while si < len(story) and story[si][3] <= f:
                if story[si][0] in ("HOLD", "PASS/TRANSITION"):
                    cur = story[si][2] or story[si][1]
                si += 1
            holder_by_f[f] = cur
        caps = {ang: cv2.VideoCapture(str(REPO / f"runs/event_demo/clips_unsolv/u{ci:02d}_{ang}.mp4"))
                for ang in ANGLES}
        tmp = str(REPO / f"runs/event_demo/ballfirst_case{ci+1}.mp4.raw.mp4")
        vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for fi in range(int(10 * FPS)):
            canvas = np.zeros((H, W, 3), np.uint8)
            canvas[:, 1240:] = (24, 24, 28)
            f = f_lo + fi
            for gi, ang in enumerate(ANGLES):
                ok, frame = caps[ang].read()
                if not ok:
                    continue
                fh, fw = frame.shape[:2]
                cell = cv2.resize(frame, (CELL_W, CELL_H))
                sx, sy = CELL_W / fw, CELL_H / fh
                cf = f + offs[ang]
                hcur = holder_by_f.get(f)
                for sid, angs in tracks.items():
                    box = angs.get(ang, {}).get(cf)
                    if box is None:
                        continue
                    if sid == hcur:
                        cv2.rectangle(cell, (int(box[0]*sx), int(box[1]*sy)),
                                      (int(box[2]*sx), int(box[3]*sy)), (0, 230, 255), 4)
                        cv2.putText(cell, "HOLDER " + name_of(roster, by_num, sid),
                                    (int(box[0]*sx), max(14, int(box[1]*sy)-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 2)
                    else:
                        cv2.rectangle(cell, (int(box[0]*sx), int(box[1]*sy)),
                                      (int(box[2]*sx), int(box[3]*sy)), (160, 160, 160), 1)
                bb = ball[ang].get(cf)
                if bb:
                    cv2.circle(cell, (int((bb[0][0]+bb[0][2])/2*sx),
                                      int((bb[0][1]+bb[0][3])/2*sy)), 7, (0, 165, 255), -1)
                hb = hoop[ang]
                if hb is not None:
                    cv2.rectangle(cell, (int(hb[0]*sx), int(hb[1]*sy)),
                                  (int(hb[2]*sx), int(hb[3]*sy)), (0, 0, 230), 2)
                x0, y0 = (gi % 2) * CELL_W, (gi // 2) * CELL_H
                canvas[y0:y0+CELL_H, x0:x0+CELL_W] = cell
                cv2.putText(canvas, ang, (x0+10, y0+24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (205, 205, 210), 2)
            x = 1256
            hcur = holder_by_f.get(f)
            cv2.putText(canvas, f"BALL-FIRST case {ci+1}", (x, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)
            cv2.putText(canvas, "HOLDER:", (x, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (170, 170, 175), 1)
            cv2.putText(canvas, name_of(roster, by_num, hcur) if hcur else "-",
                        (x, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2)
            cv2.putText(canvas, "GT: " + s["gt_cls"].replace("_", " "), (x, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 210), 1)
            cv2.putText(canvas, s["gt"][:24], (x, 214), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 210), 1)
            vw.write(canvas)
        vw.release()
        for c in caps.values():
            c.release()
        out = REPO / f"runs/event_demo/ballfirst_case{ci+1}.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", tmp,
                        "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p",
                        "-y", str(out)], check=True)
        Path(tmp).unlink()
        print(f"  render -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
