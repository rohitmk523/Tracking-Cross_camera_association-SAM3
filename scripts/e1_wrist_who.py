#!/usr/bin/env python3
"""E1 — wrist attribution (docs/PAINT_WHO_PLAN.md).

At each detected shot's release instant, run RTMPose on the candidate players
in the arc-cam frame and score ball-to-WRIST distance (the shooter's wrists
are AT the ball; the box-top proxy ties with contesting defenders). Scored on
the same 134-shot harness as E0. Adopt gate: FG WHO +5pts, no 4PT/FT loss.

  .venv/bin/python scripts/e1_wrist_who.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")
FPS = 29.97
WRIST_IDX = (9, 10)
KP_MIN = 0.3


def main() -> int:
    game = "e6fba750"
    led = json.loads((REPO / f"runs/tracking/ledger/shots_{game}_full.json").read_text())
    plays = json.loads((REPO / f"data/plays/{game}_full.json").read_text())["plays"]
    gt = [p for p in plays if ("MAKE" in p["cls"] or "MISS" in p["cls"])]
    roster = json.loads((REPO / f"data/rosters/{game}.json").read_text())
    name_num = {}
    for pr in roster["players"]:
        name_num.setdefault(pr["name"].split()[-1], pr["num"])

    from rtmlib import RTMPose
    pose = RTMPose(
        onnx_model="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
                   "onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
        model_input_size=(192, 256), backend="onnxruntime", device="mps")

    balls, trks = {}, {}
    def chunk_data(tag):
        if tag not in balls:
            b = {a: {} for a in ANGLES}
            for a in ANGLES:
                z = np.load(REPO / f"runs/ball_cache/{game}_{a}_{tag}.ball.npz")
                for bx, s, f, c in zip(z["boxes"], z["scores"], z["frame_idx"], z["classes"]):
                    if int(c) != 0:
                        continue
                    f = int(f)
                    if f not in b[a] or s > b[a][f][1]:
                        b[a][f] = ([float(v) for v in bx], float(s))
            t = defaultdict(dict)
            for p in (REPO / f"runs/events_fg_{tag}").glob(f"{game}_{tag}__n*__*.json"):
                parts = p.stem.split("__")
                pl, a = "#" + parts[1][1:], parts[2]
                d = json.loads(p.read_text())["frames"]
                t[pl][a] = {int(f): r["box"] for f, r in d.items() if r.get("present")}
            balls[tag], trks[tag] = b, t
        return balls[tag], trks[tag]

    tot = base_ok = wrist_ok = 0
    per = defaultdict(lambda: [0, 0, 0])
    for g in gt:
        cand = [o for o in led if abs(o["t"] - g["t"]) <= 1.5 and o.get("rel_f") is not None]
        if not cand:
            continue
        o = min(cand, key=lambda o: abs(o["t"] - g["t"]))
        clip = REPO / f"runs/e1_frames/{o['chunk']}_{o['rel_f']}_{o['cam']}.mp4"
        if not clip.exists():
            continue
        ball, tr = chunk_data(o["chunk"])
        ang, rel = o["cam"], o["rel_f"]
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

        cap = cv2.VideoCapture(str(clip))
        best = None                       # (norm dist, pl)
        big = False                       # any candidate tall enough for pose
        fi = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            f = rel - 3 + fi
            fi += 1
            bb = ball[ang].get(f)
            if not bb:
                continue
            bx, by = (bb[0][0] + bb[0][2]) / 2, (bb[0][1] + bb[0][3]) / 2
            cands = []
            for pl in tr:
                box = tr[pl].get(ang, {}).get(f)
                if box is None:
                    continue
                bw = max(box[2] - box[0], 1.0)
                if not (box[0] - bw <= bx <= box[2] + bw) or by > box[3]:
                    continue
                # HYBRID: pose is only reliable on big bodies (~100px+); below
                # that the box-top baseline already wins (measured: 4PT -5)
                if box[3] - box[1] >= 100:
                    cands.append((pl, box))
            if not cands:
                continue
            big = True
            kpts, ks = pose(img, bboxes=[b for _, b in cands])
            for (pl, box), kp, s in zip(cands, kpts, ks):
                bw = max(box[2] - box[0], 1.0)
                for wi in WRIST_IDX:
                    if s[wi] < KP_MIN:
                        continue
                    d = float(np.hypot(kp[wi][0] - bx, kp[wi][1] - by)) / bw
                    if best is None or d < best[0]:
                        best = (d, pl)
        cap.release()
        if big and best is not None and best[0] < 0.6:
            wok1 = best[1] in gt_ids      # wrist ON the ball: confident override
        else:
            wok1 = wok0                   # otherwise the baseline pick stands
        wrist_ok += wok1
        per[k][2] += wok1

    print(f"harness n={tot}")
    print(f"BASELINE (current picker): {base_ok}/{tot} ({base_ok/tot:.0%})")
    print(f"E1 WRIST (arc cam):        {wrist_ok}/{tot} ({wrist_ok/tot:.0%})")
    print(f"\n{'class':<5} {'n':>3} {'base':>5} {'wrist':>6}")
    for k, (n, b, w) in sorted(per.items()):
        print(f"{k:<5} {n:>3} {b:>5} {w:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
