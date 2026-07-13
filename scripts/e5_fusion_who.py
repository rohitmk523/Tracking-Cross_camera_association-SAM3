#!/usr/bin/env python3
"""E5 — fusion of the measured WHO signals (docs/PAINT_WHO_PLAN.md).

Rule family (simple, arbitrated by release distance; D_CLOSE fit on the FIRST
half of the game only, held out on the second):
  track-absent           -> E3 KPR recovery pick (if available)
  release_dist < D_CLOSE -> E1 wrist confident-override, else E2 vote
  else                   -> E2 cross-cam vote

  .venv/bin/python scripts/e5_fusion_who.py --e3-log <stageB output file>
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")
from game_meta import GAME_OFFS, GAME_CHUNKS
FPS = 29.97


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
    tdir = REPO / (f"runs/events_fg_{tag}" if game == "e6fba750" else f"runs/events_fg_{game[:3]}_{tag}")
    for p in tdir.glob(f"{game}_{tag}__n*__*.json"):
        parts = p.stem.split("__")
        pl, a = "#" + parts[1][1:], parts[2]
        d = json.loads(p.read_text())["frames"]
        tr[pl][a] = {int(f): r["box"] for f, r in d.items() if r.get("present")}
    return ball, tr


def holder_scores(ball, tr, ang, f0, f1):
    sc = defaultdict(float)
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
            sc[pl] += np.exp(-d)
    return sc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e3-log", default=None)
    ap.add_argument("--game", default="e6fba750")
    ap.add_argument("--clips-dir", default="runs/e1_frames")
    ap.add_argument("--d-close", type=float, default=None,
                    help="freeze D_CLOSE (blind runs); omit to fit on first half")
    a = ap.parse_args()
    game = a.game
    OFFS = GAME_OFFS[game]
    led = json.loads((REPO / f"runs/tracking/ledger/shots_{game}_full.json").read_text())
    plays = json.loads((REPO / f"data/plays/{game}_full.json").read_text())["plays"]
    gt = [p for p in plays if ("MAKE" in p["cls"] or "MISS" in p["cls"])]
    roster = json.loads((REPO / f"data/rosters/{game}.json").read_text())
    name_num = {}
    for pr in roster["players"]:
        name_num.setdefault(pr["name"].split()[-1], pr["num"])
    data = {tag: load_chunk(game, tag) for tag in GAME_CHUNKS[game]}

    # E3 picks in GT order of track-absent queries (stage A ordering == gt order)
    e3_picks = []
    if a.e3_log and Path(a.e3_log).exists():
        for m in re.finditer(r"query gt=#\s*(\d+) -> pick #?(\d+)", Path(a.e3_log).read_text()):
            e3_picks.append((m.group(1), m.group(2)))
    e3_iter = iter(e3_picks)

    import sys
    sys.path.insert(0, str(REPO / "src"))
    from uball_cc.fusion.homography import load_calib, project_pixels
    calib = {aa: load_calib(str(REPO / f"configs/calib/{aa}.json")) for aa in ANGLES}
    zcfg = json.loads((REPO / "configs/court_zones_court-a.json").read_text())

    from rtmlib import RTMPose
    pose = RTMPose(
        onnx_model="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
                   "onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
        model_input_size=(192, 256), backend="onnxruntime", device="mps")

    rows = []          # per shot: dict of signal-correct flags + meta
    for g in gt:
        cand = [o for o in led if abs(o["t"] - g["t"]) <= 1.5 and o.get("rel_f") is not None]
        if not cand:
            continue
        o = min(cand, key=lambda o: abs(o["t"] - g["t"]))
        ball, tr = data[o["chunk"]]
        ang, rel = o["cam"], o["rel_f"]
        num = name_num.get(g["a"].split()[-1])
        rec = next((p for p in roster["players"]
                    if p["name"].split()[-1] == g["a"].split()[-1]), None)
        dual = sum(1 for p in roster["players"] if p["num"] == num) > 1
        want_kit = ("B" if rec["team"] == 1 else "W") if (rec and dual) else None
        gt_ids = set()
        for pl in tr:
            core = pl.lstrip("#")
            kit = core[-1] if core[-1] in ("B", "W") else None
            digits = core.rstrip("BW")
            if not digits.isdigit() or int(digits) != num:
                continue
            if want_kit is None or kit in (None, want_kit):
                gt_ids.add(pl)
        if not gt_ids:
            continue
        base_ok = bool(o["pred_player"] and o["pred_player"].rstrip("?").split()[-1]
                       == g["a"].split()[-1])
        # E2 vote
        votes = defaultdict(float)
        for aa in ANGLES:
            lf = rel - OFFS[ang] + OFFS[aa]
            sc = holder_scores(ball, tr, aa, lf - int(0.8 * FPS), lf)
            if not sc:
                continue
            ranked = sorted(sc.values(), reverse=True)
            top = max(sc.items(), key=lambda kv: kv[1])[0]
            w = (ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)) / (ranked[0] + 1e-6)
            votes[top] += (2.0 if aa == ang else 1.0) * w
        e2_pick = max(votes.items(), key=lambda kv: kv[1])[0] if votes else None
        e2_ok = e2_pick in gt_ids if e2_pick else base_ok

        # E1 wrist confident-override on the release clip
        clip = REPO / f"{a.clips_dir}/{game}_{o['chunk']}_{rel}_{ang}.mp4"
        if not clip.exists():
            clip = REPO / f"{a.clips_dir}/{o['chunk']}_{rel}_{ang}.mp4"
        wr_pick = None
        wr_d = 9.9
        track_absent = True
        if clip.exists():
            cap = cv2.VideoCapture(str(clip))
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
                    track_absent = False
                    if box[3] - box[1] >= 100:
                        cands.append((pl, box))
                if not cands:
                    continue
                kpts, ks = pose(img, bboxes=[b for _, b in cands])
                for (pl, box), kp, s in zip(cands, kpts, ks):
                    bw = max(box[2] - box[0], 1.0)
                    for wi in (9, 10):
                        if s[wi] < 0.3:
                            continue
                        d = float(np.hypot(kp[wi][0] - bx, kp[wi][1] - by)) / bw
                        if d < wr_d:
                            wr_d, wr_pick = d, pl
            cap.release()
        wr_ok = (wr_pick in gt_ids) if (wr_pick and wr_d < 0.6) else None

        e3_ok = None
        if track_absent:
            try:
                gtn, pick = next(e3_iter)
                e3_ok = (pick == str(num)) if gtn == str(num) else None
            except StopIteration:
                pass

        # fused-feet court distance of a picked identity (for zone pass-2)
        def pick_dist(pl):
            if pl is None:
                return None
            ds = []
            for back in (0, 6, 12):
                pts = []
                for aa in ANGLES:
                    lf = rel - OFFS[ang] + OFFS[aa] - back
                    for df in (0, -1, 1, -2, 2):
                        box = tr[pl].get(aa, {}).get(lf + df)
                        if box is not None:
                            (x, y), = project_pixels(
                                [((box[0] + box[2]) / 2, box[3])], calib[aa])
                            pts.append([x, y])
                            break
                if pts:
                    pos = np.median(np.stack(pts), axis=0)
                    ds.append(min(
                        np.linalg.norm(pos - np.array(zcfg["baskets"]["L"])),
                        np.linalg.norm(pos - np.array(zcfg["baskets"]["R"]))))
            return float(max(ds)) if ds else None

        rows.append({"t": g["t"], "cls": g["cls"], "dist": o.get("release_dist_cm"),
                     "base": base_ok, "e2": e2_ok, "wr": wr_ok, "e3": e3_ok,
                     "e2_dist": pick_dist(e2_pick),
                     "wr_dist": pick_dist(wr_pick if (wr_pick and wr_d < 0.6) else None)})

    def fuse(r, d_close):
        if r["e3"] is not None:
            return r["e3"]
        # FT band (measured pre-fusion: isolated shooter, E2 vote +4 there;
        # wrists lose to lane players' raised hands)
        if r["dist"] is not None and 430 <= r["dist"] <= 610:
            return r["e2"]
        if r["dist"] is not None and r["dist"] < d_close and r["wr"] is not None:
            return r["wr"]
        return r["e2"]

    fit = [r for r in rows if r["t"] < 1200]
    ev = [r for r in rows if r["t"] >= 1200]
    if a.d_close is not None:
        D = a.d_close
        best = (sum(fuse(r, D) for r in fit), D)
    else:
        best = max((sum(fuse(r, D) for r in fit), D) for D in (550, 650, 750))
        D = best[1]
    n_ok = sum(fuse(r, D) for r in rows)
    ev_ok = sum(fuse(r, D) for r in ev)
    b_all = sum(r["base"] for r in rows)
    b_ev = sum(r["base"] for r in ev)
    print(f"n={len(rows)} | D_CLOSE={D} (fit first half: {best[0]}/{len(fit)})")
    print(f"BASELINE: full {b_all}/{len(rows)} ({b_all/len(rows):.0%}) | "
          f"held-out {b_ev}/{len(ev)} ({b_ev/len(ev):.0%})")
    print(f"E5 FUSED: full {n_ok}/{len(rows)} ({n_ok/len(rows):.0%}) | "
          f"held-out {ev_ok}/{len(ev)} ({ev_ok/len(ev):.0%})")
    per = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        k = r["cls"].split("_")[0] if "FREE" not in r["cls"] else "FT"
        per[k][0] += 1
        per[k][1] += r["base"]
        per[k][2] += fuse(r, D)
    print(f"\n{'class':<5} {'n':>3} {'base':>5} {'e5':>4}")
    for k, (n, b, w) in sorted(per.items()):
        print(f"{k:<5} {n:>3} {b:>5} {w:>4}")
    (REPO / "runs/shotdet_ab/e5_rows.json").write_text(json.dumps(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
