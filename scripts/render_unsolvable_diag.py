#!/usr/bin/env python3
"""Diagnostic reel: the UNSOLVABLE-class shots (wrong-WHO crowd cases), all 4
angles, with EVERY detection layer visible:
  thin gray boxes  = raw detector output (players+refs) — is detection there?
  colored + name   = identity streams (tracking's belief)
  orange dot BALL  = ball detections   |   red box HOOP = hoop detections
Right panel: GT vs pipeline call per case. New file — overwrites nothing.
"""
from __future__ import annotations

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

W, H = 1600, 900
PANEL = 360
FPS = 29.97
ANGLES = ("FL", "FR", "NL", "NR")
GRID_W = W - PANEL
CELL_W, CELL_H = GRID_W // 2, H // 2


def color_of(sid):
    h = hash(sid) % 360
    c = cv2.cvtColor(np.uint8([[[h / 2, 200, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return int(c[0]), int(c[1]), int(c[2])


def main() -> int:
    sel = json.loads(Path("/tmp/unsolv_sel.json").read_text())
    rosters, name_cache = {}, {}

    def name_of(game, sid):
        k = (game, sid)
        if k in name_cache:
            return name_cache[k]
        if game not in rosters:
            rosters[game] = json.loads((REPO / f"data/rosters/{game}.json").read_text())
        by_num = defaultdict(list)
        for p in rosters[game]["players"]:
            by_num[p["num"]].append(p)
        core = sid.lstrip("n")
        kit = core[-1] if core[-1] in ("B", "W") else None
        num = int(core.rstrip("BW"))
        c = by_num.get(num, [])
        if len(c) == 1:
            nm = c[0]["name"].split()[-1]
        else:
            m = [p for p in c if p["team"] == {"B": 1, "W": 2}.get(kit)]
            nm = (m[0]["name"].split()[-1] + ("(B)" if kit == "B" else "(W)")) if len(m) == 1 \
                 else (c[0]["name"].split()[-1] + "?" if c else sid)
        name_cache[k] = f"#{num} {nm}"
        return name_cache[k]

    tmp = str(REPO / "runs/event_demo/unsolvable_diag_4angle.mp4.raw.mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for ci, s in enumerate(sel):
        game, tag = s["game"], s["chunk"]
        offs = GAME_OFFS[game]
        chunk0 = float(tag.split("_")[0])
        t_rel = chunk0 + s["rel"] / FPS
        # caches per angle
        dets, ballz, streams, caps = {}, {}, defaultdict(dict), {}
        for ang in ANGLES:
            dp = REPO / f"runs/dets_cache/{game}_{ang}_{tag}_small_1280_t0.25.dets.npz"
            by_f = defaultdict(list)
            if dp.exists():
                z = np.load(dp)
                for b, sc, c, f in zip(z["boxes"], z["scores"], z["classes"], z["frame_idx"]):
                    if int(c) in (0, 1) and sc >= 0.4:
                        by_f[int(f)].append([float(v) for v in b])
            dets[ang] = by_f
            bz = defaultdict(list)
            bp = REPO / f"runs/ball_cache/{game}_{ang}_{tag}.ball.npz"
            if bp.exists():
                z = np.load(bp)
                for b, sc, f, c in zip(z["boxes"], z["scores"], z["frame_idx"], z["classes"]):
                    bz[(int(f), int(c))].append([float(v) for v in b])
            ballz[ang] = bz
            p = REPO / f"runs/event_demo/clips_unsolv/u{ci:02d}_{ang}.mp4"
            caps[ang] = cv2.VideoCapture(str(p)) if p.exists() else None
        tdir_t = s["tglob"]
        for p in (REPO / tdir_t.format(tag=tag)).glob(f"{game}_{tag}__n*__*.json"):
            parts = p.stem.split("__")
            d = json.loads(p.read_text())["frames"]
            streams[parts[1]][parts[2]] = {int(fr): r["box"]
                                           for fr, r in d.items() if r.get("present")}
        for fi in range(int(10 * FPS)):
            canvas = np.zeros((H, W, 3), np.uint8)
            canvas[:, GRID_W:] = (24, 24, 28)
            for gi, ang in enumerate(ANGLES):
                cap = caps[ang]
                if cap is None:
                    continue
                ok, frame = cap.read()
                if not ok:
                    continue
                fh, fw = frame.shape[:2]
                lf = round((t_rel - 6 - chunk0) * FPS) + offs[ang] + fi
                sx, sy = CELL_W / fw, CELL_H / fh
                cell = cv2.resize(frame, (CELL_W, CELL_H))
                for b in dets[ang].get(lf, []):          # raw detections: thin gray
                    cv2.rectangle(cell, (int(b[0]*sx), int(b[1]*sy)),
                                  (int(b[2]*sx), int(b[3]*sy)), (140, 140, 140), 1)
                for sid, angs in streams.items():        # identity streams: color+name
                    box = angs.get(ang, {}).get(lf)
                    if box is None:
                        continue
                    col = color_of(sid)
                    cv2.rectangle(cell, (int(box[0]*sx), int(box[1]*sy)),
                                  (int(box[2]*sx), int(box[3]*sy)), col, 2)
                    cv2.putText(cell, name_of(game, sid),
                                (int(box[0]*sx), max(12, int(box[1]*sy) - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
                for b in ballz[ang].get((lf, 1), []):    # hoop: red box
                    cv2.rectangle(cell, (int(b[0]*sx), int(b[1]*sy)),
                                  (int(b[2]*sx), int(b[3]*sy)), (0, 0, 230), 2)
                for b in ballz[ang].get((lf, 0), []):    # ball: orange dot
                    cx_, cy_ = int((b[0]+b[2])/2*sx), int((b[1]+b[3])/2*sy)
                    cv2.circle(cell, (cx_, cy_), 7, (0, 165, 255), -1)
                x0, y0 = (gi % 2) * CELL_W, (gi // 2) * CELL_H
                canvas[y0:y0+CELL_H, x0:x0+CELL_W] = cell
                cv2.putText(canvas, ang, (x0+10, y0+24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (205, 205, 210), 2)
            x = GRID_W + 16
            cv2.putText(canvas, f"HARD CASE {ci+1}/{len(sel)}", (x, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 230, 255), 2)
            cv2.putText(canvas, f"{game[:8]}  t={int(s['t'])//60:02d}:{int(s['t'])%60:02d}",
                        (x, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 175), 1)
            for yy, (lab, val, col) in enumerate([
                    ("GROUND TRUTH", "", (200, 200, 210)),
                    (s["gt_cls"].replace("_", " "), s["gt"], (235, 235, 240)),
                    ("", "", (0, 0, 0)),
                    ("PIPELINE SAID", "", (60, 180, 230)),
                    ("", s["pred"] or "no pick", (235, 235, 240))]):
                if lab:
                    cv2.putText(canvas, lab, (x, 130 + yy*56),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
                if val:
                    cv2.putText(canvas, str(val)[:26], (x, 156 + yy*56),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 240), 1)
            cv2.putText(canvas, "gray=raw det  color=identity", (x, H - 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 155, 160), 1)
            cv2.putText(canvas, "orange=ball  red=hoop", (x, H - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 155, 160), 1)
            vw.write(canvas)
        for c in caps.values():
            if c is not None:
                c.release()
        print(f"case {ci+1}/{len(sel)}", flush=True)
    vw.release()
    outp = REPO / "runs/event_demo/unsolvable_diag_4angle.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", tmp,
                    "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p",
                    "-y", str(outp)], check=True)
    Path(tmp).unlink()
    print(f"-> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
