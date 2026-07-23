#!/usr/bin/env python3
"""FULL-GAME review render: 2x2 all-angle grid + prediction/GT feeds.

Left 2x2 (FL/FR/NL/NR, offset-aligned): EVERY identity stream boxed with
"#num Name" above (team-colored), the current BALL HOLDER in thick yellow
(the ball-first state machine made visible), raw ball detections as orange
dots with a short trail, hoop detections as red boxes.
Right panel: big clock + current holder, PIPELINE event feed (high tier,
mm:ss timestamps), GROUND TRUTH feed below it.

Writes straight to x264 via an ffmpeg pipe — no raw intermediate.

  .venv/bin/python scripts/render_fullgame_grid.py --game 2c490f1a \
      --out runs/event_demo/fullgame_2c490f1a.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from game_meta import GAME_OFFS, GAME_CHUNKS

W, H = 1600, 900
PANEL = 360
GRID_W = W - PANEL
CELL_W, CELL_H = GRID_W // 2, H // 2
FPS = 29.97
ANGLES = ("FL", "FR", "NL", "NR")
TEAM_COL = {1: (255, 170, 70), 2: (90, 220, 130), None: (150, 150, 150)}
HOLDER_COL = (0, 215, 255)
TRAIL_N = 14


def short(cls):
    return cls.replace("FREE_THROW", "FT").replace("_", " ")


def load_tracks(game, tglob):
    tracks = defaultdict(lambda: defaultdict(dict))
    for tag in GAME_CHUNKS[game]:
        base = round(float(tag.split("_")[0]) * FPS)
        tdir = REPO / tglob.format(tag=tag)
        for p in tdir.glob(f"{game}_{tag}__n*__*.json"):
            parts = p.stem.split("__")
            d = json.loads(p.read_text())["frames"]
            dst = tracks[parts[1]][parts[2]]
            for fr, r in d.items():
                if r.get("present"):
                    dst[base + int(fr)] = r["box"]
    return tracks


def load_ballz(game):
    """(cam-local global frame, cls) -> boxes; cls 0 ball (BEST det only —
    multiple dets per frame draw trail spiderwebs), cls 1 hoop (all)."""
    bz = {ang: defaultdict(list) for ang in ANGLES}
    best = {ang: {} for ang in ANGLES}
    for tag in GAME_CHUNKS[game]:
        base = round(float(tag.split("_")[0]) * FPS)
        for ang in ANGLES:
            p = REPO / f"runs/ball_cache/{game}_{ang}_{tag}.ball.npz"
            if not p.exists():
                continue
            z = np.load(p)
            for b, s, f, c in zip(z["boxes"], z["scores"], z["frame_idx"],
                                  z["classes"]):
                gf = base + int(f)
                if int(c) == 1:
                    bz[ang][(gf, 1)].append([float(v) for v in b])
                else:
                    if gf not in best[ang] or s > best[ang][gf][1]:
                        best[ang][gf] = ([float(v) for v in b], float(s))
    for ang in ANGLES:
        for gf, (b, _) in best[ang].items():
            bz[ang][(gf, 0)].append(b)
    return bz


def feed(img, y0, y1, title, items, t_now, color, rows):
    cv2.putText(img, title, (GRID_W + 14, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, color, 2)
    cv2.line(img, (GRID_W + 14, y0 + 36), (W - 14, y0 + 36), color, 1)
    y = y0 + 62
    for e in items[-rows:][::-1]:
        hot = (t_now - e["t"]) <= 2.5
        col = (90, 230, 255) if hot else (222, 222, 228)
        cv2.putText(img, f"{int(e['t'])//60:02d}:{int(e['t'])%60:02d}  "
                         f"{short(e['classification'])[:20]}",
                    (GRID_W + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    col, 2 if hot else 1)
        y += 19
        cv2.putText(img, f"   {(e.get('player_a') or '?')[:24]}",
                    (GRID_W + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    col if hot else (148, 152, 158), 1)
        y += 24
        if y > y1 - 8:
            break


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("--plays", default=None)
    ap.add_argument("--tracks-glob", default=None)
    ap.add_argument("--srcdir", default="runs/event_demo")
    ap.add_argument("--ring", action="store_true",
                    help="draw a glowing floor ring under the ball holder "
                         "(follows the same fused identity across all 4 angles)")
    ap.add_argument("--ring-mov",
                    default="/home/akhilesh/Desktop/Uball/desgin/Circle.mov",
                    help="alpha Circle.mov used for the ring")
    ap.add_argument("--keep-holder-box", action="store_true",
                    help="with --ring, also keep the thick yellow holder box")
    a = ap.parse_args()
    game = a.game
    OFFS = GAME_OFFS[game]
    tglob = a.tracks_glob or ("runs/events_fg_{tag}" if game == "e6fba750"
                              else f"runs/events_fg_{game[:3]}_{{tag}}")

    plays = json.loads((REPO / (a.plays or f"data/plays/{game}_full.json"))
                       .read_text())["plays"]
    gt = [{"t": p["t"], "classification": p["cls"], "player_a": p["a"]}
          for p in plays]
    ev = json.loads((REPO / f"runs/tracking/ledger/events_v2_{game}.json")
                    .read_text())["events"]
    ev = [e for e in ev if e.get("tier", "high") == "high"]
    pos = REPO / f"runs/tracking/ledger/possession_events_{game}.json"
    if pos.exists():
        ev += json.loads(pos.read_text())["events"]
    ev.sort(key=lambda e: e["t"])

    roster = json.loads((REPO / f"data/rosters/{game}.json").read_text())
    by_num = defaultdict(list)
    for p in roster["players"]:
        by_num[p["num"]].append(p)

    def resolve(sid):
        core = sid.lstrip("#").lstrip("n")
        kit = core[-1] if core[-1] in ("B", "W") else None
        num = int(core.rstrip("BW"))
        c = by_num.get(num, [])
        if len(c) == 1:
            return f"#{num} {c[0]['name'].split()[-1]}", c[0]["team"]
        m = [p for p in c if p["team"] == roster.get("kit_team", {"B": 1, "W": 2}).get(kit)]
        if len(m) == 1:
            return f"#{num} {m[0]['name'].split()[-1]}", m[0]["team"]
        return f"#{num} ?", None

    print("loading tracks/ball/holders...", flush=True)
    tracks = load_tracks(game, tglob)
    ballz = load_ballz(game)
    label, teamof = {}, {}
    for sid in tracks:
        label[sid], teamof[sid] = resolve(sid)

    hold_at = {}
    hp = REPO / f"runs/tracking/ledger/holders_{game}.json"
    if hp.exists():
        for g0, g1, pl in json.loads(hp.read_text())["segments"]:
            for f in range(g0, g1 + 1):
                hold_at[f] = pl

    caps, next_idx = {}, {}
    for ang in ANGLES:
        p = REPO / a.srcdir / f"fullsrc_{game}_{ang}.mp4"
        caps[ang] = cv2.VideoCapture(str(p)) if p.exists() else None
        next_idx[ang] = 0
        if caps[ang] is not None and not caps[ang].isOpened():
            caps[ang] = None
    if a.dur is None:
        n = caps["FL"].get(cv2.CAP_PROP_FRAME_COUNT) if caps["FL"] else 0
        a.dur = max(0.0, n / FPS - a.t0 - 0.5)
    n_frames = int(a.dur * FPS)
    f0 = round(a.t0 * FPS)

    outp = REPO / a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "30000/1001", "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(outp)],
        stdin=subprocess.PIPE)

    # ring overlay (holder = a cross-camera fused identity -> same ring in every angle)
    ring_ov, ring_n, ring_fps, ring_stab = None, 0, 24.0, {}
    if a.ring:
        from ring_overlay import (RingStabilizer, build_overlay_cache,
                                   draw_ring_box, occlusion_from_boxes)
        cache = str(outp.parent / "ring_cache.npy")
        outp.parent.mkdir(parents=True, exist_ok=True)
        ring_n, ring_fps = build_overlay_cache(a.ring_mov, cache)
        ring_ov = np.load(cache, mmap_mode="r")
        ring_stab = {ang: RingStabilizer() for ang in ANGLES}
        print(f"ring: {ring_n} Circle.mov frames loaded", flush=True)

    trail = {ang: deque(maxlen=TRAIL_N) for ang in ANGLES}
    canvas = np.zeros((H, W, 3), np.uint8)
    frames = {}
    prev_holder = None
    import time
    t_start = time.time()
    for fi in range(n_frames):
        f = f0 + fi                      # ref (FL) frame
        t_now = f / FPS
        canvas[:] = 0
        canvas[:, GRID_W:] = (24, 24, 28)
        holder = hold_at.get(f)
        # possession switch -> snap the ring onto the new holder. apply()'s own
        # snap test compares ground-locked Y, which the vertical lock holds near
        # the old value, so a switch to a further-away player never trips it and
        # the ring slides across the floor (sitting on the wrong player meanwhile).
        if holder != prev_holder:
            for stab in ring_stab.values():
                stab.reset()
            prev_holder = holder
        for gi, ang in enumerate(ANGLES):
            cap = caps[ang]
            if cap is None:
                continue
            target = f + OFFS[ang]       # cam-local frame
            while next_idx[ang] <= target:
                ok, fr = cap.read()
                if not ok:
                    break
                frames[ang] = fr
                next_idx[ang] += 1
            cell = frames.get(ang)
            if cell is None:
                continue
            if cell.shape[1] != CELL_W or cell.shape[0] != CELL_H:
                cell = cv2.resize(cell, (CELL_W, CELL_H))
            else:
                cell = cell.copy()
            gf = target
            # identity streams: every player, name above box
            # (boxes are in source 1920x1080 coords; cell is CELL_WxCELL_H)
            holder_cell_box, cell_boxes = None, []
            for sid, angs in tracks.items():
                box = angs.get(ang, {}).get(gf)
                if box is None:
                    continue
                bx1, by1 = int(box[0] / 1920 * CELL_W), int(box[1] / 1080 * CELL_H)
                bx2, by2 = int(box[2] / 1920 * CELL_W), int(box[3] / 1080 * CELL_H)
                cell_boxes.append((bx1, by1, bx2, by2))
                is_h = (sid == holder)
                if is_h:
                    holder_cell_box = (bx1, by1, bx2, by2)
                col = HOLDER_COL if is_h else TEAM_COL.get(teamof.get(sid))
                # with --ring the ring marks the holder; suppress the thick box
                # (unless --keep-holder-box) but keep every name label.
                draw_box = not (is_h and ring_ov is not None and not a.keep_holder_box)
                if draw_box:
                    cv2.rectangle(cell, (bx1, by1), (bx2, by2), col, 3 if is_h else 1)
                cv2.putText(cell, label.get(sid, sid), (bx1, max(10, by1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, col,
                            2 if is_h else 1)
            # glowing floor ring under the fused holder (behind players in front)
            if ring_ov is not None:
                if holder_cell_box is not None:
                    occ = occlusion_from_boxes(cell_boxes, holder_cell_box, cell.shape)
                    oidx = int(gf / FPS * ring_fps) % ring_n
                    draw_ring_box(cell, holder_cell_box,
                                  np.ascontiguousarray(ring_ov[oidx]),
                                  stab=ring_stab[ang], occ=occ)
                else:
                    ring_stab[ang].reset()
            # hoop boxes
            for b in ballz[ang].get((gf, 1), []):
                cv2.rectangle(cell, (int(b[0] / 1920 * CELL_W), int(b[1] / 1080 * CELL_H)),
                              (int(b[2] / 1920 * CELL_W), int(b[3] / 1080 * CELL_H)),
                              (0, 0, 230), 2)
            # ball: dot + trail
            for b in ballz[ang].get((gf, 0), []):
                cx = int((b[0] + b[2]) / 2 / 1920 * CELL_W)
                cy = int((b[1] + b[3]) / 2 / 1080 * CELL_H)
                trail[ang].append((f, cx, cy))
            pts = [(tf, x, y) for tf, x, y in trail[ang] if f - tf <= TRAIL_N]
            for (t0, x0, y0), (t1, x1, y1) in zip(pts, pts[1:]):
                # only join temporally-adjacent, spatially-plausible points
                if t1 - t0 <= 3 and abs(x1 - x0) + abs(y1 - y0) <= 60:
                    cv2.line(cell, (x0, y0), (x1, y1), (0, 120, 200), 1)
            if pts:
                cv2.circle(cell, (pts[-1][1], pts[-1][2]), 6, (0, 165, 255), -1)
            x0, y0 = (gi % 2) * CELL_W, (gi // 2) * CELL_H
            canvas[y0:y0 + CELL_H, x0:x0 + CELL_W] = cell
            cv2.putText(canvas, ang, (x0 + 8, y0 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (205, 205, 210), 2)
        # panel
        clock = f"{int(t_now)//60:02d}:{int(t_now)%60:02d}"
        cv2.putText(canvas, clock, (GRID_W + 14, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 245), 2)
        cv2.putText(canvas, game[:8], (GRID_W + 150, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 155, 160), 1)
        hname = label.get(holder, "-") if holder else "-"
        cv2.putText(canvas, f"BALL: {hname}", (GRID_W + 14, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, HOLDER_COL, 2)
        cv_feed = [e for e in ev if e["t"] <= t_now]
        gt_feed = [e for e in gt if e["t"] <= t_now]
        feed(canvas, 104, H // 2 + 40, "PIPELINE (ours)", cv_feed, t_now,
             (60, 180, 230), 8)
        feed(canvas, H // 2 + 56, H - 30, "GROUND TRUTH", gt_feed, t_now,
             (200, 200, 210), 7)
        cv2.putText(canvas, "yellow=ball holder  orange=ball  red=hoop",
                    (GRID_W + 14, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (140, 145, 150), 1)
        ff.stdin.write(canvas.tobytes())
        if fi % 3000 == 2999:
            el = time.time() - t_start
            eta = el / (fi + 1) * (n_frames - fi - 1)
            print(f"  {t_now:6.0f}s rendered | {fi/el:5.1f} fps | "
                  f"ETA {eta/60:5.1f} min", flush=True)
    ff.stdin.close()
    ff.wait()
    for c in caps.values():
        if c is not None:
            c.release()
    print(f"-> {outp} ({outp.stat().st_size/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
