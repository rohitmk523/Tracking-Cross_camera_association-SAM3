#!/usr/bin/env python3
"""IMAGE-SPACE possession: a held ball overlaps its holder's box in the camera image, so
possession is decided per camera in PIXELS — no court projection, so the flat-court error
for elevated balls (which muted the appearance path: 288 ball frames -> 1 possession)
never enters. Near cams only (ball AP 0.80 is near-basket).

Per frame per camera: top ball detection -> the player box that contains (or is nearest
to) the ball centre -> that local track -> the fused GLOBAL identity (worldstate members
map) -> {ref_frame: holder_gid} -> events (hysteresis + min-run in events.holder_runs).
Scores itself against operator GT when a matching file exists.

  python scripts/possession_image.py --game c2a354fe --suffix _333_14 \
      --worldstate runs/tracking/c2a354fe_worldstate.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

BOX_EXPAND = 0.20            # a held ball sits at the body edge; expand boxes a little
NEAR_FRAC = 0.5              # no containment: accept nearest box centre within this * box_h


def _attribute(ball_xy, players) -> int | None:
    """players: [(track_id, (x1,y1,x2,y2))]. Containment first (smallest box wins =
    nearest player), else nearest box centre within NEAR_FRAC of its box height."""
    bx, by = ball_xy
    containing = []
    for tid, (x1, y1, x2, y2) in players:
        ex, ey = (x2 - x1) * BOX_EXPAND, (y2 - y1) * BOX_EXPAND
        if x1 - ex <= bx <= x2 + ex and y1 - ey <= by <= y2 + ey:
            containing.append((abs((x2 - x1) * (y2 - y1)), tid))
    if containing:
        return min(containing)[1]
    best = None
    for tid, (x1, y1, x2, y2) in players:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        d = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
        lim = NEAR_FRAC * (y2 - y1)
        if d <= lim and (best is None or d < best[0]):
            best = (d, tid)
    return best[1] if best else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--cams", default="NL,NR")
    ap.add_argument("--ref", default="FL")
    ap.add_argument("--clip-dir", default="data/clips")
    ap.add_argument("--worldstate", required=True)
    ap.add_argument("--weights", default="runs/rfdetr-s-1280-ourdata-v1/best.pth")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--ball-json", default=None,
                    help="template with {ang}: precomputed per-frame ball [x,y,conf] "
                         "(e.g. BallNet output) — skips the RF-DETR ball pass")
    ap.add_argument("--ball-conf", type=float, default=0.3,
                    help="min confidence for precomputed ball positions")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gt", default=None, help="default: data/gt_events/<game><suffix>.json variants")
    a = ap.parse_args()

    from uball_cc.detection.base import RFDETRDetector
    from uball_cc.fusion.audiosync import audio_offset_seconds
    from uball_cc.fusion.events import derive_events_from_holders
    from uball_cc.tracking import Track, iter_video_frames

    ws = json.loads(Path(a.worldstate).read_text())
    # (cam, local_id) -> gid per frame (non-coasting members = fresh camera evidence)
    gmap: dict[int, dict] = defaultdict(dict)
    for fr in ws["frames"]:
        for t in fr["tracks"]:
            if not t.get("coasting"):
                for cam, lid in (t.get("members") or {}).items():
                    gmap[fr["frame"]][(cam, lid)] = t["global_id"]

    detector = None
    if not a.ball_json:
        detector = RFDETRDetector(a.weights, resolution=1280, threshold=a.threshold, model="small")
    ref_clip = Path(a.clip_dir) / f"{a.game}_{a.ref}{a.suffix}.mp4"
    votes: dict[int, list] = defaultdict(list)      # ref_frame -> [(ball_score, cam, lid)]
    for ang in a.cams.split(","):
        clip = Path(a.clip_dir) / f"{a.game}_{ang}{a.suffix}.mp4"
        tracks_p = REPO / "runs/tracking" / f"{a.game}_{ang}{a.suffix}_teams.json"
        if not (clip.exists() and tracks_p.exists()):
            print(f"  {ang}: missing clip/tracks — skipped")
            continue
        players_by_f: dict[int, list] = defaultdict(list)
        for r in json.loads(tracks_p.read_text())["tracks"]:
            t = Track.from_record(r)
            if t.class_id == 0 and t.team != "REF":
                players_by_f[t.frame].append((t.track_id, t.box_xyxy))
        off = 0
        if ang != a.ref and ref_clip.exists():
            off_s, _ = audio_offset_seconds(str(ref_clip), str(clip))
            off = int(round(off_s * 29.97))
        n_attr = 0
        if a.ball_json:                             # precomputed ball (BallNet)
            bp = Path(a.ball_json.format(ang=ang))
            if not bp.exists():
                print(f"  {ang}: no ball json {bp} — skipped")
                continue
            for f_str, (bx, by, conf) in json.loads(bp.read_text()).items():
                if conf < a.ball_conf:
                    continue
                fi = int(f_str)
                tid = _attribute((bx, by), players_by_f.get(fi, []))
                if tid is not None:
                    votes[fi - off].append((float(conf), ang, tid))
                    n_attr += 1
        else:
            for fi, img in enumerate(iter_video_frames(str(clip))):
                balls = [d for d in detector.predict(img) if d.class_id == 2]
                if not balls:
                    continue
                d = max(balls, key=lambda b: b.score)
                bxy = ((d.box_xyxy[0] + d.box_xyxy[2]) / 2, (d.box_xyxy[1] + d.box_xyxy[3]) / 2)
                tid = _attribute(bxy, players_by_f.get(fi, []))
                if tid is not None:
                    votes[fi - off].append((float(d.score), ang, tid))
                    n_attr += 1
        print(f"  {ang}: ball attributed to a player box in {n_attr} frames (sync {off:+d}f)",
              flush=True)

    holder_by_frame: dict[int, int] = {}
    n_mapped = 0
    for f, vs in votes.items():
        _, cam, lid = max(vs)                       # highest-confidence ball wins the frame
        gid = None
        for df in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5):
            gid = gmap.get(f + df, {}).get((cam, lid))
            if gid is not None:
                break
        if gid is not None:
            holder_by_frame[f] = gid
            n_mapped += 1
    print(f"holder stream: {n_mapped} frames mapped to global identities "
          f"(of {len(votes)} attributed)", flush=True)

    ev = derive_events_from_holders(ws, holder_by_frame)
    out = Path(a.out or f"runs/tracking/{a.game}{a.suffix}_events_imagespace.json")
    out.write_text(json.dumps({"events": ev, "holder_frames": len(holder_by_frame)}, indent=1))
    poss = [(e["frame_window"], e["team"], e["player"]) for e in ev["events"]
            if e["event"] == "possession"]
    print(f"events: {ev['summary']} -> {out}")
    print("possessions:", poss)

    for cand in (a.gt, f"data/gt_events/{a.game}{a.suffix}.json",
                 f"data/gt_events/{a.game}_{a.ref}{a.suffix}.json"):
        if cand and Path(cand).exists():
            print(f"--- scoring vs GT {cand} ---", flush=True)
            subprocess.run([sys.executable, "scripts/eval_events_gt.py",
                            "--gt", cand, "--events", str(out)])
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
