#!/usr/bin/env python3
"""Multi-camera fusion: per-angle tracks + calibrations -> ONE global id per player
-> top-down radar (docs/06).

  python scripts/fuse_cams.py --ref FL \
    --cam FL runs/tracking/e6fba750_FL_47_12_teams.json configs/calib/FL.json data/clips/e6fba750_FL_47_12.mp4 \
    --cam NL runs/tracking/e6fba750_NL_47_12_teams.json configs/calib/NL.json data/clips/e6fba750_NL_47_12.mp4 \
    --save-video runs/tracking/e6_fused_radar.mp4

Projects each camera's foot points to court cm (DEMO homographies), audio-syncs every
camera to --ref, fuses with FusionEngine, and renders the persistent global tracks.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

FPS = 29.97
PAD = 300.0                                   # cm tolerance outside court for kept obs
TEAM_COLOR = {"A": (0, 140, 255), "B": (255, 120, 40), "REF": (0, 255, 255), None: (160, 160, 160)}


def _load_cam(tracks_json, calib_json, region_pad=250.0):
    from uball_cc.fusion.court import LENGTH, WIDTH
    from uball_cc.fusion.homography import (calib_hull, homography_from_calib, in_calib_region,
                                            load_calib, project)
    from uball_cc.tracking import Track

    tracks = [Track.from_record(r) for r in json.loads(Path(tracks_json).read_text())["tracks"]]
    calib = load_calib(calib_json)
    h = homography_from_calib(calib)
    hull = calib_hull(calib)                          # camera's calibrated region (None -> accept all)
    court = project([t.foot_xy for t in tracks], h)
    per_frame: dict[int, list] = collections.defaultdict(list)
    kept = 0
    for t, c in zip(tracks, court):
        in_court = -PAD <= c[0] <= LENGTH + PAD and -PAD <= c[1] <= WIDTH + PAD
        if in_court and in_calib_region(c, hull, region_pad):   # gate to calibrated region
            per_frame[t.frame].append((t, (float(c[0]), float(c[1]))))
            kept += 1
    reid_path = Path(str(tracks_json).replace("_teams.json", "_reid.npz"))
    reid_by_id: dict[int, object] = {}
    if reid_path.exists():
        z = np.load(reid_path)
        reid_by_id = {int(i): v for i, v in zip(z["ids"], z["emb"])}
    return per_frame, kept, len(tracks), reid_by_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", nargs=4, action="append", metavar=("ANGLE", "TRACKS", "CALIB", "CLIP"),
                    required=True)
    ap.add_argument("--ref", required=True, help="reference angle for sync timeline")
    ap.add_argument("--save-video", default=None)
    ap.add_argument("--save-worldstate", default=None, help="write the fused world-state JSON here")
    ap.add_argument("--out-fps", type=float, default=30.0)
    ap.add_argument("--no-audio-sync", action="store_true")
    ap.add_argument("--max-assoc-dist", type=float, default=200.0)
    ap.add_argument("--gate-cost", type=float, default=6.0)
    ap.add_argument("--w-t", type=float, default=4.0, help="team-disagree penalty (lower if per-cam teams noisy)")
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--w-a", type=float, default=1.5, help="ReID weight in association cost")
    ap.add_argument("--w-d", type=float, default=1.0, help="court-distance weight")
    ap.add_argument("--cluster-dist", type=float, default=250.0, help="cross-camera grouping tolerance (cm)")
    ap.add_argument("--region-pad", type=float, default=250.0,
                    help="cm tolerance outside a camera's calibrated hull before its obs are dropped")
    a = ap.parse_args()
    engine_kw = dict(max_assoc_dist=max(a.max_assoc_dist, a.cluster_dist), gate_cost=a.gate_cost,
                     w_t=a.w_t, w_a=a.w_a, w_d=a.w_d, min_hits=a.min_hits, cluster_dist=a.cluster_dist)

    from uball_cc.fusion.audiosync import audio_offset_seconds
    from uball_cc.fusion.court import draw_court
    from uball_cc.fusion.engine import FusionEngine, Observation

    cams = {c[0]: {"tracks": c[1], "calib": c[2], "clip": c[3]} for c in a.cam}
    ref_clip = cams[a.ref]["clip"]

    # near cams own the court (accurate); far cams downweighted (far-field error)
    zone = {"FL": 0.6, "FR": 0.6, "NL": 1.0, "NR": 1.0}
    # project + sync each camera onto the ref frame timeline
    aligned: dict[str, dict[int, list]] = {}
    reid_maps: dict[str, dict] = {}
    for ang, c in cams.items():
        per_frame, kept, total, reid_by_id = _load_cam(c["tracks"], c["calib"], a.region_pad)
        reid_maps[ang] = reid_by_id
        if ang == a.ref or a.no_audio_sync:
            off_f, peak = 0, 0.0
        else:
            off_s, peak = audio_offset_seconds(ref_clip, c["clip"])
            off_f = int(round(off_s * FPS))
        # cam frame fc corresponds to ref frame (fc - off_f)
        shifted: dict[int, list] = collections.defaultdict(list)
        for fc, items in per_frame.items():
            shifted[fc - off_f].extend(items)
        aligned[ang] = shifted
        print(f"  {ang}: {kept}/{total} obs in-court, sync offset {off_f:+d} frames (peak {peak:.1f})")

    frames = sorted({f for sh in aligned.values() for f in sh})
    eng = FusionEngine(**engine_kw)
    per_frame_live: dict[int, list] = {}
    raw_by_frame: dict[int, list] = {}
    for f in frames:
        obs = []
        raw = []
        for ang, sh in aligned.items():
            zc = zone.get(ang, 1.0)
            for t, xy in sh.get(f, []):
                obs.append(Observation(ang, t.track_id, xy, team=t.team, jersey=t.jersey,
                                       reid=reid_maps[ang].get(t.track_id), zone_conf=zc))
                raw.append((xy, t.team))
        live = eng.step(f, obs)
        per_frame_live[f] = [(t.id, tuple(t.pos), t.team, t.jersey) for t in live]
        raw_by_frame[f] = raw

    n_ids = len({gid for v in per_frame_live.values() for gid, *_ in v})
    print(f"fused {len(frames)} frames -> {n_ids} distinct global ids "
          f"(avg {np.mean([len(v) for v in per_frame_live.values()]):.1f} players/frame)")

    if a.save_worldstate:
        roster: dict = collections.defaultdict(lambda: {"team": collections.Counter(),
                                                        "jersey": collections.Counter(), "xy": []})
        frames_out = []
        for f in frames:
            tr = []
            for gid, xy, team, jersey in per_frame_live[f]:
                tr.append({"global_id": gid, "court_xy": [round(float(xy[0]), 1), round(float(xy[1]), 1)],
                           "team": team, "jersey": jersey})
                r = roster[gid]
                if team:
                    r["team"][team] += 1
                if jersey is not None:
                    r["jersey"][jersey] += 1
                r["xy"].append([float(xy[0]), float(xy[1])])
            frames_out.append({"frame": f, "tracks": tr})
        players = [{"global_id": gid,
                    "team": (r["team"].most_common(1)[0][0] if r["team"] else None),
                    "jersey": (r["jersey"].most_common(1)[0][0] if r["jersey"] else None),
                    "court_xy": [round(v, 1) for v in np.mean(r["xy"], axis=0)] if r["xy"] else None}
                   for gid, r in roster.items()]
        ws = {"n_global_ids": len(players), "players": players, "frames": frames_out,
              "ref_angle": a.ref, "angles": list(cams), "fps": FPS}
        Path(a.save_worldstate).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save_worldstate).write_text(json.dumps(ws))
        print(f"worldstate -> {a.save_worldstate}")

    if not a.save_video:
        return 0
    import cv2
    base, to_px = draw_court(scale=0.45, margin=30)
    out = Path(a.save_video)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for f in frames:
        img = base.copy()
        for xy, team in raw_by_frame.get(f, []):                       # faint raw per-cam obs
            cv2.circle(img, to_px(xy), 3, (90, 90, 90), -1)
        for gid, xy, team, _jersey in per_frame_live.get(f, []):       # fused global tracks
            col = TEAM_COLOR.get(team, (160, 160, 160))
            cv2.circle(img, to_px(xy), 8, col, -1)
            cv2.circle(img, to_px(xy), 8, (0, 0, 0), 1)
            cv2.putText(img, str(gid), (to_px(xy)[0] + 9, to_px(xy)[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        if writer is None:
            hh, ww = img.shape[:2]
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), a.out_fps, (ww, hh))
        writer.write(img)
    if writer:
        writer.release()
    print(f"fused radar -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
