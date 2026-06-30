"""Pipeline phases (reuse the CV stack): detect -> track(+team+reid) -> fuse -> vlm.

Each phase reads the previous phase's artifacts under the job dir and writes its own,
so they run independently. Heavy models (RF-DETR, SigLIP, OSNet, Gemini) are created
once per phase and reused across the 4 angles.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .jobs import Job, JobStore
from .vlm import narrate

REPO = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS = str(REPO / "runs" / "rfdetr-s-1280-ourdata-v1" / "best.pth")
DEFAULT_CALIB_DIR = REPO / "configs" / "calib"
ZONE = {"FL": 0.6, "FR": 0.6, "NL": 1.0, "NR": 1.0}
TEAM_CAMS = {"NL", "NR"}    # only NEAR cams vote on team (far-cam crops too small to separate teams)


# ---------------- detect ----------------
def run_detect(job: Job, store: JobStore, *, weights: str = DEFAULT_WEIGHTS, model: str = "small",
               resolution: int = 1280, threshold: float = 0.25, max_frames: int | None = None,
               stride: int = 1) -> dict:
    import os
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")
    from uball_cc.detection.base import RFDETRDetector
    from uball_cc.tracking import iter_video_frames

    detector = RFDETRDetector(weights, resolution=resolution, threshold=threshold, model=model)
    out = job.dir / "detect"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for ang in job.angles:
        per_frame = []
        for img in iter_video_frames(store.video(job, ang), max_frames=max_frames, stride=stride):
            per_frame.append([{"b": [round(float(v), 1) for v in d.box_xyxy],
                               "s": round(float(d.score), 3), "c": int(d.class_id)}
                              for d in detector.predict(img)])
        (out / f"{ang}.json").write_text(json.dumps({"angle": ang, "n_frames": len(per_frame),
                                                      "detections": per_frame}))
        summary[ang] = {"frames": len(per_frame), "detections": sum(len(f) for f in per_frame)}
    return summary


# ---------------- track (+ team + reid) ----------------
def run_track(job: Job, store: JobStore, *, min_consecutive_frames: int = 3,
              team: bool = True, reid: bool = True) -> dict:
    from uball_cc.detection.base import Detection
    from uball_cc.tracking import ByteTrackTracker, Track, summarize
    from uball_cc.tracking.reid import OSNetEmbedder, default_reid_weights, track_embeddings
    from uball_cc.tracking.teams import assign_teams

    det_dir, out = job.dir / "detect", job.dir / "track"
    out.mkdir(parents=True, exist_ok=True)
    osnet = OSNetEmbedder(model_path=default_reid_weights()) if reid else None
    summary = {}
    for ang in job.angles:
        det = json.loads((det_dir / f"{ang}.json").read_text())
        tracker = ByteTrackTracker(ang, minimum_consecutive_frames=min_consecutive_frames)
        tracks: list[Track] = []
        for fi, frame_dets in enumerate(det["detections"]):
            dets = [Detection(tuple(d["b"]), d["s"], d["c"]) for d in frame_dets]
            tracks.extend(tracker.update(dets, fi))
        video = store.video(job, ang)
        if team and tracks:
            tracks, _ = assign_teams(video, tracks)          # method="color" (no SigLIP needed)
        if reid and tracks:
            emb = track_embeddings(video, tracks, embedder=osnet)
            np.savez(out / f"{ang}_reid.npz", ids=np.array(list(emb)),
                     emb=np.stack(list(emb.values())) if emb else np.zeros((0, 512)))
        (out / f"{ang}.json").write_text(json.dumps({"angle": ang,
                                                     "tracks": [t.to_record() for t in tracks]}))
        summary[ang] = summarize(tracks)
    return summary


# ---------------- fuse (cross-camera -> world-state) ----------------
def run_fuse(job: Job, store: JobStore, *, calib_dir: Path = DEFAULT_CALIB_DIR,
             ref_angle: str | None = None, max_assoc_dist: float = 600.0, gate_cost: float = 7.0,
             w_t: float = 1.0, w_a: float = 3.0, min_hits: int = 4, cluster_dist: float = 600.0,
             region_pad: float = 250.0, audio_sync: bool = True) -> dict:
    from uball_cc.fusion.audiosync import audio_offset_seconds
    from uball_cc.fusion.court import LENGTH, WIDTH
    from uball_cc.fusion.engine import FusionEngine, Observation
    from uball_cc.fusion.homography import (calib_hull, homography_from_calib, in_calib_region,
                                            load_calib, project)
    from uball_cc.tracking import Track

    trk = job.dir / "track"
    ref = ref_angle or ("FL" if "FL" in job.angles else job.angles[0])
    fps = 29.97
    aligned, reid_maps = {}, {}
    for ang in job.angles:
        calib = load_calib(calib_dir / f"{ang}.json")
        h = homography_from_calib(calib)
        hull = calib_hull(calib)                          # gate to the camera's calibrated region
        tracks = [Track.from_record(r) for r in json.loads((trk / f"{ang}.json").read_text())["tracks"]]
        court = project([t.foot_xy for t in tracks], h) if tracks else np.zeros((0, 2))
        rp = trk / f"{ang}_reid.npz"
        reid_maps[ang] = ({int(i): v for i, v in zip(z["ids"], z["emb"])}
                          if rp.exists() and (z := np.load(rp)) is not None else {})
        off = 0
        if audio_sync and ang != ref:
            off_s, _ = audio_offset_seconds(store.video(job, ref), store.video(job, ang))
            off = int(round(off_s * fps))
        per_frame: dict[int, list] = defaultdict(list)
        for t, c in zip(tracks, court):
            in_court = -300 <= c[0] <= LENGTH + 300 and -300 <= c[1] <= WIDTH + 300
            if in_court and in_calib_region(c, hull, region_pad):
                per_frame[t.frame - off].append((t, (float(c[0]), float(c[1]))))
        aligned[ang] = per_frame

    eng = FusionEngine(max_assoc_dist=max_assoc_dist, gate_cost=gate_cost, w_t=w_t,
                       w_a=w_a, min_hits=min_hits, cluster_dist=cluster_dist)
    frames_out, roster = [], defaultdict(lambda: {"team": Counter(), "jersey": Counter(), "xy": []})
    for f in sorted({fr for sh in aligned.values() for fr in sh}):
        obs = []
        for ang, sh in aligned.items():
            for t, xy in sh.get(f, []):
                team = t.team if ang in TEAM_CAMS else None   # only near cams vote on team
                obs.append(Observation(ang, t.track_id, xy, team=team, jersey=t.jersey,
                                       reid=reid_maps[ang].get(t.track_id), zone_conf=ZONE.get(ang, 1.0)))
        live = eng.step(f, obs)
        frames_out.append({"frame": f, "tracks": [{"global_id": t.id,
                          "court_xy": [round(float(v), 1) for v in t.pos], "team": t.team,
                          "jersey": t.jersey} for t in live]})
        for t in live:
            r = roster[t.id]
            if t.team:
                r["team"][t.team] += 1
            if t.jersey is not None:
                r["jersey"][t.jersey] += 1
            r["xy"].append([float(t.pos[0]), float(t.pos[1])])
    players = [{"global_id": gid, "team": (r["team"].most_common(1)[0][0] if r["team"] else None),
                "jersey": (r["jersey"].most_common(1)[0][0] if r["jersey"] else None),
                "court_xy": [round(v, 1) for v in np.mean(r["xy"], axis=0)] if r["xy"] else None}
               for gid, r in roster.items()]
    ws = {"n_global_ids": len(players), "players": players, "frames": frames_out,
          "ref_angle": ref, "angles": job.angles}
    # --- motion-fusion ball (full-court, docs/08) -> world-state + event stream ---
    from uball_cc.fusion.ball_fuse import multicam_ball_trace
    from uball_cc.fusion.events import derive_events
    clips = {ang: str(store.video(job, ang)) for ang in job.angles}
    ball = multicam_ball_trace(clips, str(calib_dir), ref=ref, audio_sync=audio_sync, zone=ZONE)
    ball_xy = {int(f): v for f, v in ball.items()}
    for fr in frames_out:
        fr["ball"] = ball_xy.get(fr["frame"])
    ws["ball"] = {str(f): v for f, v in ball.items()}
    events = derive_events(ws, ball_by_frame={f: tuple(v) for f, v in ball_xy.items()}, fps=fps)
    ws["events"] = events
    out = job.dir / "fuse"
    out.mkdir(parents=True, exist_ok=True)
    (out / "worldstate.json").write_text(json.dumps(ws))
    return {"n_global_ids": len(players), "n_frames": len(frames_out),
            "avg_players_per_frame": round(np.mean([len(f["tracks"]) for f in frames_out]), 1) if frames_out else 0,
            "ball_frames": len(ball), "n_events": events["summary"]["n_events"],
            "n_passes": events["summary"]["n_passes"], "n_turnovers": events["summary"]["n_turnovers"]}


# ---------------- vlm (world-state + video -> play-by-play) ----------------
def run_vlm(job: Job, store: JobStore, *, narration_angle: str | None = None,
            model: str | None = None, fps: float | None = None, api_key: str | None = None) -> dict:
    ws = json.loads((job.dir / "fuse" / "worldstate.json").read_text())
    ang = narration_angle or job.narration_angle
    result = narrate(store.video(job, ang), ws, model=model, fps=fps, api_key=api_key)
    out = job.dir / "vlm"
    out.mkdir(parents=True, exist_ok=True)
    (out / "narration.json").write_text(json.dumps({**result, "narration_angle": ang}, indent=2))
    return {"narration_angle": ang, "model": result["model"],
            "n_plays": len(result["play_by_play"]), "summary": result["summary"][:300],
            "usage": result["usage"]}


RUNNERS = {"detect": run_detect, "track": run_track, "fuse": run_fuse, "vlm": run_vlm}
