"""Cross-camera ball-candidate fusion (docs/06, docs/08): turn per-camera motion ball
candidates into ONE full-court ball trace.

A single camera only tracks the ball reliably in its own region (NR can't see the far
end). The fix mirrors player fusion: project each camera's candidates to the shared court,
then fuse by CROSS-CAMERA AGREEMENT — the real ball appears at one court point in >=2
cameras, while a per-camera false positive (a moving jersey, a marking) appears in only
one. Agreement both *locates* the ball (averaged) and *scores* it (a multi-camera cluster
outweighs single-camera noise), so the downstream Kalman tracker locks onto the true ball.

Inputs are COURT-coordinate candidates already aligned to one timeline (the driver projects
via per-camera homography and audio-syncs). This is the standard multi-view ball pipeline
(per-cam 2D detection -> cross-view fusion -> Kalman track), specialised to the court plane.
"""
from __future__ import annotations

import numpy as np

CLUSTER_CM = 200.0          # candidates within this across cameras are the same ball
AGREEMENT_BOOST = 0.6       # score multiplier per EXTRA camera that agrees (cross-view confidence)


def fuse_ball_candidates(per_cam: dict[str, dict[int, list[tuple]]],
                         cluster_cm: float = CLUSTER_CM) -> dict[int, list[tuple]]:
    """{cam: {frame: [(court_x, court_y, score)]}} (sync-aligned) -> {frame: [(x, y, score)]}.

    Per frame, greedily cluster candidates across cameras (<=1 per camera per cluster) and emit
    each cluster's score-weighted centroid with an agreement-boosted score."""
    frames = sorted({f for cam in per_cam.values() for f in cam})
    out: dict[int, list[tuple]] = {}
    for f in frames:
        obs = [(cam, x, y, s) for cam, cf in per_cam.items() for (x, y, s) in cf.get(f, [])]
        if not obs:
            continue
        clusters: list[dict] = []
        for cam, x, y, s in sorted(obs, key=lambda o: -o[3]):
            best, best_d = None, cluster_cm
            for c in clusters:
                if cam in c["cams"]:
                    continue                                # one obs per camera per cluster
                d = float(np.hypot(x - c["cen"][0], y - c["cen"][1]))
                if d <= best_d:
                    best, best_d = c, d
            if best is None:
                clusters.append({"cams": {cam}, "pts": [(x, y, s)], "cen": (x, y)})
            else:
                best["pts"].append((x, y, s))
                best["cams"].add(cam)
                w = np.array([p[2] for p in best["pts"]])
                best["cen"] = (float(np.average([p[0] for p in best["pts"]], weights=w)),
                               float(np.average([p[1] for p in best["pts"]], weights=w)))
        fused = []
        for c in clusters:
            base = sum(p[2] for p in c["pts"])
            score = base * (1.0 + AGREEMENT_BOOST * (len(c["cams"]) - 1))
            fused.append((round(c["cen"][0], 1), round(c["cen"][1], 1), round(score, 3)))
        out[f] = fused
    return out


def multicam_ball_trace(clips: dict[str, str], calib_dir: str, *, angles: list[str] | None = None,
                        ref: str = "NR", region_pad: float = 500.0, audio_sync: bool = True,
                        fps: float = 29.97, zone: dict[str, float] | None = None) -> dict[int, list]:
    """{angle: clip_path} + calib dir -> {frame: [court_x, court_y]} fused ball trace.

    The whole multi-camera motion pipeline in one call: per-camera 3-frame motion candidates ->
    project to court -> region-gate -> audio-sync to `ref` -> cross-camera agreement fusion ->
    stationarity filter -> Kalman track. Reused by the 4-cam driver and the pipeline fuse stage."""
    from pathlib import Path

    from ..tracking import iter_video_frames
    from .audiosync import audio_offset_seconds
    from .ball import reject_stationary, track_ball
    from .ball_motion import motion_candidates
    from .homography import calib_hull, in_calib_region, load_calib, project_pixels

    angles = angles or list(clips)
    zone = zone or {}
    ref_clip = clips.get(ref)
    per_cam: dict[str, dict[int, list[tuple]]] = {}
    for ang in angles:
        cp = clips.get(ang)
        if not cp or not Path(cp).exists():
            continue
        calib = load_calib(Path(calib_dir) / f"{ang}.json")
        hull = calib_hull(calib)
        cand_img = motion_candidates(list(iter_video_frames(str(cp))))
        off_f = 0
        if audio_sync and ref_clip and ang != ref:
            off_s, _peak = audio_offset_seconds(str(ref_clip), str(cp))
            off_f = int(round(off_s * fps))
        zc = zone.get(ang, 1.0)
        court: dict[int, list[tuple]] = {}
        for fi, cands in cand_img.items():
            for (cx, cy), (_x, _y, s) in zip(project_pixels([(x, y) for x, y, _ in cands], calib), cands):
                if in_calib_region((cx, cy), hull, region_pad):
                    court.setdefault(fi - off_f, []).append((float(cx), float(cy), float(s) * zc))
        per_cam[ang] = court
    fused, _banned = reject_stationary(fuse_ball_candidates(per_cam))
    return track_ball(fused, fps=fps, reinit_score=0.5)
