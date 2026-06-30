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
