# 06 · Cross-Camera Fusion (the core)

This is the system's primary value and hardest component. It turns 4 sets of *local*
per-camera tracklets into **one persistent global identity per player**, consistent across all
cameras and through occlusion. The Roboflow blog and nearly all open repos stop before this.

## Design: "match-then-fuse" (open re-implementation of the MV3DT shape)
We mirror the proven architecture of NVIDIA MV3DT (per-cam lift → cross-cam tracklet exchange
→ visibility-weighted fusion → global IDs) but build it **fully open** so we can inject our
jersey-ResNet and own the association logic (which is exactly where a generic associator is
weakest for basketball — same-team, same-uniform players). Scaffolded on **tracklab (MIT)**.

## Pipeline
```
per-camera tracklets (×4)
   │  foot_xy → court_xy via per-court homography (03)
   ▼
1. PROJECT      every detection to shared court coords (cm)
2. ZONE GATE    keep a detection toward the court map only inside its camera's
                ownership zone (kills cross-cam double-counts; near owns ends,
                far owns center) — soft, not a hard drop
3. IDENTITY GATE per candidate cross-cam pair, compute association cost:
                cost =  w_d · court_distance
                      + w_a · (1 - reid_cosine_similarity)
                      + w_j · jersey_disagree   (huge bonus if same # same team)
                      + w_t · team_disagree
4. MATCH        Hungarian assignment per frame across cameras (4 cams → tiny
                matrices, real-time on CPU). Same-camera pairs never match.
5. GLOBAL TRACK one visibility-weighted Kalman filter per player on the court
                plane. Each camera that sees the player contributes, weighted
                by its zone confidence + detection score. Occlusion in one view
                is carried by the others.
6. ID MEMORY    persistent global IDs; re-entry re-attaches via jersey# +
                ReID embedding + last court position.
```

## Jersey number is the identity authority
Court proximity alone fails for same-team players standing close (the demo's "#1 is two
different people" bug). **Confirmed jersey number overrides proximity**: tracklets in
different cameras confirmed as the same `(team, number)` are the same global player,
regardless of small projection disagreements. A `(team, number)` labels **at most one** live
global track (uniqueness). This is the lesson hard-won in DEMO_UBALL, made first-class here.

## "IDs/jerseys changing carefully across all 4 angles in unison" (operator requirement)
Identity changes are decided at the **global** level, never per camera:
- A per-camera ID switch (e.g. ByteTrack swaps two players) does **not** change the global ID
  as long as the other cameras + jersey/ReID disagree with the switch.
- A global ID is only re-assigned when **multiple cameras agree** (court position + ReID +
  jersey) — a majority/consensus across views, not a single camera's opinion.
- Jersey-number confirmation uses temporal voting (same read N samples in a row) AND
  cross-camera agreement before it (re)labels a track.

## Occlusion handling (multi-view redundancy)
The whole point of 4 cameras: when a player is occluded in one view, ≥1 other view still sees
him. The global Kalman is sustained by the visible cameras; the occluded camera's track
re-attaches on reappearance via court position + ReID. This is why we don't over-invest in
single-camera occlusion (see [05](05_tracking_and_masks.md)).

## Calibration sensitivity (honest)
Fusion quality depends on calibration + sync. Far cams (FL/FR, rectilinear) calibrate well
and own the center; near cams (NL/NR, SuperView) are weaker at edges → larger projection
error there. Mitigations: zone ownership (let the accurate camera own each region), robust
(sustained-agreement) merging rather than single-frame distance, and jersey-as-authority. Hard
camera sync is a prerequisite ([03](03_cameras_and_calibration.md)).

## Validation
- Datasets: **TrackID3x3** (indoor fixed-cam 3×3 basketball — closest to our setup) and
  **SportsMOT** (basketball split) for tracking metrics; both benchmark-only (license).
- Metrics: **IDF1**, ID switches, MOTA, and a custom **cross-camera ID-consistency** (does a
  player keep one global ID across all cameras over the whole clip) + occlusion-recovery rate.
  See [13_evaluation](13_evaluation.md).

## Outputs & contract
Per frame: `[{global_id, court_xy, team, jersey#?, name?, members:[(cam,track_id)], pose}]`
+ ball + per-camera boxes → world model ([08](08_world_model.md)).

## Implementation status & results (2026-06-25)
**Built (`src/uball_cc/fusion/`):**
- `engine.py` — `FusionEngine`: gated association (court-dist + ReID + team + **jersey
  authority**) → per-camera Hungarian → **visibility-weighted court Kalman** (`kalman.py`)
  → persistent global IDs + `(team,number)` uniqueness + re-entry. **3 synthetic-GT tests pass**
  (one-id-per-player, same-team-neighbours-separate, re-entry after blackout).
- `court.py` — court model + top-down renderer (corner-origin, cm, from `configs/court_2.dxf`).
- `homography.py` — pixel→court projection; reads either a direct `homography_matrix` or manual
  correspondences. `audiosync.py` — FFT audio cross-correlation (ported from the dual_fusion repo).
- Drivers: `scripts/project_tracks.py` (single-cam radar), `scripts/fuse_cams.py` (multi-cam).

**Calibration (no manual picking):** adopted the 4 image→court homographies from
**DEMO_UBALL/demo/calibration** (`FL/FR/NL/NR`, 1920×1080, same DXF court) → `configs/calib/*.json`.
Cameras are fixed per court, so they apply to any court-a game.

**Sync:** audio cross-correlation measured **FR = −13 frames** vs FL (peak 10) — i.e. the DB
`video_metadata.sync_offset_seconds = 0` is unpopulated/wrong; real inter-angle offset ≈13 frames
(matches the dual_fusion `BAKED_OFFSET=13`). Also: `video_metadata` only registers FL+FR (the near
NL/NR files exist in S3 but aren't in the table).

**Results on e6fba750 (12 s window):**
- **NL + NR fuse cleanly** — ~14.7 global tracks/frame ≈ the real player count, teams separated.
  The two near cams are 99–100% in-court accurate.
- **Adding FL/FR over-counts** (~20/frame) — far cams project their far-field with large error
  (6–7 calib inliers of 13–15), so the same player lands >3 m apart across cameras and the
  per-camera association **spawns duplicates** instead of merging. ReID (imagenet OSNet) + zone
  downweighting only nudged it (39→36 ids), because cross-view OSNet similarity is weak and
  same-team players look alike.

**Update (2026-06-25):** the engine now does **cluster-then-match** (`engine.py` `_cluster`):
each frame, ALL cameras' observations are grouped into per-player clusters (≤1/camera, gated by
team/jersey/proximity) *then* matched to tracks — structurally avoids duplicate-spawning. With a
calibration-aware `cluster_dist=500`, the 4-cam drops from ~20 → **~15.8 players/frame** (≈ the
clean NL+NR result). Synthetic-GT tests still pass.

**Next (to fully clean the 4-cam), in priority — see [BACKLOG](BACKLOG.md):**
1. **Re-calibrate FL/FR** with more/cleaner points — now the dominant residual (the same player's
   far-cam projection lands >2–3 m from its near-cam projection, forcing a loose `cluster_dist`).
2. **ReID-trained OSNet** (Market1501) not imagenet, for stronger cross-view matching.
3. **Real NL/NR audio sync** (re-pull clips with audio; FR already synced at −13 frames).
