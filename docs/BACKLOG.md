# Backlog (deferred items)

Resume-ready notes for work intentionally parked. Each entry: status, why parked,
and exactly how to pick it back up.

## 1. FL → court homography calibration  — ✅ RESOLVED 2026-06-25 (by reuse)

Manual point-picking is no longer needed: adopted the **4 image→court homographies from
DEMO_UBALL** (`demo/calibration/{FL,FR,NL,NR}_cal_calibration.json`, 1920×1080, same DXF court)
→ `configs/calib/{FL,FR,NL,NR}.json`. Cameras are fixed per court, so they apply to any court-a
game. `homography.py` reads the direct `homography_matrix`. (Manual-picking tooling is still
there for new courts; the eyeballed `configs/calib_e6fba750_FL.json` was the 9%-in-court draft.)

## 2. Clean 4-camera fusion  — ✅ MUCH IMPROVED 2026-06-26 (calibration-region gate)

**Status:** 4-cam over-count largely fixed. **Root cause found (not what we thought):** FL/FR
homographies are *accurate where calibrated* (FL 13/13 pts @ ~13 cm, FR 14/14 @ ~12 cm) — but
their click-pairs only cover **their own half** of the court (FL x∈[1,787], FR x∈[1075,2146]).
Players outside that span were **extrapolated** → projected out of court → spawned cross-camera
duplicates. (So "only 6 inliers" was a RANSAC-threshold artifact, not a bad fit.)

**Fix (no new clicks):** baked each camera's calibrated **convex hull** (`calib_hull` in
`configs/calib/{ANG}.json`, from DEMO `*_cal_clickpairs.json`) and added a region gate
(`homography.in_calib_region`, `pad≈250 cm`) — a camera only contributes obs inside its
calibrated region. Wired into `fuse_cams.py` (`--region-pad`) and `pipeline/phases.run_fuse`.

**Result (e6 4-cam, 12 s):** 16.3 → **13.5 players/frame**, 35 → **27 global ids**, peak id
**41 → 24** (much less ID churn). Radar visibly cleaner (`runs/tracking/e6_fused_4cam_regiongate*.mp4`).
Tuned to `region_pad=250, cluster_dist=600` (tighter cluster_dist *over*-counts because cross-cam
disagreement is still several metres near hull edges).

**ReID finding (measured, both before & after the gate):** swapping imagenet OSNet → Market1501
(`osnet_x1_0_market1501`, loaded via `weights_only=True` safe path) is **neutral on players/frame**
(26 vs 27 ids). ReID is *not* the over-count lever — geometry + team + jersey dominate clustering.
Kept Market1501 as a graceful default (`reid.default_reid_weights`) since it should still help ID
stability / re-entry (not captured by this metric).

**To resume — remaining fusion polish:**
1. **Team classification balance** — radar shows too few team-A (blue) dots; SigLIP+KMeans split
   looks imbalanced per game. Check k=2 separation / per-game calibration (helps fusion uniqueness).
2. **Real NL/NR audio sync** — re-pull clips WITH audio (`pull_clip.py` keeps audio by default
   now), then `fuse_cams.py` measures their offsets. FR is already audio-synced (−13 frames).
3. **Lone edge dots** near the FT circles persist (possible refs / hull-boundary obs) — investigate.
4. (Optional) ReID-trained OSNet eval on an **ID-switch / re-entry** metric where it should pay off.
