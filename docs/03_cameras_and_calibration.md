# 03 · Cameras & Calibration

## The rig (fixed, 4 cameras)
Two mounts, each carrying one **far** + one **near** camera watching opposite halves:
- **LEFT mount**: `FR` (far → watches the RIGHT side) + `NL` (near → watches the LEFT side)
- **RIGHT mount**: `FL` (far → watches the LEFT side) + `NR` (near → watches the RIGHT side)

Cameras: FL, FR (far, ~Linear/Wide rectilinear lens), NL, NR (near, GoPro **SuperView** —
non-rectilinear, important for calibration). 1920×1080 working resolution.

> Lens note (from prior work): far cams (FL/FR) are rectilinear and calibrate cleanly with a
> single homography; near cams (NL/NR) are SuperView and a single homography fits worse
> (especially toward edges). The center of the court is owned by the far cams, which
> calibrate well — keep that in the zone design.

## Court geometry
Exact facility court CAD (`court_2.dxf`, units = cm), origin at one baseline/sideline corner:
- Length (X) 0..**2143.7 cm**, Width (Y) 0..**1426.4 cm**.
- Landmarks (baselines, lane, FT lines, center circle, 3pt arcs) defined in cm — reused from
  DEMO_UBALL `lib/court.py`. This is **not** NBA-regulation; it's the real facility.

## Per-camera calibration = homography
Each camera gets a **3×3 homography** mapping image → court-cm (or its inverse for foot-point
→ court). Straight per-camera homography is the chosen model (TPS/undistort were evaluated and
rejected by the operator in prior work — keep it simple and robust).

A detection's **foot point** (bottom-center of box, or ankle midpoint from pose) is projected
through the inverse homography to a court coordinate. All cross-camera fusion happens in this
shared court frame.

## Per-court calibration onboarding (the generality requirement)
Because we deploy the **same rig to many courts**, calibration is a **per-venue onboarding
step**, not per-frame. Two-tier approach:

1. **Assisted manual** (always available): a calibration tool — operator drags court-line
   landmark markers onto the live frame (with a loupe magnifier), exports image↔court point
   pairs → `cv2.findHomography` (RANSAC) → `{CAM}_H.npy`. (Port the DEMO_UBALL
   `calibrate.html` + `compute_calib.py` tools.)
2. **Court-keypoint model** (speed-up): fine-tune a court-keypoint detector on Roboflow
   **CC-BY** basketball-court-keypoint datasets; auto-detect court landmarks per venue, then
   the operator only confirms/nudges. Reduces onboarding from many clicks to a few.

Calibration outputs per court: `configs/courts/<court_id>/{FL,FR,NL,NR}_H.npy` + a court
metadata file (dimensions, zone ownership). The pipeline loads by `court_id`.

## Camera sync (prerequisite, flagged)
Cross-camera fusion at a single instant requires the 4 cameras to be **frame-synchronized**.
Prior assessment: 2 of 4 angles were ~10–15 frames out. **Hard sync is a foundational
prerequisite** — without it, court projections of the same player at the "same" time
disagree and fusion degrades. For v1 on recorded games we sync by timestamp/offset
alignment; the live edge path needs a genuine synchronized-capture solution (Z-cam /
hardware trigger) — tracked in [11_edge_deployment](11_edge_deployment.md).

## Zone ownership (fusion aid)
Per-camera court-zone ownership (which camera is authoritative for which court region) reduces
cross-camera double-counting before fusion. Near cams own the deep ends; far cams own the
center band (they calibrate best there). Stored per court alongside homographies. This is an
input to fusion, not a replacement for it (see [06](06_cross_camera_fusion.md)).
