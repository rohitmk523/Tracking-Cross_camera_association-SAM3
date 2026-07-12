# Events v2 — shot-origin integration + possession-change layer

**Goal:** break the 49% WHO ceiling on scoring events, and add the non-scoring
events (rebound / steal / turnover / block). Both are logic on data we already
have — no new training, no annotation.

**Why v1 topped out (measured):** our possession signal is "nearest player to the
ball, per frame." When players cluster it picks the wrong body ~half the time, and
no aggregation rule fixed it (3 tested). The fix is not a better possession
heuristic — it's a *cleaner trigger*: the shot-detection system already isolates
the exact shot and gives the ball's release point in court coordinates. Match that
one clean point to our identities and the noise disappears.

---

## What each system already produces (verified by code read)

**Shot-detection system** (`uball_shot_detection_dual_fusion_v2`):
- Tracks ball + rim per frame, **triangulated to 3-D court coordinates** (`X_cm =
  [x, y, z]` in cm) — same court frame as ours.
- Per shot: detects the **arc apex** (`apex_t`, `apex_X_cm`), the descent, and a
  **made/miss verdict** with calibrated confidence. The arc window starts ~0.8 s
  *before* apex — i.e. it contains the **release** (ball leaving the hand: the
  low point of the arc just before it rises).
- Already run on e6 (crops + per-play tracks in S3; `data/crops/e6fba750…`).

**Our tracking pipeline:**
- Per identity, per frame: **court position (x, y) in cm** (`solve_player_xcam.court()`,
  the jersey-anchored correction). 81.7% strict / 86.2% fused identity accuracy.

Both project into the **same court coordinate system** → the join is a direct
distance match, no re-calibration.

---

## PLAN REVIEW (Fable, 2026-07-13) — two corrections to Part A/Step-1

1. **Part A below is STALE where it uses their triangulated court-space release
   point** — triangulation is rejected (their own verdict). The CURRENT WHO design
   is scripts/detect_shots.py: our ball+hoop detections → arc apex at the hoop per
   far camera → release instant → shooter = identity under the ball in IMAGE space.
   Read Part A's matching idea through that lens; the court-space join is dead.
2. **We do NOT need their extract_tracks at all.** It exists to run their frozen
   far_v16 bundle — which we are retiring. Cleaner: our build_ball_cache already
   produces per-frame ball+hoop detections; write a thin ADAPTER that emits their
   P1 track-parquet schema (per shot window, per angle: t, ball_x/y/conf,
   rim_x/y/w/h) from our caches, for (a) their 185 labeled shots and (b) e6's 133
   windows. Then their P2 (features) and P3 (retrain, seed 42, LOGO ≥0.949) run
   UNCHANGED on top. One detector (ours), their proven feature+model recipe, no
   frozen-bundle/S3-work-prefix plumbing.

## Part A — Shot-origin attribution (the WHO fix)

**Algorithm (replaces frame-by-frame possession for shots):**
1. From the shot-detection output, take each shot's **release point**
   `R = (x, y)` (ball `X_cm` at arc start, floor-projected — drop z) and
   `t_release`.
2. At `t_release`, project **our identities** to court (x, y) — we already have this.
3. **Shooter = the identity whose court position is nearest to `R`** (gate at a
   sane radius, e.g. ≤ 150 cm; the ball at release is at the shooter's hands).
4. Points: keep our **feet-based zone** (2/3/4 PT, already 62%) — the shooter's feet
   are the officiating reference; cross-check against `R`'s zone as a tiebreaker.
5. Made/miss: **consume** the shot-detection verdict. Emit the full event:
   `{player, points, made/miss, source:"cv"}`.

**Why this breaks the ceiling:** one clean release event in court space vs a noisy
per-frame nearest-ball vote. The shot-detection has already done the hard work of
finding *when* and *where* the ball left toward the hoop; we only answer *who was
there*, which is exactly what our identity layer is good at (86% fused).

**Expected:** WHO on shots should approach our identity accuracy (~80%+) rather than
the 49% the frame-possession heuristic caps at. To be measured, not promised.

## Part B — Possession-change layer (rebound / steal / turnover)

Rides on the **possession timeline** (we have it) + the **shot events** (Part A):
- **REBOUND:** after a shot-detection MISS at `apex_t`, the first identity to gain
  sustained possession within `[apex_t, apex_t+4s]` near that rim. Team of the
  rebounder vs the shooter's team ⇒ offensive/defensive.
- **TURNOVER + STEAL:** possession changes **team** with no shot in between ⇒
  TURNOVER (losing ball-handler) paired with STEAL (gaining defender). Straight
  possession-timeline rule.
- **BLOCK:** a shot arc interrupted before apex (ball knocked down; the
  shot-detection flags "no clean arc" / abnormal early descent) with a defender at
  the ball ⇒ block. Hardest; **phase 2b** (defer until A + rebound/steal land).

Honest caveat: these inherit possession-signal quality, but the ones anchored to a
shot event (rebound especially) are far more reliable than free-floating ones
because the shot gives a precise time+place to look.

## Part C — Build order & effort

| Step | What | Data | Effort |
|---|---|---|---|
| A1 | Read shot-detection e6 output → per-shot `(t_release, R, made/miss)` | S3 tracks / rerun its pipeline on e6 | ~1 day (mostly its I/O format) |
| A2 | Release↔identity matcher + emit scoring events; score vs 219-play GT | our court positions (have) | ~1 day |
| B1 | Possession-timeline export (already computed in detect_events) → rebound + turnover/steal rules; score | ball cache + tracks (have) | ~1-2 days |
| B2 | Block detection (phase 2b) | shot arcs | later |

**Gate everything on the 219-play GT**, same as v1, per event type.

## Step 1 runtime — how to run THEIR make/miss pipeline on e6 (verified by code read)

Repo: `../uball_shot_detection_dual_fusion_v2`. Their pipeline, in order:
1. `pipeline/extract_tracks.py --game-id <e6-uuid>` — needs:
   - **Frozen detector bundle** from S3 (`FROZEN_BUNDLE_S3` in `pipeline/common.py`;
     it's the immutable v1 far+near YOLO weights + config, sha256-verified).
   - **e6 in their manifest** (`data/games_manifest.json`): each Game has
     `game_id`, `s3_prefix` → `video_s3_uri(game, angle, bucket)` locates the 4
     videos in `uball-videos-production`. ADD e6's row (gid + court-a prefix) if
     absent.
   - **Shot windows from Supabase**: `load_gt_shots(game_id)` reads the plays →
     `shot.buffered_window()`. e6 has 133 shots → 133 windows.
   - Detection is ultralytics YOLO; runs on MPS (`device="mps"`) locally or CUDA on
     their g4dn-spot AWS pattern (docs/04). Only shot windows are processed → fast.
   - Output: P1 tracks `s3://uball-cv-results/.../dual-fusion-v2/tracks/<gid>/…` (or
     local).
2. `pipeline/p2_dataset.py` → per-shot features (frac_inside_rim, arc fit, bounce-out,
   through_hoop) → `data/p2_features*.parquet`. Local CPU.
3. `pipeline/p3_angleaware.py` applies `data/p3_model_angleaware.joblib` (HGB, 0.955
   acc) → make/miss per shot. Local CPU.
4. Score make/miss vs the plays `classification` column (…_MAKE / …_MISS).

Env: their `far_angle` conda (ultralytics/opencv/torch) — or our .venv works for the
YOLO/parquet parts. Supabase via the plays REST (they use a service key in their .env).

**DECISION (user, 2026-07-13): far_v16 is RETIRED from our stack — the unified
corrected ball+hoop yolo26s is the shot detector.** The A/B below is therefore the
MANDATORY path (retrain P3 on our features), not optional; far_v16 survives only as
a fallback if our P3 misses the 0.949 LOGO benchmark. Prerequisite state: corrected
retrain (shot-frame split fix: train shot-frame hoops 0->1,415) must be fetched
(instance terminated) and hoop mAP verified on the CLEAN valid split (~0.90
expected); then REBUILD the full-game e6 ball+hoop cache with the new weights
(old-weights cache quarantined in runs/ball_cache_oldweights/).

**Detector A/B (now mandatory):** to use OUR retrained ball+hoop yolo26s instead of far_v16,
re-extract P2 features on their 185 labeled shots with our detector, retrain the P3
HGB (deterministic seed 42, whole-game LOGO split), and only adopt if it beats 0.949
LOGO. Their P3 was trained on far_v16 features, so a naive detector swap will shift the
feature distribution — retrain P3, don't just swap.

## What I need from you

1. **Access to run/read the shot-detection system's e6 shot outputs** — the
   per-shot ball trajectories (court `X_cm`) and made/miss. Its tracks are in S3;
   I may need to run its pipeline once on e6 to emit them, or point me at existing
   outputs. This is the one dependency for Part A.
2. Nothing else — **no web research** (this is all internal integration), and the
   logic building is on caches/systems we already own.

Confirm (1) and I start with A1 (read its output format) → A2 (the matcher + first
measured WHO-on-shots number vs the 49% baseline).
