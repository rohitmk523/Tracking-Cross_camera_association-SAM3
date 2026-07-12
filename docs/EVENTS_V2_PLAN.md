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

## What I need from you

1. **Access to run/read the shot-detection system's e6 shot outputs** — the
   per-shot ball trajectories (court `X_cm`) and made/miss. Its tracks are in S3;
   I may need to run its pipeline once on e6 to emit them, or point me at existing
   outputs. This is the one dependency for Part A.
2. Nothing else — **no web research** (this is all internal integration), and the
   logic building is on caches/systems we already own.

Confirm (1) and I start with A1 (read its output format) → A2 (the matcher + first
measured WHO-on-shots number vs the 49% baseline).
