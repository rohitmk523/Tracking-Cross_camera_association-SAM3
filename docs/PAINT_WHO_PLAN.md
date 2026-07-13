# Crowd/paint attribution — the planned effort (2026-07-13)

**The problem.** One failure mode caps three numbers at once: shooter-WHO on
close shots (FG 17/47 = 36%), rebounder-WHO (15/44 = 34%), and therefore the
complete-event score (56%). Long/isolated shots are already good (4PT 88%,
FT ~80%); everything hard happens where bodies overlap. Per the user: solve
this properly FIRST, then tune remaining metrics, and only then run c2a —
**c2a stays untouched (blind) until this lands.**

## What we know (all measured on e6, full game)

1. Attribution evolution so far: nearest-body single-frame 60% → integrated
   ball-holder over [rel−0.8s, rel] 65% → +release walk-back (set point, not
   global-lowest which latched onto PASSES). Each step measured; kept.
2. Error anatomy (12 sampled FG errors): **8/12 the correct player's track IS
   present near the ball at release** (pure logic/ambiguity failure — fixable),
   **4/12 the track is absent** (occlusion in the scrum — needs recovery, not
   better scoring).
3. Wrong picks are contesting defenders and adjacent teammates — 2D "under the
   ball" cannot separate raised hands along the camera ray.
4. The possession signal itself flips ~200×/game at team level (why
   turnover/steal stay experimental).
5. Ball coverage is NOT the limiter (proved in v1: detector upgrade moved WHO
   <1pt).

## Signals we own but do not use yet

| # | Signal | Why it should discriminate |
|---|---|---|
| S1 | **RTMPose wrists** (stack already built for ankles) | the shooter's WRISTS are at the ball at release; box-top is a crude proxy that ties with defenders |
| S2 | **Cross-camera agreement** (we attribute on the arc cam only) | a defender occludes along ONE ray; four rays rarely agree on the wrong body |
| S3 | **KPR embedding on the release crop** | identity from appearance directly — works even when the identity TRACK is absent (the 4/12 class) |
| S4 | **Jersey read at release** | a legible number in the release window is near-ground-truth |
| S5 | **Ball-motion vector before the set** | pass INTO the crowd → receiver is the shooter; ball moving WITH a body → that body is the handler |
| S6 | **Raw detector boxes** (dets cache) at release | candidate set even where tracks dropped; assign identity via S3/S4 |

## Experiment plan (fixed harness, adopt only on measured gains)

**Harness (H0):** one command scores WHO on the same 142-shot + 54-rebound GT
with per-class breakdown. Every experiment reports ΔWHO-overall, ΔWHO-FG,
ΔWHO-rebound and must NOT regress 4PT/FT. All thresholds frozen before c2a.

- **E0 — Oracle ceilings first (half day).** (a) oracle-cam: score correct if
  ANY camera's candidate is right → the cross-cam ceiling for S2. (b) oracle
  window: any frame ±0.3s → timing ceiling. (c) oracle-track: GT shooter's
  track present at release → the pure-logic ceiling (~87% expected from the
  2/3 error anatomy). These bounds decide where the effort goes; nothing is
  built that
  its oracle says can't pay.
- **E1 — Wrist attribution (1 day).** RTMPose on release-window crops only
  (134 shots × ~10 frames × candidates — minutes on MPS). Score = ball-to-wrist
  distance; keep torso gate. Gate to adopt: FG WHO +5pts minimum.
- **E2 — Cross-cam release vote (half day).** Run the E1 scorer per camera at
  the release instant; weighted vote (weights: ball size in that cam, margin
  between best and second candidate). Gate: +3 overall without 4PT/FT loss.
- **E3 — KPR release-crop identity (1 day).** Embed the region under the ball;
  match against the game's 13 identity prototypes (restricted unpickler rules
  apply). Primary target: the track-absent third. Gate: recovers ≥ half of
  track-absent errors on the sampled set.
- **E4 — Pass-vs-hold discrimination (half day).** Ball velocity vector in
  [rel−1.5s, rel]: incoming-pass trajectories point INTO the candidate; a
  handler's ball moves WITH him. Kills the receiver-vs-passer and
  tip-scramble confusions the walk-back still admits.
- **E5 — Fusion (half day).** Combine adopted signals into one scored vote
  (weights fit on first half, held out on second — same split discipline as
  the zone calibration). This is the number that goes in STATUS.md.
- **Rebounds inherit E1-E5** via the same scorer on the post-miss window
  (longest-hold stays as the segment picker; the scorer ranks WHO within it).

## After the WHO effort (in order, still before c2a)

1. Zone pass 2: re-fit release-distance boundaries WITH the newly-attributed
   paint shots (zone errors correlate with WHO errors today); FT label gets
   the isolation check (two close FGs currently mislabel as FT).
2. Shot-ness gate: P3-probability + arc-quality filter for the 83 CV events
   that match no GT shot (real putbacks stay, rim-pass noise drops).
3. Reel v3 + STATUS.md update with the post-effort table.

**Then and only then: c2a, fully blind** — zones, thresholds, weights all
frozen from e6; c2a plays GT used exactly once, for scoring. That is the
generalization test and it is worthless if we peek.
