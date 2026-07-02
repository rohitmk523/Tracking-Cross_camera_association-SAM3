# Cross-Camera Basketball Analysis — Progress

_Last updated: 2026-07-02 (post-audit)_

## What the system does
Turns **four fixed-camera videos of a game** into **descriptive, timestamped play-by-play** —
automatically. It figures out *who* is on court (team + jersey), *where* everyone is, and *what
happened* (possessions, passes), and writes it up like a commentator. It runs on the existing
4-camera rig (far-left, far-right, near-left, near-right).

## 2026-07-02: full adversarial audit
Before scaling up, we ran a deep audit of every subsystem — re-measuring all claims from the
saved run artifacts and checking our methods against current published research (SoccerNet
challenges, CVPR sports workshops, production ball-tracking systems). Several claims below were
**corrected** as a result. That is the point of an audit: we now know precisely what works, what
doesn't, and what to build next — before spending more annotation and GPU budget.

| Stage | What it does | Audited status |
|---|---|---|
| **1. Detect** | Find every player, referee, ball in each camera, every frame | ✅ Working, **measured** (player 0.91 / referee 0.90 / ball 0.80 near-basket, held-out games) |
| **2. Track** | Follow each person over time + team + appearance signature | ✅ Working with known caveats (player↔referee flicker; ID fragmentation) |
| **3. Fuse** | Merge 4 cameras into one top-down map, one identity per player | 🔶 Core works (12 stable IDs on the test window); precision limits found + being fixed |
| **4. Events** | Deterministic "what happened" — possession, passes | 🔶 Logic sound; **blocked on real ball tracking** (see below) |
| **5. Narrate** | AI commentator writes play-by-play from the verified state | ✅ Working; grounding video (boxes+IDs burned in) built |

## What the audit confirmed as solid
- **Detection** is the one subsystem with real measured accuracy on held-out games — the
  event-anchored data pipeline (game DB → shot frames → annotate → retrain) works and is the
  template for everything else.
- **Per-camera tracking** (ByteTrack) is standard practice, correctly implemented; 25/25 tests pass.
- **The fusion architecture** (project all cameras to one court plane → match → track) is the
  established pattern in the literature, and the calibration-region gate was a genuine fix
  (over-count 16.3 → ~13 players/frame).
- **12 on-court players get one stable global identity** ≥50% of the clip on the e6 window (real,
  re-verified). Transient ghost identities remain (30 total IDs for ~13 people).
- **Camera sync is now verified, not assumed**: FR runs **−13 frames** vs FL; NL/NR are <1 frame
  off. Sync failures now stop the pipeline loudly instead of silently fusing unaligned cameras.

## What the audit corrected

**Ball tracking (the big one).** The earlier "full-court ball tracking works" claim did not
survive measurement. The motion+agreement tracker follows **players** (their moving limbs
generate most motion candidates, and multiple cameras "agree" on players just as well as on the
ball): the produced trace sits a median 77 cm from the nearest player, covers ~24% of the court
(not full-court), and the "740 cm travel" figure traced to a different, discarded artifact.
- **Why it's still progress:** motion *is* the right cue for a ball this small (confirmed by the
  literature — TrackNet/WASB), and the per-camera motion + sync + fusion machinery is built and
  tested. What's missing is a **trained** ball scorer on top of the motion signal instead of
  hand-tuned color/size rules, and 3D-aware cross-camera geometry (a flying ball violates the
  flat-court assumption).
- **Short-term honest path:** near-basket ball detection (AP 0.80, already trained) + possession
  from player trajectories; **medium-term:** train the motion-aware ball detector (the same
  annotate→train flywheel that fixed detection).

**Events.** The possession/pass/turnover engine is logically sound and degrades honestly without
a ball — but fed the flawed ball trace it over-fired (11 possession changes in 12 s). It will be
re-validated once the ball input is real, against a small hand-labeled ground-truth set.

**Team labels.** Correct on near cameras (verified); far cameras cannot separate these teams by
color and are rightly excluded from voting. Robustness across arenas/lighting needs a stronger
method eventually (tracked in backlog).

**Jersey numbers (identity).** Direction confirmed — but the plan was upgraded to match what
wins SoccerNet's jersey-recognition challenge: a **scene-text-recognition model (PARSeq)
fine-tuned on our crops** (reads *any* number, including ones never seen in training) gated by a
**legibility classifier**, instead of a fixed-list classifier that can only output numbers it was
trained on. Every annotation made so far feeds the new plan (nothing wasted), and three data
fixes landed before further annotation: the labeling queue now interleaves **all 25 games**
(was: one game), the train/val split is leak-proof (same play can't appear on both sides), and
the "unclear" labels now train the legibility gate instead of being discarded.

## Where we are
- **Working end-to-end:** detection → tracking → cross-camera player fusion → grounded narration,
  with verified camera sync and honest per-stage caveats.
- **In flight:** jersey numbers (PARSeq zero-shot baseline running on the first 91 crops — it
  decides how much annotation is needed), annotation continuing across all 25 games.
- **Next, in order:** (1) jersey recognizer (STR fine-tune + legibility gate + tracklet voting),
  (2) trained motion-aware ball detector, (3) a small hand-labeled ground-truth set (ball
  positions + possession/pass labels on 2–3 windows) so every future claim is measured, not
  eyeballed, (4) fusion hardening (per-track class votes, fisheye undistortion, re-entry gate).
  _(Shot make/miss stays out of scope — separate repo.)_

**One-line summary:** players, teams and cross-camera identity work and are now precisely
characterized; narration is grounded; the ball is the one honest blocker, with a
literature-backed plan (trained motion-aware detection) and the data flywheel to execute it.
