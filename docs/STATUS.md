# Cross-Camera Basketball Analysis — Progress

_Last updated: 2026-07-03_

## What the system does
Turns **four fixed-camera videos of a game** into **descriptive, timestamped play-by-play** —
automatically. It figures out *who* is on court (team + jersey), *where* everyone is, and *what
happened* (possessions, passes), and writes it up like a commentator. It runs on the existing
4-camera rig (far-left, far-right, near-left, near-right).

| Stage | What it does | Status |
|---|---|---|
| **1. Detect** | Find every player, referee, ball in each camera, every frame | ✅ Working, measured (player 0.91 / referee 0.90 / ball 0.80 near-basket on held-out games) |
| **2. Track** | Follow each person over time + team + appearance signature | ✅ Working, hardened |
| **3. Fuse** | Merge 4 cameras into one map, one identity per player | ✅ **Working — validated on a second, untouched game** |
| **4. Events** | Deterministic "what happened" — possession, passes | 🔶 Engine ready; waiting on real ball tracking + a ground-truth check |
| **5. Narrate** | AI commentator writes play-by-play from the verified state | ✅ Working; grounding video (boxes + IDs burned in) built |

## Week in review — Mon Jun 29 → Fri Jul 3 (44 commits)

**Monday.** Published the first stage-by-stage status document; the freshly retrained
detector (ball 0.80, referee recovered to 0.90 on held-out games) became the pipeline default.

**Tuesday.** Annotation-tool friction fixes (nested boxes, ball-on-player duplicates); built
the event-anchored ball training set; first event-quality fixes (possession hysteresis,
stationary-false-positive filter).

**Wednesday.** Built motion-based ball detection + 4-camera motion fusion and wired it through
to events and narration; fixed team A/B (jersey-colour classifier + near-camera-only voting);
launched the jersey-number annotation round (crop extractor, labelling tool with tight
number boxes, resume + undo); built the set-of-marks grounding video for the AI commentator.

**Thursday — the audit.** Ran a full adversarial audit of every subsystem (re-measuring all
claims from saved artifacts + checking methods against published research), then fixed what it
found the same day: a silent camera-sync failure (now strict + measured: one camera ran 13
frames off), the Wednesday ball claim retracted honestly (the tracker followed players, not
the ball — now an explicit experimental flag), jersey data methodology corrected before more
annotation (queue now covers all 25 games, leak-proof train/val split, tool hardening), engine
hardening (identity over-count 28% of frames → under 7%), a ground-truth labelling tool so
events become measurable, and the jersey training stack (modern scene-text recogniser: 78%
zero-shot on our crops; fine-tune + legibility trainers smoke-tested, one command when labels land).

**Friday.** Lens (fisheye) correction refit for all four cameras — the audit's biggest
remaining error source — plus identity-fragment stitching: the benchmark window now yields
**16 identities for 13 people, all 13 stable, 1 phantom, both referees**. Cold-validated the
whole stack on a second, untouched game (matching quality; also caught + fixed a
black-vs-white-kit team-naming bug). Eliminated the court-map flicker (234 dropout gaps → 5;
players-per-frame now sits exactly at truth). Produced the client demo video (same player =
same number in every camera). Ran **SAM3** — an independent state-of-the-art model — over both
games on a cloud GPU as a cross-check: it confirms **91–99% of everything our pipeline
detects** (our precision is clean) and quantified the known far-distance camera recall gap,
which the 4-camera fusion design already compensates; its disagreement frames double as a
free hard-example list for the next detector improvement round.

## The audit in detail: audit → fix → re-validate
We ran a **full adversarial audit** of every subsystem (re-measuring all claims from saved run
artifacts, checking methods against current published research), then fixed what it found and
**re-measured**. Highlights:

**Camera identity quality is now close to ground truth.** Three root causes were found and fixed:
the cameras were being fused without verified time-sync (now measured and enforced — one camera
ran 13 frames off), the fisheye lenses were never corrected before mapping players onto the court
(now corrected with a per-camera lens model — near-camera position error dropped from ~32–41 cm to
~11–23 cm), and identity fragments are now stitched back together. Measured result on the
benchmark window:

| | Before audit | Now | Reality |
|---|---|---|---|
| Identities created for ~13 people | 30 | **16** | 13 |
| Real people tracked ≥ half the clip | 12 | **13 (all)** | 13 |
| Phantom identities | 6 | **1** | 0 |
| Team identities for 10 players | 22 (+7 unknown) | **13, 0 unknown** | 10 |
| Referees identified | 1 | **2** | 2 |

**Validated on a second game it was never tuned on.** The full pipeline cold-ran on another
game's footage (black-vs-white kits, full-court sequence): **15 identities / 12 stable / 1
phantom / clean 5-v-5 + 3 referees roster**, matching the benchmark profile. The run also caught
and fixed a cross-game bug: black/white kits have no "colour", so the team-naming rule now falls
back to brightness — every camera names the teams consistently regardless of kit colour.

**Jersey numbers — plan upgraded and de-risked before spending annotation.** Following what wins
the SoccerNet jersey-recognition challenge, the recognizer is now a **scene-text model (reads any
number, even ones never seen in training)** gated by a legibility check, rather than a fixed-list
classifier. The pretrained model already reads **78% of our annotated crops with zero training**;
fine-tuning machinery is built and smoke-tested (one command when labels are ready). Annotation
quality fixes landed first: the labeling queue now covers **all 25 games evenly** (was: one game),
the train/validation split is leak-proof, and previously-discarded "unclear" marks now train the
legibility gate. Every label counts now.

**Honesty on the ball.** The earlier full-court motion-fusion ball did not survive the audit —
it followed players, not the ball — so it is now an explicitly experimental option, and events
degrade honestly without it. The literature-backed replacement (a **trained** motion-aware ball
detector) is the next model to build, using the same data flywheel that fixed detection. Ball
detection near the basket (where shots happen) already works (0.80 AP).

**Everything is measurable from here.** A ground-truth labelling tool now exists: ~20 minutes of
operator time per clip turns every future possession/pass claim into a measured precision/recall
number instead of an eyeball check.

## Where we are
- **Working and cross-game validated:** detection → tracking → 4-camera fusion (stable
  identities, correct teams, referees) → grounded narration, with verified sync and corrected
  lens geometry. 38 automated tests pass.
- **In progress (operator):** jersey-number annotation across all games (99 usable spanning 8
  games so far, of a ~300–400 target) and the first ground-truth event labels (not started).
- **Next, in order:** (1) fine-tune the jersey reader on the multi-game labels → named players
  end-to-end, (2) trained motion-aware ball detector → trustworthy possession/pass events,
  (3) score events against ground truth and tune the remaining occlusion under-count.
  _(Shot make/miss stays out of scope — separate repo.)_

**One-line summary:** the identity layer (who/where, teams, refs) is fixed, measured, and
validated on unseen footage; jersey names and a real ball tracker are the two remaining builds,
and both have their data pipelines, training code, and evaluation tooling already in place.
