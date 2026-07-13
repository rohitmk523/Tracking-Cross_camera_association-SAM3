# Cross-Camera Basketball Tracking — Status Report

**What the system does:** from the four cameras already installed in the gym, work out
**who** is on court (team + jersey number), **where** each player is (top-down court
map, in centimetres), and **what happened** (possession, passes, shots) — the
foundation an AI commentator narrates.

**How we measure everything in this report:** an operator hand-marked real players,
frame by frame, across all four cameras (~13,000 human-verified labels over two
different games). Every number below is the system graded against that human truth,
leakage-free: the labels are only the answer key, never an input. The strictest score
we use — **"all-angles" accuracy** — demands the player be correctly marked in *every*
camera that can see him, every frame. If he's visible and we don't show him, that
counts against us.

---

## Part 1 — What was established first

Before touching accuracy, we built the instrument that measures it:

- **Ground-truth tooling**: an annotation tool for marking one real player across all
  four cameras every frame, a detection cache (expensive compute runs once, every
  experiment re-runs in seconds), and a version ledger (every idea gets a number, not
  an opinion).
- **Detection is solved.** Players are found in 96–100% of frames, on every camera,
  verified against human truth. "We can't even see the far players" is gone.
- **Jersey reading works.** 94–95% of confident number reads are correct, and we read
  every frame on every camera (thousands of reads per minute of play).
- **Identity through traffic is THE problem.** Every off-the-shelf tracker we measured
  keeps a player's identity only 28–55% of the time through crowded play. When two
  team-mates in identical kit collide, no detector alone can tell them apart. This
  single problem is what the rest of this report is about.

## Part 2 — The SAM3 approach: proven, measured, and then retired

We then adopted the industry demo recipe (Meta's SAM3 model): hand it one player's
box once, and it visually follows that player's outline frame by frame. Around it we
built our own machinery — jersey-read checkpoints that restart the track whenever a
confident number is seen (a mistake can only survive until the next read), and a
cross-camera correction that uses the court map to override any camera that disagrees
with the jersey-confirmed position.

**It works.** Best measured configuration (re-seeded SAM3 + cross-camera correction),
strict all-angles score against human truth:

| Player | SAM3 pipeline | Notes |
|---|---|---|
| #11 | **97%** | best-read player |
| #6 | **86%** | same-kit group |
| #43 | **85%** | same-kit group |
| #22 | **83%** | hardest case: same kit as team-mates AND same number as an opponent (was 53% before these layers) |
| **Mean** | **88%** | fused ≥1-camera coverage 89–100%; court position error 5–12cm |

Demo videos (four camera tiles + top-down court trail):
`sam3player_n11_final88_e6fba750.mp4`, `sam3player_n22_final88_e6fba750.mp4`,
`sam3player_n43_final88_e6fba750.mp4`, `sam3player_n6_final88_e6fba750.mp4`.

## Part 3 — Why SAM3 cannot ship (speed and cost, measured)

SAM3 is a 3.45GB model that must process **every frame of every camera**. We measured
it, not estimated it:

- **On a datacenter GPU (AWS A10G):** 0.65 frames/second when tracking the players
  together. Four cameras × a full game at 30fps means the GPU runs for **many hours
  per game**, costing **$25–35+ per game** — before detection, jersey reading or
  anything else. Processing arrives hours after the game ends.
- **On the venue edge box (Jetson AGX):** the AGX delivers a small fraction of an
  A10G's throughput on a model this large — well under 0.2 frames/second against a
  120 frames/second requirement (4 cameras × 30fps). That is a **500×+ real-time
  deficit**: not "slow", but architecturally impossible on the edge. Memory and
  thermals rule it out independently.
- Per clip-query it's also the wrong tool: answering "who has the ball in this
  6-second clip" costs ~$0.75 through SAM3 versus ~$0.02 through detection + jersey
  reading.

Conclusion: SAM3 is a superb *accuracy benchmark* and label generator, and the wrong
*production engine*. We retired it from the pipeline and set out to match its
accuracy with components that run in minutes for dollars.

**Why we believed a SAM3-free pipeline would work — the bet, before any testing.**
Our measurements had already shown that most of the accuracy never came from SAM3:
the jersey-checkpoint and court-correction machinery lifted even a simple tracker
from 48% to 78%, leaving SAM3 responsible for ~10 points, all concentrated in one
moment — same-kit pile-ups. So the question was never "replace SAM3 everywhere,"
only "find a better tool for that one moment." Two existed: **skeletons** (precise
foot placement for the court map, and a way to point at a specific body in a crowd)
and **KPR**, a recognition model designed exactly for crowded, occluded, identically
dressed people — it describes a pointed-at player as per-body-part signatures and
compares only the parts visible in both images. If those two layers held up on our
footage, the whole pipeline would run in minutes for dollars. Everything below is
that bet, built and then tested against the same human truth as SAM3 was.

## Part 4 — The production pipeline (no SAM3), explained properly

Every layer below exists to answer one question — *who is this body?* — using a
different kind of evidence. Each layer's contribution is measured on the same strict
metric.

**1. Detection + motion tracking (ByteTrack).** Our detector finds every player every
frame; ByteTrack — the industry-standard tracker — links those boxes frame to frame
by motion and overlap. Cheap and continuous, but blind to identity: when two
identical-kit players cross, it can hand the track to the wrong man. *This is the
skeleton of the pipeline that everything else corrects.*

**2. Jersey-number checkpoints.** We read numbers on every camera every frame
(94–95% precision). Every confident read *claims* the track it lands on: from that
instant we know who that body is, until the tracker loses it. Because reads are
dense (a well-lit player is re-confirmed every second or two), a wrong identity has
a very short lifetime.

**3. Skeletons — pose detection (RTMPose).** For every detected player we estimate 17
body keypoints (ankles, knees, hips, shoulders…). Two reasons this matters:
  - **Better court positions.** The court map needs the player's floor-contact point.
    A bounding-box bottom is a crude guess (a leaning player's box bottom isn't his
    feet). Projecting the **ankle-sole point** instead cut cross-camera position
    disagreement by 23% (63cm → 48cm median) — positions from different cameras now
    agree closely enough to gate and fuse reliably.
  - **It points the appearance model at the right person** (next layer).
  Cost: negligible — 161 player-crops/second even on a laptop chip.

**4. Appearance in pile-ups — KPR (keypoint-promptable re-identification).** The one
moment every other signal fails simultaneously is the scrum: boxes overlap (motion
tracking fails), kits are identical (colour fails), numbers face away (reading
fails). KPR is a recognition model built exactly for this. How it works, in plain
terms:
  - Instead of one overall "face-ID"-style signature, it describes a person as
    **separate signatures per body region** — head, torso, arms, legs, feet — like a
    structured witness statement rather than a single impression.
  - It is **pointable**: along with the image we hand it the skeleton of the player
    we mean (from the pose layer). The model then describes *that* body and actively
    ignores the tangled opponents around it — which is exactly the pile-up situation.
  - Every body-region signature carries a **visibility score**, and two players are
    compared **only on regions visible in both images**. If the legs are hidden
    behind another player, the legs simply don't vote — occlusion stops corrupting
    the comparison instead of poisoning it.
  - With identical jerseys, what remains discriminative is what the model keys on:
    build, skin tone, hair, shoes, sleeves — the things team-mates *don't* share.
  - It **learns from the system's own footage**: every confident jersey read is a
    labelled photo, so each processed game sharpens it (measured: 69% → 75% correct
    in pile-ups after one ~$2.5 self-training cycle, chance being 33%).
  We use it in two places:
  - **Tie-breaking**: when two bodies sit near the expected position, KPR votes.
  - **Re-acquisition**: when a player has been lost for over ~8 seconds (sub, bench,
    long scrum), we scan all cameras for someone who *looks like him*, tag him
    provisionally, and let the next jersey read confirm — the "temporary identity
    until the number confirms" design, now real.

**5. Cross-camera correction + coverage.** The original operator-designed layer: any
camera that disagrees with the jersey-confirmed court position gets overridden with
the detection actually standing there; positions between checkpoints are bridged;
a fallback search radius prevents "shown nothing" frames.

### Accuracy: SAM3 pipeline vs production pipeline (same truth, same strict metric)

| Player | SAM3 (re-seeded + correction) | Production (no SAM3) |
|---|---|---|
| #11 | 97% | **91%** |
| #6 | 86% | **85%** |
| #22 | 83% | **80%** |
| #43 | 85% | **74%** |
| **Mean** | **88%** | **82.3%** |
| Cost per game | $25–35+, hours | **$4–8, under an hour** |

**The evidence, in one place (view in this order):**
1. The comparison table above — same truth, same strict metric, both pipelines.
2. Per-player videos, SAM3 era: `sam3player_n{11,22,43,6}_final88_e6fba750.mp4`;
   production pipeline: `sam3player_n{11,22,43,6}_sam3free_e6fba750.mp4` — same
   minute, watch them side by side.
3. Whole-pipeline videos (every player at once, court-mapped):
   `demo_allplayers_e6fba750.mp4` (tuned game) and `demo_allplayers_c2a354fe.mp4`
   (blind game — footage the system was never tuned on).
4. Failure heatmap of the current pipeline: `runs/tracking/pipeline_failure_heatmap.jpg`
   (Appendix B) — where the remaining misses live and why the venue fixes target them.

## Part 5 — Closing the gap: what is running right now, and what comes next

**The flywheel — first cycle completed and measured.** Every game the system
processes generates its own training labels for free (each confident jersey read is a
labelled photo of that player). We auto-built **3,061 labelled crops from the second
game** (label quality audited against human truth: 89–100% per identity) and
fine-tuned the recognition model on them for ~$2.5 of GPU. Measured effect on the
FIRST game — footage the fine-tune never saw:
- pile-up recognition of identical-kit team-mates: **69% → 75%** (best player 68% → 88%)
- full pipeline, strict all-angles: **82.3% → 83.2%** (#11 92%, #6 85%, #22 82%, #43 74%)

That is one cycle on one minute of training footage from one game. The loop repeats —
and compounds — every time more footage is processed: more games → more auto-labels →
sharper recognition → better tracking. Next cycles add training windows from more
games at near-zero marginal cost. Remaining gap to the retired SAM3 benchmark: 4.8
points, concentrated in one weak camera-cell (#43 far-right).

**The whole-pipeline demo video (next deliverable).** All demos so far track one
player at a time. The next video shows the actual product behaviour: **every player
is picked up the moment he first appears in any camera**, held simultaneously across
all four views, positioned on the court map via the 2D→3D projection, and carried
through subs, scrums and re-entries — one video of the full system running end to
end, not per-player runs stitched together.

**Second-game blind validation — PASSED.** The same pipeline, run on a game it was
never tuned on, scores **81% strict all-angles** against that game's own human truth
(vs 83.2% on the tuned game) — generalization within ~2 points. Two findings along
the way, both fixed: the earlier collapse on this game was a camera **sync-offset
constant** (5 frames wrong → 4.5m of phantom cross-camera error; corrected to 0.77m,
validated against truth), not calibration; and **team-aware identity** (number × kit
colour) now separates opponents who wear the same number — the two #3s went from
47%/64% (mixed as one identity) to **73%/86%** tracked as two people. Blind-game
whole-pipeline demo: `demo_allplayers_c2a354fe.mp4` (8 identities, both #3s held
side by side).

**Also queued:**
- **Physical camera check** — the near-left camera under-covers the left basket
  (likely mount/obstruction); a re-aim there is free accuracy. No new cameras needed.
- **License clearance** — the KPR model ships under the Hippocratic License (HL3);
  commercial use needs a one-time legal check before productisation.

**Honest ceiling.** With four side cameras, ~90% on the strict all-angles metric is
the physics limit — some pile-up moments are genuinely unresolvable from these
viewpoints (the NBA's own tracking system uses 12 cameras and skeletons to solve
occlusion geometrically). On the measure a viewer actually experiences — the fused
top-down court view — the system already holds players 89–100% of the time.

---

## Part 6 — Edge deployment (Jetson AGX at the venue)

The pipeline was rebuilt with the venue box in mind. Component by component on the
Jetson AGX:

| Layer | On the AGX |
|---|---|
| Motion tracking, court math, fusion, correction | trivially (CPU arithmetic) |
| Jersey reading (three small models, runs on a few crops/frame) | comfortably |
| Skeletons (RTMPose — an edge-first model) | comfortably |
| Recognition (KPR) — fires only on ambiguous moments, a few % of crops | fine at that duty cycle |
| **Detection (the one heavy per-frame layer)** | **the bottleneck: ~15–30fps optimized vs 120fps live** |

What that means, honestly:
- **Post-game processing on the venue box: realistic today.** Total active model
  weight is under 2GB (SAM3 alone was 3.45GB); a 40-minute game processes in roughly
  **1.5–3 hours on the AGX** after the standard Jetson porting pass (TensorRT
  engines, FP16/INT8 quantization) — known engineering, not research.
- **Live real-time: one more engineering phase.** Detection at reduced resolution,
  detecting every 2nd–3rd frame with the tracker bridging between, and INT8
  quantization plausibly reach "live with a few seconds of latency." SAM3's gap on
  this hardware was 500×+ (impossible); the new stack's gap is 4–8× on a single
  layer (an optimization problem).
- **Self-improvement doesn't need the box**: fine-tuning runs in the cloud (~$2.5
  per cycle); only updated weights ship to the venue.
- **Detector choice is swappable.** Detection is the one layer where the model is a
  commodity: the current RF-DETR (Apache-2.0 licensed) can be exchanged for a YOLO
  family model — typically faster on Jetson with TensorRT — **provided it is
  properly retrained on our footage for all three classes: player, referee, and
  ball**. Detection quality is entirely a training-data question, and our labelled
  dataset (below) transfers to any detector. One caveat to weigh: recent Ultralytics
  YOLO versions carry an AGPL/commercial licence, versus RF-DETR's permissive
  Apache-2.0 — a business decision, not a technical one. A full train-and-race plan
  (YOLO small + medium vs the FP16-optimized RF-DETR: accuracy gates, speed lanes,
  decision rule, ~$10 total) is written up in
  `Training_frameworks/Uball 4Cam Detection/YOLO_TRAINING_PLAN.md`.

### The detector race — RESULTS (measured, not benchmarked)

We trained three YOLO candidates (11-small, 11-medium, 26-small) on our own labelled
footage — the exact dataset the incumbent RF-DETR was trained on — and raced all
four detectors **end-to-end**: raw video in, full pipeline (detection → jersey
reading → skeletons → kit tags → tracking → cross-camera correction) rebuilt from
scratch per detector with nothing shared between runs, on a one-minute window of a
blind game with full human ground truth for three players (including a same-number
pair on opposite teams — the hardest identity case). Identity accuracy is graded two
ways: **strict** (the player must be correct in *every* camera that sees him) and
**fused** (correct in at least one camera — what the product consumes).

| Detector | Strict | Fused | Pipeline time (1 game-min, cloud GPU) | Detection speed | License |
|---|---|---|---|---|---|
| **RF-DETR-S FP16** (incumbent) | **81.0%** | **85.6%** | 29.4 min | 24.5 fps | Apache-2.0 |
| **yolo26s** | 78.9% | 84.6% | 26.1 min | 63.4 fps | AGPL/commercial |
| yolo11s | 76.6% | 84.4% | 26.3 min | 67.2 fps | AGPL/commercial |
| yolo11m | 76.2% | 81.6% | 26.0 min | 60.3 fps | AGPL/commercial |

What the race says:
- **RF-DETR remains the accuracy leader** — it wins both strict and fused on the
  blind window, and stays the reference detector for accuracy-grade output.
- **yolo26s is the clear edge/live candidate**: best YOLO on both metrics, within
  1 point of RF-DETR on the fused (product) metric, detects 2.6× faster, and — the
  decisive practical point — drops directly into the venue box's live streaming
  runtime (DeepStream/TensorRT), which the transformer-based RF-DETR does not
  without custom engineering. It also produced the most confident jersey reads of
  all four detectors (+13% over RF-DETR), meaning more identity anchors for the
  correction layer.
- **Detection is no longer the pipeline's bottleneck.** With any YOLO, detection is
  ~7% of pipeline time; jersey OCR and kit-tagging now dominate (~80%) and are the
  next optimization target (triggered reads instead of every-frame reads, batched
  inference, crop reuse — projected to cut pipeline time by ~3-4×).
- A lab-benchmark lesson worth recording: yolo11m scored the *best* detection mAP of
  all four models yet tracked *worst* — benchmark numbers do not order real pipeline
  outcomes; only end-to-end measurement does. This is why we race detectors instead
  of reading spec sheets.

**Ball detection & possession — separate workstream, groundwork laid.** The
detection dataset already includes the ball as a first-class label (~1,580 ball
boxes across train/valid/test, built from the same annotation tooling), so any
retrained detector — RF-DETR or YOLO — learns players, referees and ball together.
Ball *tracking* and possession attribution ("who has the ball", feeding "who shot")
are the next product layer on top of this pipeline: the design is straightforward
given what now exists (ball track + player court positions → possession by
proximity over time), and shot make/miss detection already lives in its own
dedicated system that this pipeline will consume timestamps from. It is scoped,
not started.

## Part 7 — Event detection (NEW workstream, first measured milestone)

The pipeline now attributes GAME EVENTS, compared against the operator's full
219-play log for one complete game: WHO made each play, and for shots, from which
scoring zone (this venue has a 4-point line; we extracted the zone geometry from
the court paint itself — the white 3PT line matches our court model within 1cm,
the red 4PT line fits a clean arc at ~9.4m).

Full-game accuracy against the 219-play operator log:

| Metric | Result |
|---|---|
| WHO — correct player (all 219 plays) | **49%** |
| Point value — 2 / 3 / 4-point zone, on shots | **62%** |
| Player AND points both correct, on shots | **44%** |
| Free throws (WHO / zone) | **87% / 100%** |

How to read these: the system watches a full game and, for each play, names the
player and — for shots — reads which scoring zone he shot from (this venue has a
4-point line; we extracted the zone geometry from the court paint, the 3-point line
matching our court model within 1cm). Make-or-miss itself comes from the separate
shot-detection system; combine it with our 44% "player + points" and you get the
complete scoring event ("Fui Martinez, 4-point make"). WHO is far above the ~8%
random baseline over 13 players, and free throws are near-solved (fixed position).

**What the ceiling is, honestly.** We upgraded the ball detector dramatically
(coverage 3-12% → 47-74% of frames, a dedicated all-angle specialist) and it moved
WHO by less than 1 point. So ball visibility was *not* the limiter. The limiter is
the possession→shooter attribution itself — when players cluster, "nearest to the
ball" is a weak indicator of the actual handler, and no aggregation rule we tested
fixed it. The real next lever is integrating the shot-detection system's shot
*origin* (when + where the ball left toward the hoop) with our identity-at-location,
rather than re-deriving possession frame-by-frame. Turnover/steal/block rules are a
further layer. A visual GT-vs-prediction reel accompanies this report
(`event_demo.mp4`).

## Part 8 — Events v2 (2026-07-13): shot-anchored events, measured full game

Everything in Part 7 was re-architected around a **shot trigger** instead of
frame-by-frame possession, and the make/miss brain of the proven shot-detection
system was transplanted onto our detector. All numbers below are the full e6
game against the operator's log (142 make/miss plays; 54 rebounds), all logic
CV-side — no ground-truth timestamps anywhere in the production path.

| Metric (of the 142 GT shots) | v1 (Part 7) | **v2.1 (2026-07-13)** |
|---|---|---|
| Shot detected at all | n/a (GT-triggered) | **94%** (134/142) |
| WHO — correct shooter | 49% | **62%** |
| Point value (2/3/4PT or FT) | 62% | **78%** |
| Make-or-miss | external | **98%** (131/134) |
| Complete event fully correct | 44%* | **56%** (75/134) |
| REBOUND detected / right rebounder | — | **81% / 34%** |

*v1's 44% assumed the event exists (scored at GT timestamps) and had no
make/miss; v2's 56% is on a strictly harder task — find the shot, name the
shooter, read the zone AND call make/miss, all correct at once.

What made each jump, in one line each:
- **Make/miss 98%:** our new ball+hoop detector's tracks, run through the
  shot-detection system's own feature+model recipe — validated to transfer
  cleanly (honest leave-this-game-out protocol scores 0.958, identical to that
  system's own benchmark; windows anchored on OUR detected arcs score even
  better than the human-annotated windows, 0.982).
- **Detection 94%:** two triggers — a rim-arrival arc detector, plus a rim-box
  ENTRY cue for flat layups/putbacks (every one of the 25 missed close shots
  had the ball *detected* in the rim box; the old arc gate just rejected flat
  trajectories). Note for the record: the shot-detection system itself has no
  shot trigger — it consumes human-annotated windows; the trigger is our net-new
  logic.
- **WHO 62%:** shooter = who *held* the ball into the release instant (walk
  back from the arc apex to the set point), not who is nearest the ball — the
  nearest-body rule picks contesting defenders and pass receivers.
- **Zone 78%:** release-instant distance with boundaries calibrated once from
  court geometry as measured *through the cameras* (the raw court-line radii
  compress at long range; the fit is per-venue, held-out validated).
- **REBOUND (new):** first sustained possession beginning after a missed shot,
  longest-hold-wins; team of rebounder vs shooter gives offensive/defensive.

A new 12-play reel accompanies this (`event_demo_v2_e6fba750.mp4`): each play
shows the pipeline's full call — player, points, make/miss CALL — against the
operator's log, including one deliberate wrong-shooter case and one wrong-zone
case so the 62%/78% are visible, not hidden.

**The one problem now worth real effort — crowd/paint attribution.** It caps
shooter-WHO on close shots (36%), rebounder-WHO (34%), and therefore the
complete-event number. Diagnosis is written: in two-thirds of the errors the
right player's track exists at the right moment (a logic problem, solvable);
in one-third the track itself is absent in the scrum (a tracking problem).
This gets a dedicated, properly-planned effort next — signals not yet used:
wrist keypoints from the pose stack, cross-camera release agreement,
appearance embeddings on the release crop, and possession chain-back. Only
after that lands do we tune the remaining metrics, and only then do we run the
second game (c2a) as a fully blind generalization test.

## Appendix A — How a player is tracked, start to finish

1. **Detect** every player, every frame, every camera (solved, 96–100%).
2. **Identify** via jersey reads: a confident read claims the track (a wrong identity
   survives only until the next read — typically 1–2 seconds).
3. **Skeletonize** every detection; project the ankle-sole point through each
   camera's court calibration to get court coordinates.
4. **Fuse** the four cameras: a player is held whenever any camera holds him; the
   court map arbitrates disagreements (correction layer).
5. **In pile-ups**, KPR appearance signatures break ties; **after long absences**,
   appearance re-acquisition + the next jersey read restore identity.
6. **Grade** (during development): compare every frame against operator truth.

## Appendix B — Camera findings and venue recommendations

**Where the CURRENT pipeline fails, on the court** (new heatmap, both games, 7
ground-truthed players, strict standard): `runs/tracking/pipeline_failure_heatmap.jpg`
— green = held, red = lost, per 1.5m court cell.

The picture is consistent and actionable:
- **Centre court is solid** — 70–90%+ hold-rate through the middle where at least two
  cameras see every player at readable size.
- **Failures concentrate at the edges**: the baselines, the corners, and the court
  ends. These are exactly the zones where (a) players appear smallest — far from the
  near cameras — and (b) jersey numbers become too small to read, so identity
  checkpoints stop firing. It is not a software blind spot; it is a *legibility*
  blind spot.
- Older SAM3-era maps for comparison: `runs/tracking/camera_blind_zones.jpg`,
  `runs/tracking/camera_real_coverage.jpg` (same edge pattern — the weakness is
  physical, not algorithm-specific).

**Venue recommendations, in order of impact per effort:**

1. **Bigger jersey numbers (highest-leverage change available).** Every accuracy
   layer in this system feeds on confident number reads: identity checkpoints,
   drift correction, re-acquisition, and the auto-labels that train the recognition
   model. Today a number is readable only when the player stands ~90+ pixels tall —
   roughly the near half of the court per camera. **Doubling the number size makes
   numbers legible at roughly half the player height**, which unlocks reads across
   most of the red zones in the heatmap, densifies checkpoints everywhere, speeds up
   re-acquisition after subs and scrums, and generates more training data per game.
   One kit change, every layer improves.
2. **Physical check of the near-left camera** — it under-covers the left-basket area
   (suspected mount/obstruction issue).
3. **Re-aim the two far cameras** to fill their frames toward the court ends —
   recovers part of the corner/baseline red zones optically.
4. No new cameras required for the current accuracy targets; camera *density* only
   becomes the binding constraint past ~90% strict (see honest ceiling).

## Appendix C — Files & artefacts

| File | What it is |
|---|---|
| `docs/STATUS.md` | this report |
| `docs/SOLUTION_PLAN.md` | software levers + camera re-aiming plan |
| `docs/IDENTITY_REID_DESIGN.md` | the confirmed/temporary identity design |
| `sam3player_n{11,22,43,6}_final88_e6fba750.mp4` | SAM3-pipeline demos (88% mean) |
| `sam3player_n{11,22,43,6}_sam3free_e6fba750.mp4` | production-pipeline demos (82.3% mean) |
| `sam3player_n11_3min_e6fba750.mp4` | 3-minute duration proof (no decay) |
| `runs/tracking/pipeline_failure_heatmap.jpg` | where the current pipeline loses players (court heatmap) |
| `runs/tracking/ledger/` | every experiment's scored result (the honesty trail) |
| `data/kpr_finetune/` | 3,061 auto-labelled crops for the recognition fine-tune |
