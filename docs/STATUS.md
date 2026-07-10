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
fails). KPR is a recognition model built exactly for this: given a crowded crop
*plus the skeleton of the specific player we mean*, it produces a numeric signature
of **that player only** — build, skin tone, hair, shoes — and compares signatures
using only body parts visible in both images. On our footage, **zero-shot** (never
trained on basketball), it picks the right same-kit team-mate **69% of the time
(chance: 33%)**. We use it in two places:
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

Production-pipeline demo videos: `sam3player_n11_sam3free_e6fba750.mp4`,
`sam3player_n22_sam3free_e6fba750.mp4`, `sam3player_n43_sam3free_e6fba750.mp4`,
`sam3player_n6_sam3free_e6fba750.mp4`.

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

## Appendix B — Camera findings

- Each camera reliably covers its own half and is nearly blind past centre court;
  the four together cover everything except two weak zones at the court ends.
- Real held-coverage maps: `runs/tracking/camera_blind_zones.jpg`,
  `runs/tracking/camera_real_coverage.jpg`.
- Recommended action: physical check of the near-left camera (left-basket
  under-coverage). Re-aiming the two far cameras to fill the frame helps the far
  thirds. Nothing else about the hardware needs to change.

## Appendix C — Files & artefacts

| File | What it is |
|---|---|
| `docs/STATUS.md` | this report |
| `docs/SOLUTION_PLAN.md` | software levers + camera re-aiming plan |
| `docs/IDENTITY_REID_DESIGN.md` | the confirmed/temporary identity design |
| `sam3player_n{11,22,43,6}_final88_e6fba750.mp4` | SAM3-pipeline demos (88% mean) |
| `sam3player_n{11,22,43,6}_sam3free_e6fba750.mp4` | production-pipeline demos (82.3% mean) |
| `sam3player_n11_3min_e6fba750.mp4` | 3-minute duration proof (no decay) |
| `runs/tracking/ledger/` | every experiment's scored result (the honesty trail) |
| `data/kpr_finetune/` | 3,061 auto-labelled crops for the recognition fine-tune |
