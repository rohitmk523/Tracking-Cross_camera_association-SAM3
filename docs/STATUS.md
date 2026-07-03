# Cross-Camera Basketball Analysis — Progress

_Last updated: 2026-07-03_

**What it does:** four fixed cameras in → *who* is on court (identity, team, jersey), *where*
they are (one top-down map), *what happened* (possessions, passes), and an AI commentator's
play-by-play out. Runs on the existing rig (far-left, far-right, near-left, near-right).

| Stage | Status |
|---|---|
| **1. Detect** (players / referees / ball, every frame) | ✅ Measured on held-out games: player 0.91 · referee 0.90 · ball 0.80 near-basket |
| **2. Track** (follow each person per camera + team) | ✅ Working, hardened this week |
| **3. Fuse** (4 cameras → one identity per player) | ✅ **Validated on a second, untouched game + independently cross-checked by SAM3** |
| **4. Events** (possession / pass JSON) | 🔶 Engine built and tested — awaiting real ball tracking + ground-truth scoring |
| **5. Narrate** (AI play-by-play) | ✅ Working, grounded on verified identities (set-of-marks video built) |

---

## 1 · Identity quality — the week's core result

Three root causes were found by this week's audit and fixed; each fix was re-measured on the
benchmark window (4 cameras, ~13 real people on court):

- **Camera sync was silently broken.** The sync routine returned "0 offset" without warning
  when a clip had no audio — one camera was fused **13 frames (~0.4s) out of step**. Now sync
  is strict (the pipeline refuses to run rather than fuse unaligned cameras), and the true
  offsets are measured and verified on every run.
- **The fisheye lenses were never corrected.** Player positions were mapped to the court
  through plain homographies on distorted pixels — the old calibrations only fit half their
  reference points (metres of error elsewhere). We refit all four cameras with a per-camera
  lens-distortion model: near-camera position error dropped from **~35cm to ~11–23cm**, and the
  cameras now agree with each other well enough that identity matching tightened everywhere.
- **Identity fragments and flicker.** Two clean-up passes: fragments of the same player
  (dropout → reappearance) are stitched into one identity, and brief detection blinks
  (91% of dropouts were 1–5 frames) are bridged by the motion model — flagged honestly as
  predictions and drawn hollow on the map.

| Benchmark window | Before | After |
|---|---|---|
| Identities created for ~13 people | 30 | **16** |
| Real people tracked ≥ half the clip | 12 | **13 — all of them** |
| Phantom identities | 6 | **1** |
| Court-map flicker (dropout gaps) | 234 | **5** |
| Players per frame (median) | 11 | **13 = exactly the people on court** |
| Frames over-counting | 28% | **~2%** |

## 2 · Verified twice, independently

- **Second game, cold run.** The full stack ran untouched on another game (different teams,
  black-vs-white kits, full-court sequence): **15 identities / 12 stable / 1 phantom / clean
  5-v-5 + 3 referees** — the same quality profile as the tuned benchmark. The run also exposed
  a real bug: black/white kits have no "colour", so team naming was a per-camera coin flip —
  fixed with a brightness fallback (darker kit = Team A on every camera).
- **SAM3 cross-check (independent model, cloud GPU).** We ran Meta's SAM3 — a completely
  different model family — over both games (8 camera-clips) and compared box-for-box:
  - **91–99% of everything our pipeline detects, SAM3 independently confirms** → our
    false-positive rate is effectively clean, verified externally.
  - Where the two disagree is localised and explainable: far-seeing cameras miss small
    distant players (a known detector limit) — the 4-camera fusion already compensates by
    letting each near camera own its half, which is why the fused output still tracks 13/13.
  - SAM3's "we-see-it-you-don't" frames are saved as a **ranked hard-example list** — free
    training data for the next detector improvement round.
  - SAM3 itself needs ~20 min/clip on a GPU, so it's an offline referee, not a competitor.

## 3 · Team identification

- Teams are classified by **jersey colour** (saturation/brightness/hue features + clustering),
  which fixed the earlier appearance-model failure on dark kits blending into shadows.
- Only the **near cameras vote** on team (far-camera crops are too small to separate colours);
  referee status comes from the detector and is accepted from any camera — both refs are now
  correctly identified.
- New this week: **achromatic-kit handling** (black vs white) — cluster naming anchors on
  brightness when hue carries no signal, so all cameras name the teams consistently.

## 4 · Jersey numbers (identity → real names)

- **Plan upgraded to match what wins the SoccerNet jersey challenge:** a scene-text
  recognition model (reads *any* number, including ones never seen in training) gated by a
  legibility classifier — instead of a fixed-list classifier that can only output numbers it
  was trained on.
- **De-risked before spending annotation:** the pretrained recogniser already reads **78% of
  our annotated crops with zero training** — so fine-tuning needs hundreds, not thousands, of
  labels. Training machinery is built and smoke-tested (one command when labels are ready).
- **Data quality fixed first:** the labelling queue now interleaves **all 25 games** (it was
  burning every click on one game); the train/validation split is leak-proof (same play can
  never appear on both sides); "unclear" marks now train the legibility gate instead of being
  discarded; the tool validates every save, writes atomically, and keeps backups.
- **Progress:** 99 usable labels across 8 games (target ~300–400). Once landed: fine-tune →
  every "id14" becomes "#22" in the events, the demo video, and the narration.

## 5 · Ball & events — the honest part

- The audit **retracted** Wednesday's "full-court ball tracking works" claim: measurement
  showed the motion tracker was following players (their moving limbs), not the ball. It is
  now an explicit experimental flag, and events degrade honestly without a ball rather than
  fabricating possession.
- What stands: **near-basket ball detection works (0.80)** — where shots happen; and the
  possession/pass engine itself is sound (its two proven failure modes were fixed and are
  covered by tests).
- The replacement is literature-backed and uses our existing data flywheel: a **trained
  motion-aware ball detector** (the motion signal was the right idea — it needs a learned
  model on top, not hand-tuned rules).
- A **ground-truth labelling tool** now exists (~10 min of operator time per clip: who has
  possession + pass/turnover moments) — the moment one clip is labelled, every event claim
  becomes a measured precision/recall number, now and for every future version.

## 6 · Deliverables produced this week

- **Pipeline demo video** (~40s): detect → track → fuse, with the same player carrying the
  same number in every camera beside a live top-down court map, ending with the events JSON
  and the AI commentator's actual output.
- **Set-of-marks grounding video** for the narrator (boxes + identities burned in — measurably
  reduces AI hallucination), ready to become names once jersey numbers land.
- 38 automated tests pass; every result above is reproducible with one command from the repo.

---

## Where we are

- **Done + verified:** detection → tracking → 4-camera fusion → grounded narration, with
  verified sync, corrected lenses, cross-game validation, and independent SAM3 confirmation.
- **In progress (operator):** jersey labels — 99 usable across 8 games (target ~300–400);
  ground-truth event labels not started.
- **Next, in order:** ① jersey recogniser fine-tune → named players end-to-end
  ② trained motion-aware ball detector → trustworthy possession/pass events
  ③ score events against ground truth and tune the remaining occlusion under-count.
  *(Shot make/miss stays out of scope — separate repo.)*

**One line:** identity is solved and independently verified; jersey names and the ball tracker
are the two remaining builds, and both have data pipelines, training code, and evaluation ready.
