# Cross-Camera Basketball Analysis — Progress

_Last updated: 2026-07-06_

**What it does:** four fixed cameras in → *who* is on court (identity, team, **jersey number**),
*where* they are (one top-down map), *what happened* (possessions, passes), and an AI
commentator's play-by-play out. Runs on the existing rig.

| Stage | Status |
|---|---|
| **1. Detect** (players / referees / ball) | ✅ Measured: player 0.91 · referee 0.90 · ball 0.80 near-basket (held-out games) |
| **2. Track** (per camera + team) | ✅ Working, hardened |
| **3. Fuse** (4 cameras → one identity per player) | ✅ Validated on a second game + SAM3 cross-checked |
| **4. Jersey numbers → named players** | ✅ **NEW: live end-to-end, verified against visible jerseys** |
| **5. Events** (possession / pass JSON) | 🔶 Engine ready; **accuracy now MEASURED against operator ground truth** — trained ball detector is the one remaining build |
| **6. Narrate** (AI play-by-play) | ✅ Working, grounded — now with real jersey numbers to name |

---

## 1 · NEW — Named players, end to end

The full trained jersey stack shipped this week, every model validated on **games it never
trained on** (leave-games-out):

- **Legibility gate** (is a number readable at all?) — 84.6%
- **Number localizer** (find the number inside a player crop) — 99.2% mAP, trained in 9 min
- **Number reader** (open-vocabulary scene-text model) — 95.3%, up from 89.1% zero-shot —
  the operator's annotations demonstrably lifted cross-game accuracy
- **Combined stack on unseen-game crops: 69% read correctly · 27% honest abstain · 5% wrong
  → 94% precision when it commits an answer.** Every stage abstains rather than guesses,
  and a per-player vote (≥2 agreeing reads, ≥60% majority) means one wrong read can never
  name a player. Referees are structurally barred from carrying numbers.
- **Live on both games** — e6: `A #11, B #6, B #11, B #22` (independently matching what the
  AI commentator read with its own eyes last week); second game: **10 of 18 identities
  named**, verified against the visible jerseys in the frames.
- Everything downstream speaks names now: roster, events (`B #22 → B #7`), the marked video,
  and the narrator's grounding.

## 2 · NEW — Event accuracy is now a measured number

The operator ground-truth round completed (3 windows labeled with the new playlist tool —
two cameras side by side so possession is never out of view). First scoring of the old
experimental events against human truth:

| Metric | Experimental events vs operator GT |
|---|---|
| Possession-team accuracy | **14%** (≈81% after correcting a since-fixed team-naming inversion) |
| Possession changes | system claimed **11**, truth was **2** — over-firing confirmed |
| Passes | precision 0.43 / recall 0.75 (timing good, phantoms invented) |
| Turnovers | **0 detected correctly** — all 4 were fabricated, both real ones missed |

This is the audit's ball retraction, now quantified — and it is the **exam** the trained
ball detector must pass. Every future events version gets this score automatically.

## 3 · Identity layer (recap of last week's fixes — unchanged, still holding)

- Sync verified per run (one camera had been 13 frames off); fisheye lenses corrected
  (position error ~35cm → ~11–23cm); fragments stitched; blink-free court map (dropout
  gaps 234 → 5, players/frame median = exactly the people on court).
- Benchmark window: **16 identities for 13 people, all 13 stable, 1 phantom**; cold-run on a
  second game reproduced the profile.
- **SAM3 cross-check** (independent model, cloud GPU): confirms **91–99%** of everything we
  detect; its extra finds map to the known far-camera recall gap, which fusion's zone
  design already compensates — and its disagreement frames are banked as free hard
  examples for the next detector round.

## 4 · Ball & events — the one remaining build

- **Interim (running now):** the retrained detector's near-basket ball (0.80) projected
  through the corrected lenses → possession events → scored against the operator's GT.
  Half-court possessions may already be usable; the score will say.
- **The real fix (next):** a **trained motion-aware ball detector** (the audit-endorsed,
  literature-standard design — learned model on top of the motion signal, not hand-tuned
  rules). Teacher data plan: SAM3 with a "basketball" concept prompt generates pseudo-labels
  on the GPU; the event-pool ball annotations anchor it; the operator's GT scores it.
- Until it passes, events stay honestly gated: no ball → no fabricated possessions.

## 5 · Where we are

- **Working, verified, and named:** detection → tracking → fusion → jersey identity →
  grounded narration. 43 automated tests pass. **Zero operator dependencies remain** — all
  annotation targets were met and every set proved its value in a measured number.
- **In flight:** interim ball baseline scoring (today); SAM3 ball-teacher probe (next).
- **Next, in order:** ① trained ball detector → beat the 14% baseline → trustworthy
  possession/pass events ② polish: fold the one duplicated-number fragment, re-cut the
  client demo with named players ③ scale: run the full pipeline across more games.
  *(Shot make/miss: separate repo.)*

**One line:** the identity promise is delivered and independently verified — the system now
says "**B #22 passes to B #7**" from raw four-camera video; the ball tracker is the last
build, and for the first time its success will be a single measured number.
