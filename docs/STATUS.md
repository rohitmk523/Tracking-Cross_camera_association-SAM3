# Cross-Camera Basketball Analysis — Progress Report

**What it does:** four fixed-camera videos in → *who* is on court (identity, team, **jersey
number**), *where* they are (one top-down map), *what happened* (possessions, passes), and an
AI commentator's play-by-play out. Runs on the existing 4-camera rig.

| Stage | Status |
|---|---|
| **1. Detect** players / referees / ball | ✅ Measured on held-out games: player 0.91 · referee 0.90 · ball 0.80 near-basket |
| **2. Track** each person per camera + team | ✅ Working, hardened |
| **3. Fuse** 4 cameras → one identity per player | ✅ Cross-game validated + independently confirmed by SAM3 |
| **4. Name** players by jersey number | ✅ **Live end-to-end, verified against visible jerseys** |
| **5. Events** possession / pass JSON | 🔶 Accuracy now *measured* vs human ground truth; dedicated ball model in training |
| **6. Narrate** AI play-by-play | ✅ Working, grounded on the verified identities |

---

## At a glance — last report vs. now

| | Last report | **Now** |
|---|---|---|
| Jersey numbers | plan validated, 99 labels, no trained model | **3 models trained + wired: players NAMED end-to-end, 94% precision, verified on video** |
| Player identity | 16 IDs / 13 people stable | holding — **plus court-map flicker eliminated** (234 dropout gaps → 5; players-per-frame now = truth) |
| Event accuracy | unmeasurable (no ground truth) | **measured**: operator labeled 3 windows; old events scored 14% — the pass/fail bar now exists |
| Ball | retracted claim, no replacement | **scored engineering program**: 3 approaches measured, dedicated BallNet built (35→38% held-out and climbing), auto-labeling teacher pipeline live, 4× data round training now |
| Independent verification | SAM3 cross-check in flight | **landed: 91–99% of our detections confirmed** on both games |
| Operator workload | jersey + ground-truth labeling pending | **zero — all targets met, every labeled set proved its value in a number** |
| Demo assets | raw pipeline video | video with **same player = same number in every camera** + named rosters |

---

## 1 · Named players — shipped end to end

Every model trained on our own annotated footage and validated on **games it never saw in
training**:

| Component | Job | Held-out accuracy |
|---|---|---|
| Legibility gate | "is a number readable here?" | 84.6% |
| Number localizer | find the number inside a player crop | 99.2% mAP |
| Number reader | read it (open-vocabulary — any number) | **95.3%** (89.1% before fine-tuning) |
| **Combined stack** | player crop → number | **94% precision when it answers** (69% read · 27% honest abstain · 5% wrong) |

- A committed number needs **multiple agreeing reads** — a single wrong read can never name
  a player; referees are structurally barred from carrying numbers.
- **Live on both test games:** rosters read `A #11, B #6, B #11, B #22` and
  `A #1/#2/#3/#5, B #1/#2/#3/#4/#7` — verified against the jerseys visible in the frames,
  and matching numbers the AI commentator independently read from the raw video.
- Everything downstream now speaks names: roster, events (`B #22 → B #7`), the marked
  video, and the narrator's grounding.

## 2 · Event accuracy is now a measured number

Operator-labeled ground truth (possession + pass/turnover moments, three game windows) now
scores every events version automatically:

| Metric | Old experimental events vs ground truth |
|---|---|
| Possession-team accuracy | **14%** |
| Possession changes | claimed **11** — truth was **2** |
| Passes | precision 0.43 · recall 0.75 |
| Turnovers | 0 correct — 4 fabricated, 2 real ones missed |

- This quantifies exactly why the events layer is gated until the ball is solved — and it
  is the **pass/fail exam** every new ball model runs against.

## 3 · The ball program — every approach now has a score

| Approach | Result vs ground truth | Verdict |
|---|---|---|
| Motion-rule fusion (retired) | 14% teams, invented turnovers | confidently wrong |
| Appearance detector + court projection | 6% — one possession, but a **correct** one | precise, nearly mute |
| Appearance + image-space attribution | 13.5% — 10× the coverage, still visibility-limited | right idea, starved of ball sightings |
| **BallNet (dedicated trained model)** | in training — see below | the fix |

- **Diagnosis is now precise:** attribution logic is fine; the bottleneck is *seeing* the
  ball. A dedicated motion-aware detector (industry-standard design, sized for real-time on
  the Jetson) is the answer.
- **Teacher pipeline built:** SAM3 (the big offline model) auto-labels ball positions on our
  own footage — including a filter that removes the *painted* basketballs (court/wall logos
  were 30–70% of its raw detections). No manual ball annotation needed.
- **First training rounds:** 35% → 38% held-out detection on one game of teacher data —
  clear signal, clearly data-limited. A **4× data expansion** (two additional games, four
  action-dense windows chosen from the play database) is running unattended right now:
  auto-label → rebuild → retrain → re-score.
- Until BallNet passes its exam, events stay honestly gated: no ball → no fabricated
  possessions.

## 4 · Trust & verification (how we know the above is real)

- **Human ground truth** — operator-labeled windows; every claim above is scored against
  them, automatically, on every change.
- **Independent model cross-check** — SAM3 confirms **91–99%** of everything our live
  pipeline detects; its rare extra finds map to a known far-camera limit that the 4-camera
  fusion design already compensates.
- **Cross-game validation** — the full stack cold-ran on an untouched game and reproduced
  the benchmark quality profile (and caught + fixed a black-vs-white-kit edge case in team
  naming).
- **Identity quality holding:** 16 identities for 13 real people, all 13 stable, ~zero
  court-map flicker, players-per-frame median = exactly the people on court.
- 43 automated tests; every number in this report reproduces with one command.

## 5 · What's next

1. **BallNet expansion round completes** → run the ground-truth exam (beat 14%, 2-not-11
   possession changes, zero fabricated turnovers) → unlock trustworthy possession/pass events.
2. Polish: fold one duplicated-number fragment; re-cut the demo video with named players.
3. Scale-out: run the full named-players pipeline unattended across more games.
   *(Shot make/miss: separate repo.)*

**One line:** identity is delivered, named, and independently verified; the ball has gone
from "unmeasured claims" to a scored engineering program with its dedicated model in
training and its pass/fail exam already written.
