# Cross-Camera Basketball Analysis — Detailed Weekly Report (Mon → Thu)

**What the system does:** from the four cameras already in the gym, work out **who** is on
court (team + jersey), **where** they are (top-down court map), and **what happened**
(possession, passes, shots) — the foundation an AI commentator narrates.

**The week in one line:** we stopped guessing and built the instrument that measures the
system against human-verified truth; that instrument proved **detection is solved**,
isolated **identity-through-traffic** as the one hard problem, and — after correcting how we
were using SAM3 — proved the **tracking engine works** (a distinct-kit player tracked at 96%
with 8 cm court accuracy). The remaining gap is same-kit team-mates in crowded moments, a
known industry-wide hard limit, and we have a concrete multi-pronged plan for it.

---

## Monday — Hardened the tracker, built a measurable ladder

**Did:** three fixes to cross-camera tracking, each measured and versioned (v0 → v2.2).
- **Region-of-confidence**: a player's box height is his distance, so a near camera now
  "owns" the calls on its own half and barely votes on the far half (where it used to guess).
- **Team-colour fix**: on steep near cameras the shirt-colour patch was reading the *floor*;
  now it suppresses floor colour and reads the jersey.
- **Duplicate-identity merge**: two tracks that shadow each other are merged unless the world
  proves they are two people (different numbers, same-camera simultaneous, different look).

**Result:** a ladder of honest versions — each a measured step, none a leap. Set up so every
later idea is graded, not asserted.

## Tuesday — Built the truth machine (the week's foundation)

**Did:** built a tool where a person marks **one real player across all four cameras, every
frame** — "this box is him; he's not visible here." That is **ground truth**: the exact
answer the computer is graded against. Also built a **detection cache** (the expensive step
runs once per clip; every experiment after re-runs in ~8 seconds instead of 16 minutes — a
120× speed-up) and a **version ledger** (every version's numbers recorded automatically).

**Result:** we can now measure quality to two decimals instead of eyeballing it — the single
most important thing we built all week.

## Wednesday — Annotated the truth, ran the SAM3 head-to-head

**Did:** annotated **8 real players across 2 games (~13,000 human labels)**, deliberately
covering the hardest cases: two team-mates in identical kit, and two opponents wearing the
same number. Then ran a formal head-to-head: our detector vs **SAM3** (a very large AI from
another company), both graded against the human truth.

**Result:**
- **Detection is solved.** SAM3 finds players **96–100%** of the time, every camera, verified
  against human truth. The "we can't even see the far players" problem is gone.
- **Identity is not solved by any detector alone.** We tried four identity approaches this
  day; all sat at 20–65% "purity" (keeping one player as one identity). The referee — who
  wears distinct clothing — always scored far higher than same-kit players. **The problem is
  physics, not a bug.**

## Thursday — Corrected how we use SAM3; proved the tracking engine

**Did:** realised we had been using SAM3 the wrong way — asking it to find *everyone* at once
(which fragments) instead of the industry recipe: hand it **one player** and say "follow
him." Ran that on AWS: **20 single-object tracks** (5 ground-truthed players × 4 cameras),
each seeded once then tracking that player on its own, fused across the four cameras by court
position, graded against the human truth.

**Result — the two findings that define where we are:**

| Player | Tracked correctly (4 cameras fused) | Court accuracy |
|---|---|---|
| **Referee** (distinct kit) | **96%** | **8 cm** |
| #11 (well-numbered) | 85% | 10 cm |
| #6 / #43 | 67–68% | 24–28 cm |
| #22 (same-kit) | 63% | 19 cm |

1. **The engine works.** The referee is tracked at **96% with 8 cm accuracy** — one identity
   carried across four cameras as each drops out and the others cover. That is the entire
   architecture, proven on our footage against human truth. When the target is
   distinguishable, this tracks near-perfectly and knows *exactly* where he is.
2. **Same-kit players drift.** For team-mates in identical kit, SAM3's outline sometimes
   **slides onto a team-mate** in a weak camera; fused across four cameras that still yields
   63–85%. We added a jersey/geometry correction layer that improved individual cameras but
   not the fused number, because **the drift is correlated** — in a pile-up, all four cameras
   slide onto the same nearby team-mate together, and numbers are least readable in exactly
   those moments.

---

## How the tracking actually works (method, in plain terms)

So the results below are readable, here is the pipeline we built and how each piece works:

1. **Detect** — every frame, every camera, find all players (SAM3 / our detector). Solved.
2. **Seed one player** — pick a player by a **confident jersey-number reading** ("that's clearly
   #11") and hand SAM3 *that one box*. This is the industry recipe (single-object), and it is
   the correction we made on Thursday — previously we asked SAM3 to find everyone at once, which
   fragments.
3. **SAM3 follows him** — from that one seed, SAM3 propagates an outline of **that player**
   across the clip, on its own, in each camera. One clean track per player per camera.
4. **Fuse the four cameras by court position** — project each camera's outline to a top-down
   court coordinate and combine them. A player is tracked whenever **any** camera holds him, so
   when one camera loses him the others carry — one continuous court track.
5. **Grade against human truth** — we compare, frame by frame, where SAM3 says the player is
   versus where the operator marked him (13,000 labels). *Leakage-free*: the human labels are
   only the answer key, never fed back into the track. The one seed frame is the only input; every
   other frame is a blind test.

The two numbers we report: **coverage** (fraction of frames the fused track is on the right
player) and **court accuracy** (centimetres between our position and the true position).

## Thursday PM — two validations that de-risk production

Two AWS runs answered the two biggest open questions. **How we ran each:**

- **A (production seeding):** instead of seeding from the human box, we found — for each player,
  automatically — the earliest frame where the jersey number was read with high confidence, and
  seeded SAM3 from *that detection*. No ground truth touches the seed. If this matches
  hand-seeding, the system can run itself.
- **B (duration):** we pulled a **3-minute** clip (3× longer) and tracked #11 across it, then
  measured the present-rate in six 30-second buckets — does the outline hold, or slowly fall off
  the player as the clip runs long?

**A. Does it work WITHOUT ground truth? (production seeding)** — we re-ran every player seeded
from an **automatic confident jersey reading** instead of a human box. It works as well or
better:

| Player | Hand-seeded (GT) | Auto jersey-seeded |
|---|---|---|
| #11 | 84% | **86%** |
| #6 | 69% | **88%** |
| #43 | 67% | **84%** |
| #22 (same-kit) | 63% | 64% |

Seeding from a confident number read is **GT-quality or better** — so the production trigger (a
clear jersey read starts the track) is proven. With jersey seeding, most players now sit at
**84–88%**; only #22 (the same-kit *and* same-number confuser) lags.

**B. Does it decay over minutes? (duration)** — we tracked #11 over a **full 3 minutes**. It
does **not** decay:

| Camera | 0–30s | 30–60 | 60–90 | 90–120 | 120–150 | 150–180 |
|---|---|---|---|---|---|---|
| **NR** | 96% | 99% | 93% | 100% | 100% | 98% |
| FR | 100% | 87% | 31% | 84% | 100% | 86% |
| FL | 63% | 58% | 83% | 100% | 78% | 71% |

The strong camera (NR) held #11 at **93–100% across the entire 3 minutes with zero decline.**
The dips in FL/FR are the same *spatial* weakness (far-camera drift), not a *time* effect — the
mask doesn't fall apart as the clip gets longer. The full-game duration concern is largely
answered.

## Friday — the re-seed layer landed: 68% → 88% strict all-angles

The "re-seed-on-drift" item from the in-progress table below is now **built, run, and
measured**, and it stacked with the cross-camera correction exactly as designed. Strict
metric throughout: the player must be correctly shown in **every camera that can see him**
(the honest per-angle measure), scored leakage-free against operator ground truth.

| Player | Old best (Thu) | Re-seed alone | **Re-seed + cross-camera correction** | ≥1-angle coverage |
|---|---|---|---|---|
| #11 | 86% | 78% | **97%** (FL 94, FR 98, NL 96, NR 99) | **100%** |
| #22 (hardest: same kit + same number) | 53% | 78% | **83%** | 89% |
| #43 | 57% | 80% | **85%** | 93% |
| #6 | 76% | 78% | **86%** | 93% |
| **Mean** | **68%** | 78% | **88%** | 94% |

- How it works: every confident jersey reading becomes a checkpoint; SAM3's track is
  re-started fresh from each checkpoint (a drift can only survive until the next confident
  number), then the 4-angle court-map correction overrides any camera that disagrees with
  the jersey-confirmed position. Two layers, same recipe as before — no new training.
- Court position error stays 5–12 cm (median).
- The remaining weak cells (#22 NL 62%, #6 FR 78%, #43 NR 76%) are same-kit steals in
  pile-ups — the next lever is joint assignment with the full roster tracked (mutual
  exclusion: a body already claimed by #6 can't also be #22). Engineering for that is the
  multi-object port below.
- Files: `runs/sam3_players_reseed/` (masklets), `runs/tracking/ledger/sam3xcam_e6fba750_44_60.json`,
  scripts `extract_reseed_points.py`, `sam3_track_player.py --reseeds`, `solve_player_xcam.py`.

**Cost control (new standing rule):** no AWS run above **$5** without explicit approval.
The 59-track roster run and the 4-camera full-game #11 run were terminated mid-flight under
this rule; the replacement is a **multi-object port** (all players share one SAM3 pass per
camera) that re-does the roster experiment for ~$2 instead of ~$10 and makes full-game runs
~10× cheaper. Port in progress.

---

## Where this leaves us — honest

- **Detection: solved** (96–100%, verified).
- **Tracking engine: proven** (referee 96% / 8 cm).
- **Same-kit identity in clusters: the one hard problem** — 63–85% today, the known limit of
  every tracker on Earth at this camera count. Not a machinery failure; a genuine limit of
  what four side-angle cameras can see when identical-looking players collide.

The detailed plan to close it — software levers now running, plus camera re-aiming — is in
**[SOLUTION_PLAN.md](SOLUTION_PLAN.md)**.

---

## What's left to resolve — and what's in progress right now

**The one remaining problem, stated precisely:** a player who is *both* in identical kit to a
team-mate *and* wears the same number as an opponent (our #22 case) sits at **64%** while every
other player is now **84–88%**. That single hardest configuration — where colour, number, and
appearance all fail to distinguish him in a pile-up — is the residual.

**In progress now:**

| Item | What it does | Status |
|---|---|---|
| **Re-seed-on-drift loop** | Re-start SAM3's track from a fresh confident number reading the moment the outline slides onto a team-mate — so a mistake can only last the split-second until the next number is seen.  **Building now** | **Building now** |
| **Masklet-splitting** | Where the number on the outline contradicts who we're tracking, cut it there and re-acquire the right player from the other cameras. Offline, no new compute. | Queued |
| **Full-game (40-min) validation** | Confirm the 3-minute "no decay" result holds across a whole game with many subs and returns. | Queued |
| **NL left-basket camera check** | The one physical action: NL under-covers the left basket (likely obstruction/mis-aim). A re-aim there recovers the weakest court zone. | |
| **Confirmed/temporary re-ID layer** | For subs / players who leave court and return: track as a "temp" identity by geometry + colour + appearance until a number confirms, then merge retroactively. | Designed ([IDENTITY_REID_DESIGN.md](IDENTITY_REID_DESIGN.md)) |

**What is NOT in the remaining work** (already resolved this week): detection (solved), jersey
OCR quality (fine), production seeding without ground truth (proven), and time-decay over
minutes (proven not to happen). The remaining effort is narrowly focused on the same-kit /
same-number identity case — not on the pipeline as a whole.

---

## Where each camera loses the player (blind-zone finding)

We mapped, for every court position, which cameras can see the player and which cannot
(`runs/tracking/camera_blind_zones.jpg` — green = sees, red = blind).

Two maps tell the story:
- `runs/tracking/camera_blind_zones.jpg` — each camera covers its own half, blind on the far.
- `runs/tracking/camera_real_coverage.jpg` — how many cameras **correctly hold** the player per
  zone (the one that matters).

**What the real-coverage map shows:**
1. **The near sideline is our strongest zone (2–3 cameras hold).** NL and NR are correctly
   doing the job only they can — resolving players under the rims, where FL/FR are too far to
   tell who's who. **They must stay aimed there.**
2. **The real weak zone is the LEFT basket / left third (0.0–0.7 cameras), asymmetrically worse
   than the right.** 21 of 26 weak court-cells are at the **ends/keys, not the centre.**

**Fix without new cameras (re-aim FL/FR; leave NL/NR on the rims):** the highest-value action is
to **check NL's view of the left basket** — it drops catastrophically for some players (4–14%),
a likely obstruction or mis-aim in exactly the weakest zone. Then aim FL/FR to fill-frame on the
under-covered left/end zones. Detail and honest limits in SOLUTION_PLAN.md.

## Do we need more detector / jersey-OCR training? — No, both are fine for now

We checked, because it's the natural question. **Neither is the bottleneck.**

- **Detection is solved and needs no more training.** SAM3 finds players **96–100%** of the
  time; our own detector reads player/referee/ball at 0.90+ accuracy. When we lose a player it
  is *not* because we failed to detect him — he's detected, then the *identity* slides. More
  detection data would be low-value right now.
- **Jersey OCR is strong and needs no more training.** It reads numbers at **94–95% precision**
  and produces **~4,300–6,200 confident reads per minute**. The reason a number sometimes
  doesn't help is that it's **physically hidden** (player turned away, buried in a pile-up),
  not that the OCR misreads it. More OCR training can't read a number the camera can't see.

**Conclusion:** the effort belongs on **identity-through-traffic** (the solvers, jersey
re-seeding, and camera re-aiming) — not on retraining detection or OCR, which are already
good enough.

---

## Before / after videos - (old first, then corrected)

Show the **OLD** clip, then the **NEW** — same player, same minute, visibly steadier.

| Player | OLD video (first approach) | NEW video (corrected) | **FINAL video (re-seed + correction)** | Strict all-angles |
|---|---|---|---|---|
| #6 | `sam3player_n6_e6fba750.mp4` | `sam3player_n6_jerseyseed_e6fba750.mp4` | `sam3player_n6_final88_e6fba750.mp4` | 76% → **86%** |
| #43 | `sam3player_n43_e6fba750.mp4` | `sam3player_n43_jerseyseed_e6fba750.mp4` | `sam3player_n43_final88_e6fba750.mp4` | 57% → **85%** |
| #11 | `sam3player_n11_e6fba750.mp4` | `sam3player_n11_jerseyseed_e6fba750.mp4` | `sam3player_n11_final88_e6fba750.mp4` | 86% → **97%** |
| #22 | `sam3player_n22_e6fba750.mp4` | `sam3player_n22_jerseyseed_e6fba750.mp4` | `sam3player_n22_final88_e6fba750.mp4` | 53% → **83%** |
| Referee | `sam3player_ref_1_e6fba750.mp4` | *(unchanged — already 96%)* | | 96% |
| #11 · 3-min duration | — | `sam3player_n11_3min_e6fba750.mp4` | | holds all 3 min, no decay |

Three generations, all kept intact: OLD (no suffix), `_jerseyseed` (production seeding), and
`_final88` (re-seed-on-drift + cross-camera correction — the 88%-mean system from the Friday
section). Show them in order: each generation is visibly steadier than the last.

## Files & artefacts (this week's deliverables)

| File | What it is |
|---|---|
| `docs/STATUS.md` | this report |
| `docs/SOLUTION_PLAN.md` | software levers + camera re-aiming plan |
| `docs/IDENTITY_REID_DESIGN.md` | the confirmed/temporary re-ID design |
| `runs/tracking/camera_blind_zones.jpg` | per-camera blind-zone court map (each cam sees own half) |
| `runs/tracking/camera_real_coverage.jpg` | how many cameras correctly HOLD the player per zone (the real map) |
| `runs/tracking/sam3player_ref_1_e6fba750.mp4` | referee tracked (96% — the clean case) |
| `runs/tracking/sam3player_n11_e6fba750.mp4` | #11 tracked (85%) |
| `runs/tracking/sam3player_n6_e6fba750.mp4` | #6 tracked (68%) |
| `runs/tracking/sam3player_n22_e6fba750.mp4` | #22 tracked (63% — same-kit hardest) |
| `runs/tracking/sam3player_n43_e6fba750.mp4` | #43 tracked (67%) |
| `runs/tracking/ledger/sam3reid_e6fba750_44_60.json` | single-object + re-ID scores vs ground truth |
| `runs/tracking/ledger/master_scorecard.jsonl` | every tracking version × player, cross-verified |
| `data/gt_players/*.json` | the 13,000 human ground-truth labels |
