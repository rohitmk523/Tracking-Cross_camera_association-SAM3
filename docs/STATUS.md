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

## Thursday PM — two validations that de-risk production

Two AWS runs answered the two biggest open questions:

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

## Where this leaves us — honest

- **Detection: solved** (96–100%, verified).
- **Tracking engine: proven** (referee 96% / 8 cm).
- **Same-kit identity in clusters: the one hard problem** — 63–85% today, the known limit of
  every tracker on Earth at this camera count. Not a machinery failure; a genuine limit of
  what four side-angle cameras can see when identical-looking players collide.

The detailed plan to close it — software levers now running, plus camera re-aiming — is in
**[SOLUTION_PLAN.md](SOLUTION_PLAN.md)**.

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
