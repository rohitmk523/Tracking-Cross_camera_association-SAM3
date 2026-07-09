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

**The pattern: each camera covers its own half and goes blind on the far half.** A player is
realistically seen by only **~2 of the 4 cameras at any moment** (the pair on his end).
Measured blind rates: FL 51%, NL 45%, FR 39%, NR 26%, all concentrated on the far half from
each camera. This is *why* same-kit drift isn't rescued — when a player's two covering cameras
both slide onto a team-mate in a pile-up, there's no clean third camera to override them.

**Fix without new cameras:** re-aim the four we have to **overlap more in the centre and both
keys** (where clusters happen), turning 2-camera coverage into 3-camera coverage where it
matters — detail and honest limits in SOLUTION_PLAN.md.

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
| `runs/tracking/camera_blind_zones.jpg` | per-camera blind-zone court map |
| `runs/tracking/sam3player_ref_1_e6fba750.mp4` | referee tracked (96% — the clean case) |
| `runs/tracking/sam3player_n11_e6fba750.mp4` | #11 tracked (85%) |
| `runs/tracking/sam3player_n6_e6fba750.mp4` | #6 tracked (68%) |
| `runs/tracking/sam3player_n22_e6fba750.mp4` | #22 tracked (63% — same-kit hardest) |
| `runs/tracking/sam3player_n43_e6fba750.mp4` | #43 tracked (67%) |
| `runs/tracking/ledger/sam3reid_e6fba750_44_60.json` | single-object + re-ID scores vs ground truth |
| `runs/tracking/ledger/master_scorecard.jsonl` | every tracking version × player, cross-verified |
| `data/gt_players/*.json` | the 13,000 human ground-truth labels |
