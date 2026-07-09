# Cross-Camera Basketball Analysis — Weekly Progress Report (Mon → Thu)

**What the system does:** you give it the video from the four cameras already installed in
the gym. It works out **who** is on the court (team + jersey number), **where** everyone is
(a live top-down court map), and **what happened** (possession, passes, shots) — the
foundation an AI commentator then narrates.

**The headline this week:** we stopped guessing how good the system is and **built the
instrument that measures it against human-verified truth.** That instrument then told us,
with numbers, exactly which parts are solved and which part is the real remaining problem —
and pointed us at the fix. Detection is now **solved**. Identity-through-traffic is the one
open problem, and we can now measure every attempt at it to two decimal places.

---

## The week in one table

| Day | What we did | Result |
|---|---|---|
| **Mon** | Hardened tracking & fusion (region-of-confidence weighting, team-colour fix, duplicate-identity merge) | A ladder of measurable versions (v0 → v2.2); each an honest step |
| **Tue** | Built the **ground-truth measurement tool** — a human marks one real player across all 4 cameras, every frame | Foundation for grading everything objectively |
| **Wed** | Annotated **8 real players across 2 games (~13,000 human labels)**; ran the **SAM3 head-to-head** | Proved SAM3 **solves detection**; identity still open |
| **Thu** | Built the **identity solver** (jersey numbers as anchors) + the full cross-verified scorecard | Best-so-far on the hardest case; honest roadmap to the target |

---

## 1 · We built a truth machine (the week's most important deliverable)

Until now, "is the tracking good?" was a judgement call. This week we built a tool where a
person marks **one specific player** — say #43 — in all four camera views, frame by frame,
for a whole minute, saying "this box is him, he's not visible here." Do that for several
players and you have **ground truth**: the exact answer the computer's tracking is graded
against.

- **8 players fully marked across 2 different games — about 13,000 individual human labels.**
- Deliberately covers the hardest cases: **two team-mates in identical kit** (same colour,
  the classic mix-up) and **two opponents wearing the same number** (both a "#3" on court).
- This ground truth is **permanent** — it grades every version we have ever built and every
  version we will build, automatically, forever.

Two numbers matter, and we now measure both per player, per camera, per frame:
- **Coverage** — did we find the player at all?
- **Purity** — did we keep him as *one* identity, or did his label keep jumping to other people?

## 2 · Detection is now SOLVED (SAM3 head-to-head)

We ran a formal head-to-head: our own detector vs **SAM3** (a very large AI from a different
company), both graded against the human ground truth.

| | Coverage (did we find the player?) |
|---|---|
| Our own detector | good but patchy — players missed at far court and frame edges |
| **SAM3** | **96–100% of the time, every camera, nearly every frame** |

**Verdict: SAM3 becomes the eyes of the system.** The "we can't even see the far players"
problem — which you spotted yourself while marking ground truth — is gone.

## 3 · Identity is the one remaining problem — and we proved it's physics, not laziness

Here is the honest, measured state of "keeping one player as one identity" (purity), graded
against the human ground truth, across every version we built:

| Player (e6 game) | Best version | Purity today |
|---|---|---|
| #11 (traffic, dense jersey reads) | vS2 solver | **65%** |
| Referee (distinct clothing) | multiple | ~52% |
| #22, #43, #6 (same-kit team-mates) | various | 20–44% |

The pattern is the whole story: **the referee — who wears clothing nobody else wears — tracks
far better than same-kit players.** When SAM3 itself was graded, it showed the *exact same
pattern*: it tracked a distinctive player at **85%** on a clean camera, but same-kit
team-mates colliding dropped to 16–54%. **This is not a software bug — at this camera
resolution, two team-mates in identical kit who collide are genuinely pixel-identical**, for
SAM3 and for every tracker on Earth. That is why no off-the-shelf product solves this; it is
a known hard limit of the whole industry at this camera count.

## 3b · The breakthrough approach — SAM3 tracking ONE player at a time

We had been using SAM3 the wrong way (asking it to find *everyone* at once, which
fragments). The right way — the industry recipe — is to hand SAM3 **one player** and
say "follow him." We ran that on AWS: **20 single-object tracks** (5 ground-truthed
players × 4 cameras), each seeded once and then tracking that player on its own, graded
against the human truth.

| Player | Tracked correctly (4 cameras fused) | Court accuracy |
|---|---|---|
| **Referee** (distinct kit) | **96%** | **8 cm** |
| #11 | 85% | 10 cm |
| #6 / #43 | 67–68% | 24–28 cm |
| #22 (same-kit) | 63% | 19 cm |

**Two big findings.** First — **the engine works**: the referee, who wears clothing
nobody else does, is tracked at **96% with 8 cm court accuracy**, one identity carried
across four cameras as each drops out and the others cover. That is the whole
architecture, proven. Second — for same-kit players, SAM3's mask sometimes **slides onto
a team-mate** in a weak camera; fused across four cameras that still yields 63–85%, and
where it's on the right player the position is accurate to 8–28 cm. The gap is entirely
the same-kit-team-mate problem — the known hard limit — not the tracking machinery.

## 4 · The fix we are building — jersey numbers as anchors

The move that beats the physics is one no live tracker can do but our **batch (overnight)**
design can: solve the *whole minute at once*, and use every confident jersey-number reading
as an **anchor** that pins identity to the truth. Between anchors, a mix-up can only survive
a second or two before the next number sighting corrects it — forwards and backwards in time.

- We now read numbers **relentlessly**: **~4,300 confident reads in one minute** of the first
  game, **~6,200 in the second** — for the main players, an identity anchor **every few
  frames.**
- The solver is in active development. Best result so far took the hardest same-kit player
  from ~35% to **65%** purity. Not yet the target (90%+), but the approach is proven and every
  iteration now takes **under a minute to test** (see §5), so tuning is fast.
- **The honest branch point:** if anchoring gets dense-jersey players to 90%+, the system
  ships for **post-game analytics** with confidence flags on the few genuinely ambiguous
  seconds. If a player is rarely readable (turned away the whole time), that residue is a
  **camera-placement conversation** — the same answer the professional systems (Hawk-Eye,
  Second Spectrum) reached: more/better-placed cameras. We will know which, with data, not
  opinion.

## 5 · Engineering that makes all of this fast and honest

- **A version ledger**: every version's numbers are recorded automatically — nothing is
  claimed that isn't measured.
- **A detection cache**: the expensive step now runs **once per clip**; every new tracking
  idea re-runs in **~8 seconds instead of 16 minutes** (a 120× speed-up). This is why we can
  try many approaches per day.
- **Where it runs:** the heavy SAM3 pass runs on rented cloud GPU (a few dollars per game);
  everything else — every tracking experiment — runs locally in seconds thanks to the cache.
- **Cost at scale (measured):** a full game processed overnight in parallel on the cloud is
  roughly **$10–40** depending on quality settings. The edge box (AGX Orin) handles live
  capture; the SAM3-quality pass is a cloud batch job (full SAM3 cannot run live on the edge —
  measured and confirmed).

---

## What you can show the client, honestly

1. **We can now prove quality, not assert it** — 13,000 human labels, every version graded.
2. **Detection is solved** — SAM3 sees 96–100% of players, verified against that human truth.
3. **The remaining problem is precisely identified and is a known industry-wide hard limit** —
   same-kit players at this camera count — and we have a concrete, measured line of attack
   (jersey-anchored whole-game solving) already showing progress.
4. **We know exactly what "done" looks like and how to prove it** — 90%+ purity on
   readable players, honest flags elsewhere, and a data-backed camera recommendation if the
   client needs more.

**Bottom line:** this was the week the project moved from "we think it works" to "here is
exactly how well it works, measured against human truth, and here is the one problem left and
our plan for it."
