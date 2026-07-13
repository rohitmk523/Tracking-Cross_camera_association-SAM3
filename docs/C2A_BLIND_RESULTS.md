# c2a — fully BLIND second-game results (2026-07-13)

**Protocol.** Every threshold frozen from e6 before this game was touched:
shot triggers, release logic, zone boundaries (710/840cm), FT band, P3
make/miss model + 0.310 threshold, possession rules. c2a's operator log (315
plays, validated row-for-row against the database) was used exactly once — to
score. The only per-game calibration: the DB↔video clock constant (+2.35s,
estimated from detection↔log time offsets; e6's was +2.30s) and the venue's
per-game camera sync offsets — infrastructure, not model tuning.

**Compute:** ~$10.5 AWS (ball+hoop cache + 3-way detection/pose/anchors prep),
~2h local (tracking, cross-camera identity, event chain).

## Headline vs e6 (what transferred, what didn't)

| metric (of GT shots) | e6 (tuned on) | **c2a (blind)** | transferred? |
|---|---|---|---|
| Shot detected | 94% (134/142) | **88% (166/189)** | YES |
| Make/miss (of detected) | 98% | **95% (157/166)** | YES |
| Zone / FT label | 78% | **64%** | partial |
| WHO (shooter) | 69% fused / 62% base | **37% base; fused 29%** | NO (see below) |
| All fields correct | 56% | **24%** | dragged by WHO |
| REBOUND detected | 81% (44/54) | **79% (76/96)** | YES |
| Rebounder correct | 34% | 25% (19/76) | weak both games |

Per shot type (c2a, n / detected / who / zone / make-miss):
3PT 44/42/19/24/40 · 4PT 51/43/17/23/42 · FG 74/64/19/49/61 · FT 20/17/6/10/14

## The blind test earned its keep — two real findings

1. **Dual-number kits are the #1 generalization hazard.** c2a has FIVE numbers
   worn by both teams (1,2,3,4,8 — 10 of 16 players); e6 had one. The first
   pass silently coin-flipped those identities (WHO 27%); restoring the
   kit-split + kit-aware naming recovered to 37%. The remaining WHO deficit
   vs e6 (62% base) is identity-track quality under kit ambiguity — jersey
   anchors resolve duals only when number AND kit shade are legible together,
   which halves usable anchors.
2. **The e6-adopted WHO fusion (wrists + cross-cam vote + KPR) did NOT
   transfer**: 29% vs the 37% baseline (3PT worst hit, 19→8). Its e6 gains —
   +7 points, split-half validated WITHIN e6 — were still e6-specific. The
   events output therefore uses baseline attribution; the fusion goes back to
   the bench until it is re-fit across BOTH games and re-validated on a third.
   (This is exactly the outcome a blind game is for.)

## What held up well

- **The shot trigger** (rim-arrival + rim-box entry): 88% recall on a game
  with a very different shot mix (73 long-range misses; e6 was paint-heavy).
  All 4 far-cam/hoop assumptions held at a different venue date/lighting.
- **Make/miss 95%**: the transplanted shot-detection brain on our detector,
  honest twice over (c2a was that model's held-out test game, and windows are
  arc-derived, no GT timestamps).
- **Rebound detection 79%** on 96 rebounds (e6 had 54).
- The clock constant, court zones, calibration — all venue-stable.

## Honest error anatomy (where the misses live)

- **23 undetected shots**: 3 FTs (low soft arcs), rest are blocked/short-rim
  attempts and two cache-edge plays. Same failure family as e6's tail.
- **Zone 64% vs 78%**: c2a shooters live at the 3PT/4PT boundary (declared
  radii 710/840cm; long-game distances cluster right at 840). The distance
  MEASUREMENT (projection at range) is the limiter, as diagnosed on e6 —
  per-camera radii refinement is the known next lever, untouched here to keep
  the test blind.
- **WHO 37%**: ~half the errors pick the same-number OTHER-team player or a
  crowd neighbor. Fixes queued: multi-game fusion refit; kit-anchor density
  (bigger jersey numbers request to the venue also directly helps this).
- **649 CV events vs 189 GT shots**: the long-range game produces far more
  rim interactions (tips, putback attempts, rattle-outs re-entering the box).
  381 events match no GT shot; the rebound layer consumes some; a stricter
  shot-ness gate (P3-probability + arc-quality) is queued for precision.
- TURNOVER/STEAL remain experimental (GT n=9; team-possession flips too
  noisy) — same verdict as e6.

## Deliverables

- `runs/tracking/ledger/events_v2_c2a354fe.json` + `possession_events_c2a354fe.json`
  (plays-style, source:"cv", NOT written to Supabase)
- `runs/event_demo/demo5min_c2a354fe.mp4` + `demo5min_e6fba750.mp4` — 5-minute
  continuous side-by-side (pipeline feed vs operator log over the live video)
- This file; STATUS.md Part 8 carries the e6 story.

## Queue after this (in leverage order)

1. Multi-game WHO fusion refit (fit e6+c2a jointly, validate on a third game).
2. Kit-anchor density for dual-number games (+ venue ask: bigger numbers).
3. Per-camera zone radii (the 3PT/4PT boundary at range).
4. Shot-ness precision gate for putback/tip noise.
5. FT low-arc trigger variant (3 misses here, 0 on e6).
