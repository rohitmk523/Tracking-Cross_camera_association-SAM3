# 2c490f1a — third blind game / dual-number ablation (2026-07-13)

**Why this game:** chosen by database survey of all remaining 4-angle
verified games as the one with the FEWEST shared jersey numbers — exactly
one (#8 on both teams) vs c2a's five. Gray vs Orange, 18 players, 153 GT
shots (218 plays). Purpose: decide whether c2a's weak WHO (37%) was the
dual-kit collision or the attribution logic.

**Protocol:** identical frozen pipeline (all thresholds from e6; baseline
attribution — the benched fusion untouched). GT used once, to score. Clock
constant +2.70s (e6 +2.30, c2a +2.35 — venue-stable). Compute: same 4-instance
AWS pattern + local chain.

## Headline (blind)

| metric | e6 (dev) | c2a (blind, 5 duals) | **2c4 (blind, 1 dual)** |
|---|---|---|---|
| Shot detected | 94% | 88% | **91%** (139/153) |
| **WHO** | 62% | 37% | **59%** |
| Zone/FT | 78% | 64% | **72%** |
| Make/miss | 98% | 95% | **89%*** |
| All correct | 56% | 24% | **50%** |
| Rebounds det/WHO | 81%/34% | 79%/25% | 67%/1-of-4 (GT n=6 only) |

Per type (n/det/who/zone/mm): 3PT 40/35/26/27/33 · 4PT 31/28/20/20/27 ·
FG 55/49/20/39/42 · FT 27/27/16/14/22.

## The answer

**WHO 59% blind with clean jerseys — within 3pts of the development game;
the attribution logic generalizes. c2a's 37% was the dual-number collision.**
Consequence: dual-kit identity is the single highest-leverage WHO lever
(docs/WHO_MASTER_PLAN.md, workstream W1).

## *The asterisk — a real venue fault found mid-run

**The near-right (NR) camera recording ends at 1559s** of this game's 3079s
(the other three run full length) — two independent AWS jobs and a direct
S3 probe confirmed it. The entire second half ran on THREE cameras; the
pipeline tolerated it (empty-NR placeholders). Make/miss dips to 89% mainly
from this: rim visibility per shot window fell to 0.506 (0.78-0.84 on the
other games), starving the make/miss features. Add NR reliability to the
venue findings next to the earlier near-left concern.

## Error anatomy

- 14 undetected shots: same families as before (soft FT arcs 0 this time —
  all 27 FTs detected; misses are blocked/short attempts + 3-cam tail).
- Zone 72%: FT-band mislabels (14/27 FT zone-correct) and the known
  long-range compression; NR loss removes one near-cam vote in the back half.
- FG WHO 20/49 — paint crowding, unchanged diagnosis.
- Rebounds: this operator annotated only 6 rebounds (vs 96 on c2a) —
  annotation-style variance; detection 4/6, too small to read WHO.
- 133 unmatched CV rim events — putback/tip noise, precision gate queued.

## Deliverables
- runs/tracking/ledger/{events_v2,possession_events}_2c490f1a.json (no Supabase writes)
- STATUS.md Part 9 addendum (the three-game ablation table)
- Next: docs/WHO_MASTER_PLAN.md — W1 dual-identity work starts immediately;
  game 13e1ffad stays sealed as the final blind validation.
