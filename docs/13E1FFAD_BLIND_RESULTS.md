# 13e1ffad — fourth blind game / oldest-footage stress test (2026-07-14)

**Why this game:** the final sealed game (reserved since the three-game
scoreboard was built). A Day in Miami (Yellow, 7) vs Premier Mtg Elite
(Black, 8), 2026-01-31 — the OLDEST footage in the working set, ~7 weeks
before the calibration-era games. 206 GT plays / 122 shots.

**Protocol:** identical frozen pipeline (thresholds from e6; frozen P3 at
thr 0.310; --no-team-prior). GT touched once, to score. Clock constant
fitted post-detection: **+1.40s** (118/122 shots matched, p25..p75
+1.10..+2.07 — venue drifts from the +2.30..+2.70 of the March/April games).
Compute: 3× g5.2xlarge prep + 2 ballcache instances (~$10), local chain.

## Headline (blind)

| metric (of GT shots)  | e6 (dev) | c2a | 2c4 | **13e1 (blind #3)** |
|---|---|---|---|---|
| Shot detected | 94% | 88% | 91% | **89% (109/122)** |
| WHO (of detected) | 66% | 38% | 62% | **34%** (detshots rq-scorer: 39%) |
| Zone / FT | 79% | 65% | 72% | **45%** |
| Make/miss | 98% | 95% | 89%* | **61%** |
| All correct | 60% | 27% | 52% | **15%** |
| Rebounds det / WHO | 93% / 19-50 | 81% / 24-78 | 67% (n=6) | **78% (25/32) / 6-25** |

## The two structural findings (both quantified, neither a logic regression)

1. **January footage vs March-era calibration.** The anchor-geometry sweep
   puts NR's best cross-camera agreement at **538 cm** median (vs 47-67 cm
   on e6) and NL at 145 cm — the cameras were evidently repositioned before
   the calibration was built. Everything court-plane degrades: zones (45%),
   cross-camera identity reacquisition (WHO), shooter positions. Per-camera
   image-space stages (detection, arcs, claiming) are unaffected — which is
   exactly the split the numbers show.
2. **Rim visibility 0.517 vs 0.844 norm** in shot windows → the frozen P3's
   rim-relative features starve → make/miss 61% (precision 0.83, recall
   0.39: it calls MISS when it can't see the rim interaction). The 2c4
   NR-outage already showed this failure direction at 89%; here BOTH
   near-rim views are compromised. AUC is still 0.80 — the model ranks
   correctly; the frozen threshold sits wrong for this rim-starved regime.

## Also stacked against WHO in this game (roster + GT quirks)

- #10 worn by THREE players (two on the SAME team — Miro/Bogans are
  irreducible by number+kit; Moten on the other team). Kit-split succeeded
  on 4/6 chunks; two chunks' shade clusters were not kit-opposed (Yellow
  washes out under these January exposures).
- GT names: 57 placeholder names resolved deterministically ("Player #0" ->
  Elton Walker etc.); ~12 plays credit an unrostered "Player #00" and a few
  are empty — permanently unscoreable for WHO.
- NR sync unknowable: its audio track is unusable (peaks ~6, ±40-frame
  swings — mic fault) and the calibration mismatch breaks the geometric
  sweep. NR runs at offset 0 in a degraded role.

## What transferred cleanly (the pipeline's spine)

- **Shot detection 89%** on 7-week-older footage with different camera
  positions — arc triggers are venue-robust.
- Clock-shift fitting, chunked AWS prep, kit-split machinery, holder-cache
  build (2,064 segments), rebounds at 78% detection — all ran unmodified.
- The chain surfaced its own faults with numbers (rim frac, sweep medians)
  rather than silently producing garbage.

## Verdict

The January game is a **calibration-era stress test, not a logic test** —
and it cleanly separates the pipeline into what depends on calibration
(zones, make/miss-via-rim, xcam WHO) and what doesn't (detection, arcs,
tracking, rebounds). Actionables, in value order:
1. Per-era court calibration (or auto-recalibration from court lines) —
   would recover zones + a large slice of WHO/m-m in one stroke.
2. Rim-visibility-aware P3 threshold (AUC 0.80 says the signal survives).
3. Venue: this game doubles the dual-number lesson (#10 x3!) — jersey
   uniqueness remains the top WHO lever.

Deliverables: runs/tracking/ledger/{events_v2,possession_events,holders}_13e1ffad.json,
runs/event_demo/fullgame_13e1ffad.mp4 (4-angle review render),
runs/shotdet_ab/{arc_windows,eval_ours_arcwin}_13e.json. No Supabase writes.
