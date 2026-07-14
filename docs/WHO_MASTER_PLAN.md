# WHO — the master plan (2026-07-13, after the three-game evidence)

**Objective:** "who took the shot / who did the event" is THE product metric.
Everything below is ordered by measured leverage, not speculation.

**The evidence base (all measured, GT-scored):**
| game | duals | WHO | notes |
|---|---|---|---|
| e6 | 1 | 62% (69% e6-fit fusion) | oracle ceiling 86% (track exists any-cam) |
| 2c490f1a blind | 1 | **59%** | logic generalizes |
| c2a blind | 5 | **37%** | kit collision, not logic |

Error anatomy (e6 deep-dive): ~2/3 of errors have the right player's track
present (ambiguity in crowds — logic-fixable), ~1/3 the track is absent in
the scrum. Rebound-WHO 25-34% (same possession-in-crowd root). Per-signal:
wrists win close-range, 4-cam vote wins long-range, KPR crop recovers
~25-30% of track-absent; the e6-fit fusion did NOT transfer to c2a.

**Standing rule for ALL work below — the three-game scoreboard:** every
change is scored on e6 + 2c490f1a + c2a (142+153+189 shots, one command per
game); adopt only what helps ≥2 games and regresses none by >2pts. New
fitted parameters are fit on e6+c2a jointly and blind-validated on 2c4 or a
fourth game (13e1ffad, the other 1-dual game, is reserved UNTOUCHED).

## W1 — Dual-number identity (measured value: the 37→59 gap, +22pts on dual games)
The single biggest lever. Today a dual number is only usable when number AND
kit shade are legible together — halving anchors exactly where they matter.
1. **KPR-based stream disambiguation**: assign B/W kit streams by appearance
   prototype (per-team embedding built from unambiguous frames) instead of
   hue-only; hue fails under lighting shifts. Gate: c2a WHO +5.
2. **Kit-anchor densification**: propagate kit tags along tracks (a stream
   tagged B at frame N stays B until contradicted) so single-frame shade
   failures don't discard reads. Gate: c2a WHO +4.
3. **Team-context prior**: offense/defense role at release (shooter is on
   the attacking team) — halves the candidate set for duals for free.
   Gate: c2a WHO +3, no e6/2c4 loss.
4. Venue ask (client-side, already in STATUS): bigger jersey numbers.
Target after W1: **c2a ≥ 52-55%**, dual games behave like clean games.

## W2 — Multi-game fusion refit (measured value: +7 within e6, 0 transferred)
The signals are real (wrists close, vote long); the ARBITRATION overfit.
1. Re-measure each signal per game (e5 rows exist for e6/c2a; run 2c4).
2. Refit the routing (D_CLOSE, FT band, wrist-confidence bar) on e6+c2a
   jointly; blind-validate on 2c4. Adopt iff: 2c4 ≥ +3 AND no game regresses.
3. If routing still won't transfer, replace hand rules with a tiny logistic
   over (release_dist, box-height, contest-count, per-signal agreement) —
   fit 2 games, validate on the third, final check on 13e1ffad.
Target: clean-jersey games 62-59% → **68-72%** (ceiling 86%).

## W3 — Track-absence recovery (the 14% ceiling gap)
E3 (KPR on a naive under-ball crop) recovered 5/18-6/29. Upgrade: use raw
DETECTOR boxes at release (they exist even when identity tracks dropped) as
the candidate set, then KPR + jersey-read + W1 team-prior to name them.
Gate: recover ≥40% of track-absent shots on e6+2c4. This raises the CEILING,
not just the score.

## W4 — Rebound/possession WHO (25-34% today)
Apply the W1-W3 picker at the possession-gain instant (today rebounds use
raw nearest-box possession). The rebound layer already isolates WHEN; naming
WHO there inherits every shot-WHO improvement. Gate: rebound WHO ≥50%.

## W5 — Beyond shots (steals/turnovers) — AFTER W1-W4
Blocked on possession-signal quality; revisit once W1-W3 land (the team
timeline gets cleaner automatically as identity streams stabilize).

## Order of execution
1. W1.1 + W1.2 (dual identity) — biggest lever, attacks c2a's 37%.
2. W1.3 team prior — cheap, helps every game.
3. W2 refit — lifts the clean-game ceiling toward 70%.
4. W3 recovery — raises the oracle ceiling itself.
5. W4 rebounds — inherits everything.
Re-run the three-game scoreboard after each step; update STATUS Part 9 table
when a step lands; 13e1ffad stays sealed for the final validation.


---
## W-log (living)
- 2026-07-14 W1 round 1 ADOPTED: kit-birth streams, widened window,
  rq-matching -> e6 69/87, 2c4 63/75, c2a 40/70 (WHO/zone). Refuted with
  numbers: anchor sparsity, possession team-prior (52%), 3 snap variants.
- 2026-07-14 W2 VERDICT: fusion OBSOLETE post-round-1. Fresh signal rows:
  e6 fused 66% == baseline 66% (round-1 absorbed the old +7); c2a fused 25%
  vs baseline 37% (harmful). No refit performed — nothing left to fit.
  Production picker = baseline + rq-matching. Wrist/vote signals archived
  (scripts kept; may matter again after W3 changes the candidate pool).
- NEXT = W3: track-absent recovery via raw detector boxes at release + KPR/
  jersey naming (e6 14% / c2a 21% of shots have no track — the ceiling gap),
  which also partially covers c2a's chain release-instant losses.
- 2026-07-14 W3 VERDICT: NOT adopted. Round-1 tracking shrank track-absence
  itself (e6 18->3, c2a 29->10, 2c4 8 shots); detector-box+KPR recovers 4/21
  of the residue (~+1pt for real complexity). Ceiling gap self-closed.
- 2026-07-14 fresh ceilings (oracle, any-cam): e6 85 / c2a 80 / 2c4 89.
  Current (e5-harness basis): 66 / 37 / 62. THE remaining structure:
  c2a arc-cam ceiling 66 vs any-cam 80 -> cross-cam presence unexploited;
  all games: pick-vs-ceiling gap is now the whole problem (crowd ambiguity
  + c2a release instants). Next candidates: rq-aware cross-cam attribution
  (attribute on the cam with best rq per arc, not fixed arc cam), W4 rebounds.
- 2026-07-14 rq-aware cam ORDERING refuted: all three games down
  (c2a 40->34, e6 69->63, 2c4 63->58) — the instantaneously cleanest view is
  often a bystander's; the arc cam watches the actual launch. Reverted.
  c2a's arc-vs-any-cam ceiling gap needs a different key (identity-aware,
  not geometry-aware, cam selection).
- 2026-07-14 W4 grab-instant attribution refuted: e6 rebounds 15->14,
  c2a 20->20 — the grab-moment under-ball evidence IS the possession signal;
  re-scoring it adds nothing. Rebound-WHO's lever = identity-stream quality
  in paint scrums (a tracking problem, not a scorer problem). Flag kept
  default-off.
- 2026-07-14 R1-LITE (single-crop KPR veto at release) REFUTED: e6 63->56,
  c2a 34->32, 2c4 58->53. WHY: the true shooter's release crop is the most
  blurred/distorted in the scene — single-crop self-consistency favors crisp
  bystanders, veto fires backwards. Does NOT kill R1-FULL (tracklet-level
  split/merge averages many crops; literature +4-10 HOTA). R1-full needs
  frame-sampling infra (~1 crop/2s per stream per cam from S3) — next
  session's build. SAM2 stays last per user directive.
- 2026-07-14 R1-FULL verdict: coarse patches HARMFUL (c2a 40->36, 212k boxes,
  reverted); dense boundary-precise patches (14, 1189 boxes, real n8B/n8W
  confusions) NEUTRAL (40->40) — real switches, wrong moments. NOT adopted
  (gate unmet); machinery kept (r1_sample_frames/tracklet_refine/
  dense_verdict/apply_patches) for post-R2 retry on e6/2c4.
  NEXT: R2 claim carry-over across track death (SportSORT out-of-view
  re-association) — targets arc-cam presence ceiling (c2a 66%).
- 2026-07-14 R2 claim carry-over: c2a WHO 40->40, arc ceiling 66->62 (wrong
  inheritances pollute slightly). REVERTED. NOTE: c2a tracks currently from
  R2 hybrid (fine — solved dumps equivalent at WHO level; canonical code
  restored).
- LADDER PRE-SAM2 EXHAUSTED. Kept total this session: e6 62->69, 2c4 59->63,
  c2a 37->40 (+zones +6-8). c2a's residual gap (40 vs 79 ceiling) =
  release-instant in tip chains + crowd pick — every cheap rung measured.
  NEXT RUNG = SAM2 scrum-windows (user-gated: last resort, ~5% frames).
- 2026-07-14 SAM2 SCRUM-WINDOW PROTOTYPE (final rung): 4/20 hard crowd shots
  recovered (gate was >=8). Misses include d=0px mask-on-ball wrong picks =
  seeds (our tracks at rel-2.2s) already wrong OR mask slid bodies in the
  pile — SAM2 propagates identity, it cannot create it. LADDER COMPLETE:
  every software family measured (geometric, appearance single+tracklet,
  re-association, mask propagation).
- CONCLUSION: the residual hard-crowd WHO gap (~c2a 40 vs 79 ceiling; paint
  FG everywhere) is INFORMATION-LIMITED in this footage. Rational next moves:
  (1) venue levers — BIGGER JERSEY NUMBERS (multiplies anchor density, the
  one change that lifts every layer), camera re-aim, NR/NL reliability;
  (2) user's 4-angle video review to confirm the unsolvable class visually;
  (3) engineering effort re-aims at precision/robustness of what works
  (detection 88-94, make/miss 89-98, zone 70-87) rather than the last WHO %.
