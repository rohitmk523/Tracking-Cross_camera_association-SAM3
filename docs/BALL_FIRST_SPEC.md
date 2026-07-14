# BALL-FIRST: rendering spec + event-engine v3 (from user visual review, 2026-07-14)

## The user's diagnosis — confirmed against the frames
Reviewed frames confirm it: in the cited moments the ORANGE ball detection is
sitting next to the true actor (e.g. hard-case 1: ball beside #22 McKnight,
who GT says shot it — while the pipeline credited Baad), and in the
highlights reel the box sits on the eventual shooter (#10) for the whole
clip while #6 visibly carries the ball. The information is IN our data;
the architecture never uses it as a continuous signal. Verdict on the
"unsolvable" label: WRONG for these cases — a human following the ball
resolves them trivially; so must we.

**Core principle (user):** every event except FOUL is a BALL story. Track
the ball perfectly; maintain WHO HOLDS IT at every moment (the "yellow
player"); events are TRANSITIONS of that state:
- holder A -> flight -> holder B (same team) = PASS
- holder A -> flight toward hoop -> rim interaction = SHOT by A (who = the
  last holder, BY CONSTRUCTION — no release-instant attribution needed)
- rim -> new holder = REBOUND by that holder (off/def by team)
- holder A -> flight -> holder B (other team, no rim) = STEAL/TURNOVER
- holder at FT spot, play stopped = FREE THROW (one event per ball cycle —
  duplicates impossible by construction)

## Why our previous "possession" failed and why this is different
The v1 possession vote was per-frame nearest-box on the OLD ball cache
(3-12% coverage) and was never rebuilt after the detector upgrade (now
0.876 mAP, 47-74% per-cam coverage, union across 4 cams far higher). The
old ball-interp refutations are OBSOLETE EVIDENCE (measured at low recall).
Event engine v2 went shot-trigger-first and attributes WHO at one computed
instant — the exact step the review shows failing while the ball's own story
is unambiguous.

## Event engine v3 — the possession state machine
1. **Ball trajectory layer**: link per-frame ball detections per cam
   (Kalman + bounded gap bridge, their fusion repo's track_clean recipe),
   then fuse across 4 cams on the court plane -> ONE ball track with
   confidence, HOLD/FLIGHT/RIM labels per segment (speed + proximity).
2. **Holder layer**: ball in HOLD near identity stream S for >=k frames ->
   holder := S (hysteresis; multi-cam agreement weighted). Holder persists
   through short occlusions unless FLIGHT begins elsewhere.
3. **Event layer**: emit events from state transitions per the table above.
   Points = holder's court zone at the HOLD->FLIGHT transition (the true
   release, given by the state machine, not trajectory walk-back).
4. Make/miss stays P3 (proven 89-98%).

## Rendering spec (all future review/demo videos)
- 4-angle grid as now, plus:
- **YELLOW thick box + name = current ball HOLDER** in every view, at every
  frame (the star of the video). Holder changes = yellow jumps.
- Ball: orange dot + short trail (last 0.5s) so flight reads visually;
  dashed orange line during FLIGHT from last holder to landing.
- Hoop: red box. Identity streams: thin colored boxes + names (as diag reel).
- Right panel: current holder name (big), then the running event feed
  (events appear AT their transition), GT feed below.
- Nothing rendered that the pipeline doesn't believe — the video IS the
  state machine made visible; if the yellow box is wrong on screen, the
  event will be wrong in the JSON. One artifact to debug both.

## Validation protocol (unchanged discipline)
Same three-game GT harness; the state machine must beat WHO 69/63/40 and
specifically the user's cited cases (highlights 00:19 #6-carry; hard cases
1/3/5). New per-event metrics: PASS precision spot-check (no GT), REBOUND
det/WHO, FT dedup (exactly 1 event per GT FT). 13e1ffad stays sealed.

## Order of work
1. Ball trajectory + HOLD/FLIGHT/RIM segmentation (per cam -> fused).
2. Holder state machine + yellow-box render (visual validation FIRST on the
   user's cited clips — the fastest feedback loop we have).
3. Event emission from transitions -> three-game scoreboard.
4. Swap into events_v2 assembler as engine v3 if it gates.


## Prototype findings (2026-07-14, scripts/ballfirst_proto.py)
- **CASE 1 SOLVED**: state machine reads 'Baad hold -> pass -> McKnight hold
  1.5s -> rim' => SHOT by McKnight, CORRECT where release-attribution said
  Baad. The user's architecture works on the exact case that defeated all
  twelve scorer-side approaches.
- CASE 5 (quick catch-and-finish): plain hysteresis misses the catch; fast
  catch-switching over-switches to defenders. REQUIRED RULE for the full
  build: at FLIGHT-END, assign holder by the ball TRAJECTORY LANDING point
  (extrapolate last flight segment; the box it lands in gets the catch),
  proximity votes only confirm. Also: e6 dual #6 needs the kit-split rerun
  (name shows '#6?') — e6 tracks predate the kit-birth fix.
- Renders: runs/event_demo/ballfirst_case{1,5}.mp4 (yellow holder + ball +
  hoop + gray streams).

## Engine v3 full-game status (2026-07-14 night)
- scripts/ballfirst_who.py: full-game holder machine (catch + landing rules).
  c2a: v3 alone 39% ~ baseline 40%. **COMPLEMENTARITY IS THE FINDING:
  both-right 44 | v3-only 20 | baseline-only 23 | both-wrong 79 ->
  union ceiling 52% (vs 40).** The +12pt prize is arbitration.
- Hold-length arbiter (>=15f clean hold -> v3) REFUTED: 37% (stale long
  holds are v3's own failure mode — length doesn't discriminate).
- NEXT arbiter candidates (measure each, no fitting on c2a alone):
  (a) v3-baseline AGREEMENT as high-confidence + landing-rule-fired flag,
  (b) arc rq routing (clean release -> baseline; dirty -> v3),
  (c) per-class routing (FG/paint -> v3, long-range -> baseline),
  (d) vote-mass margin of the hold.
- e6 kit rerun DONE (n6B/n6W solved) — e6/2c4 v3 runs pending.

## Arbiter round verdict (2026-07-14, late)
All four pre-declared arbiters FAIL to capture the union (best = ties
baseline; none beats it on any game). Unions exist everywhere (e6 77%,
2c4 74%, c2a 52%) but no shallow feature (agreement/rq/class/dist)
discriminates v3's wins from its losses. STOP RULE: no further rule-fishing
(= fitting the test set). ROOT CAUSE: the holder machine runs on RAW
per-frame ball detections — spec layer 1 (Kalman trajectory + 4-cam fusion)
was skipped for speed. Next build = layer 1 properly, then v3 alone should
rise past arbitration-need. Shot-WHO stays baseline (69/63/40) until then.

## v3 FIRST PRODUCTION ADOPTION (2026-07-14/15 night): REBOUNDS
- Trajectories (layer 1) on shot-WHO: mixed (2c4 +3, e6 0, c2a -3) — shot-WHO
  stays baseline 69/63/40. v3's shot value remains only-in-union; parked.
- **REBOUNDS adopted on the v3 emission path** (first NEW holder after each
  miss emits the event — time AND name from the ball story):
  e6 detection 81->93% (50/54), WHO 15->16 of more-detected (all-GT 28->30%);
  c2a WHO 20->25/78 (all-GT 21->26%). 2c4 GT n=6, noise.
  Probe upside (37/28%) partially eaten by event-time matching tolerance —
  emission-time refinement is a known follow-up.
- Holder caches: runs/tracking/ledger/holders_{game}.json (RLE, all 3 games,
  built by ballfirst_who.py). detect_possession_events consumes them when
  present; falls back to possession segments otherwise.

## Emission-refinement round (2026-07-15): ALL FOUR REFUTED
Measured on the assembler/rebound harnesses, each vs the adopted state:
1. Rebound grab-window widened [f0-4,+4.5s]->[f0-15,+6.0s]: e6 det 93->87%
   (earlier tip holders displace the real securer). REVERTED.
2. Putback dedup by (t,player): e6 emitted 97->69, det -3 (kills real
   putback chains — same player CAN rebound twice). REVERTED.
3. Ball-cycle dedup (shot-family pairs <=8s, no holder-start between, keep
   better rq): e6 det 134->129, m/m 98->93 — holder-cache GAPS make "no
   holder between" false-positive on real consecutive shots. REVERTED.
4. Narrowed to FT-pairs-only <=6s: still -2/-3 det everywhere (e6 131,
   c2a 164, 2c4 137) — real GT FT pairs land within 6s. REVERTED.
LESSON: the tier gate already killed phantom spam (83/381/219 -> 12/8/15);
holder-cache coverage is not yet dense enough to power NEGATIVE evidence
("no holder change happened") — only POSITIVE evidence (holder at t = X).
Adopted state stands: e6 94% det / 59 all-correct, c2a 88/27, 2c4 91/52;
rebounds e6 93% det.

## Landing-rule velocity bugfix (2026-07-15): NEUTRAL, kept
flight_vec stored vx=vy=0.0 (traj NPZ velocities were loaded then discarded)
— the LANDING rule extrapolated with zero velocity since birth, i.e. "landing
point" = last flight position. Fixed to carry real smoothed vx,vy (+ pick the
highest-conf flight cam). Measured: shot-WHO identical on all 3 games
(60/36/62 — rule fires too rarely to move totals); rebounds e6 50/54=93%
WHO 16 unchanged, c2a WHO 25->24 (noise), 2c4 unchanged. KEPT for spec
conformance; verdict = the landing rule is not where v3's shot gap lives.

## Team-gate + ambig-fallback round (2026-07-15)
Miss diagnosis first (probe): v3's shot misses are DEFENDER picks (e6 28/53,
c2a 76/106, 2c4 25/53) + 16 AMBIG on 2c4; picked-holder hold length at pick
median 20-32 frames (not flickers — the machine genuinely believes the
defender holds it through the contest).
1. ATTACK-TEAM-GATED walk-back (skip defending-team holders; atk per
   (rim-end, half) from majority-of-baseline with change-point halftime
   search): REFUTED HARD — e6 60->40, c2a 36->30, 2c4 62->37. Root cause:
   the atk tables only reach 57-63% consistency (c2a "halftime" fit 520s =
   noise) — either cam-prefix is not a reliable rim-end label or these rec
   games don't keep ends cleanly; gating on a ~60% table poisons picks.
   Code removed; the DEFENDER diagnosis stands, this atk source is dead.
2. AMBIG FALLBACK (v3 pick resolves to '?' -> baseline's answer): 2c4
   86->95 (62->68%), CROSSES baseline 88 (63%); e6/c2a unchanged (no AMBIG).
   2c4 n8B/n8W kit streams exist — the 16 AMBIG are phantom-number streams
   (misread jerseys not on either roster). Fails the >=2-games adoption gate
   as a shot-path switch -> shots STAY baseline; kept as scorer metric.
Shot-WHO state: baseline 69/40/63 vs v3-best 60/36/68. Ball-first wins
rebounds (adopted) + one game's shots; crowd DEFENDER confusion at release
is the remaining shot gap on all engines.
