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
