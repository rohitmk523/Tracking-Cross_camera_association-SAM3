# Solution Plan — closing the same-kit identity gap

## The problem, precisely

SAM3 tracks one seeded player well **until identical-kit team-mates cluster** (screens,
rebounds, dead-ball scrums). Then its outline slides from our player onto a team-mate, and —
critically — **this drift is correlated across all four cameras**, because every camera views
the pile-up from a similar low side angle, so they all lose the player the same way at the
same instant. Jersey numbers, which would break the tie, are least readable in exactly those
crowded moments. Measured today: 63–85% fused tracking on players, 96% on the distinct-kit
referee (the ceiling when identity is unambiguous).

Root cause is **geometry + kit design**, not the tracker: four side-angle cameras cannot
separate two identical-looking bodies that overlap in every view.

---

## Software levers (ranked by expected gain ÷ effort)

### 1. Jersey-confident seeding + re-seed on drift  ·  RUNNING NOW (AWS)
Production-realistic: seed each player from a **confident number reading** (not a human box),
and — the new part — **re-seed the moment drift is detected** (the number on the mask flips,
or the mask disagrees with the other cameras' consensus). Each re-seed re-anchors SAM3 to the
confirmed player before it drifts far, so a mask can only be wrong for the short interval
between a drift and the next confident number. **Status:** jersey-seeded pass launched
(`i-0f4dac85e305988cc`); re-seed-on-drift loop is the follow-on once it lands.
**Expected:** lifts the 63–68% players toward the #11 (85%) level; provable on the same GT.

### 2. Masklet splitting at drift points  ·  offline, no new GPU
Where the number on a mask contradicts its seeded identity, **cut the mask there**, keep only
the segment that matches, and re-acquire the correct player for the rest from the other
cameras' court position. Pure post-processing on data we already have; directly attacks the
correlated-drift coverage gap the gating layer couldn't.
**Expected:** recovers the frames where a camera silently tracked the wrong person.

### 3. Basketball-specific appearance model  ·  ~3 days, AWS
Our current appearance embedding is generic. Fine-tune it on basketball players
(DeepSportradar, permissive licence). Even small same-kit differences — sleeve, socks, build,
tattoos — become a usable signal to tell two green players apart when numbers are hidden.
**Expected:** helps the residual cluster moments where jersey fails.

### 4. Multi-hypothesis tracking  ·  larger build
When two same-kit players are ambiguous, **track both hypotheses forward** and let the next
confident number reading collapse them to the truth — retroactively relabelling the whole
segment (our offline advantage). This is how the professional systems handle contact; it is
the most powerful software option and the largest build.

### 5. Two-tier confirmed/temporary identity  ·  designed
The long-term re-ID layer ([IDENTITY_REID_DESIGN.md](IDENTITY_REID_DESIGN.md)): a returning
player (sub / left court / scrum exit) is a **temp id** tracked by geometry + team colour +
appearance until a number confirms it, then merged retroactively. Handles the full-game
leave-and-return case that a 60-second window doesn't exercise.

### Validation running in parallel
**Long-clip (3-minute) duration test** — does the 96%-referee / 85%-#11 behaviour hold over
minutes and many events, or decay? Clip pulled; runs after the jersey-seed pass.

---

## Hardware recommendations

The measured per-camera data says the four cameras are **not the problem individually — their
shared geometry is.** Average "held the player" across all five: FR 66%, NR 66%, FL 49%,
NL 40%, and **no camera is reliably above ~85%** because they all sit at similar low side
angles and lose clustered players together.

### Recommendation A — add one overhead centre-court camera  ·  highest impact
A single camera mounted **directly above centre court, looking straight down**, sees a pile-up
**from above, where players do not occlude each other.** This breaks the exact failure mode
that no software fully fixes: correlated same-kit drift in clusters. It is also how the
category leaders (Hawk-Eye, Second Spectrum, Noah) achieve their accuracy — placement, not
just algorithms. **One overhead camera would likely move same-kit tracking from ~65% toward
the 90s**, because the hardest moments (bodies overlapping in side views) become clean from
directly above.

### Recommendation B — raise / re-aim the two far cameras (FL, FR)
FL in particular swings between 22% and 86% depending on the player — a sign its view is
partly obstructed or too low, so far-court players fall below or behind foreground bodies.
Mounting FL/FR **higher, with a steeper downward angle**, flattens the occlusion and steadies
far-court coverage.

### Recommendation C — check NL's placement
NL shows **catastrophic drift for specific players** (4%, 14%) while being fine for others —
a localised blind spot or a recurring obstruction in its view of one part of the court.
Worth a physical check of what NL sees where those players spend time.

### Recommendation D (premium) — overhead rim cameras for shot analytics
Separate from tracking: an **overhead, rim-axis camera per hoop at higher frame rate** turns
make/miss and shot-arc analytics from inferred to directly measured — the same principle the
market leader uses, replicable on our low-cost stack (documented separately).

### Cost/impact summary
| Change | Effort | Expected effect |
|---|---|---|
| **A. 1 overhead centre camera** | 1 camera + mount + re-calibration | **Same-kit tracking ~65% → 90s** (breaks cluster occlusion) |
| B. Raise/re-aim FL, FR | re-mount + re-calibration, no new hardware | Steadier far-court coverage |
| C. Inspect/adjust NL | physical check | Fixes localised drop-outs |
| D. Overhead rim cameras | 2 cameras, higher fps | Near-certain make/miss + shot metrics |

**Bottom line on hardware:** software will keep pushing same-kit identity up from 65%, but the
clean, durable fix is **one overhead camera** — it changes what the system can *see*, and the
occlusion that defeats every algorithm simply isn't there from above.

---

## Execution order

1. ✅ SAM3 single-object + geometric fusion (engine proven: referee 96% / 8 cm).
2. ⏳ **#1 jersey-confident seeding** (AWS, running) → re-seed-on-drift loop.
3. ⏳ **#2 long-clip duration test** (clip pulled, queued).
4. ▢ #2 masklet-splitting + #3 basketball appearance model (software).
5. ▢ **Hardware proposal to client** — one overhead camera (Recommendation A), backed by the
   measured per-camera data.
6. ▢ Full-game validation once the software levers land.
