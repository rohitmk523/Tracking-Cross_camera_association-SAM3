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

## Camera re-aiming (no new cameras — re-angle the existing 4)

Constraint: we cannot add cameras, only **re-aim the four we have.** The blind-zone map
(`runs/tracking/camera_blind_zones.jpg`) shows the exact problem to attack: **each camera
covers its own half and goes blind on the opposite half** — so a player is realistically
seen by only **~2 of the 4 cameras at once** (the pair on his end). When those two both slide
onto a same-kit team-mate in a pile-up, there is no clean third camera to override them. That
2-camera-deep coverage is the ceiling; re-aiming should raise it toward 3 where it matters.

Measured blind zones (fraction of player-visible frames each camera cannot see him):
FL 51%, NL 45%, FR 39%, NR 26% — all concentrated on the **far half** from each camera.

### What the real-coverage map says (`runs/tracking/camera_real_coverage.jpg`)
This map counts, per court zone, how many cameras **correctly hold the player** (not just see
him). Two findings that overturn the naive advice:

1. **The near sideline is the strongest zone (2–3 cameras hold).** NL and NR are doing their
   essential job — watching under the rims, where FL/FR are too far to tell players apart.
   **Do not re-aim them toward centre;** that would abandon the one thing only they can do, and
   the centre isn't even the main gap.
2. **The real weak zone is the LEFT basket / left third (0.0–0.7 cameras hold) — and it's
   asymmetric**, much weaker than the right. Only **5 of 26 weak cells are in the centre; 21
   are at the ends/keys.** The problem is the *ends*, not the middle.

### Concrete, data-driven re-aiming (FL/FR only; leave NL/NR on the rims)

1. **NL — investigate first (your own hypothesis).** NL has catastrophic drop-outs for some
   players (4–14%) and the left third it should own is the weakest zone on the court. That
   signals a **recurring obstruction or a mis-aimed view of the left basket area** — a physical
   check of what NL actually frames on the left is the single highest-value action.
2. **FL, FR — fill the frame with court.** These are the flexible cameras. If either wastes
   field of view on crowd/ceiling/floor, re-aim so the court fills the frame — every extra pixel
   on a far-court player extends how long he's held before he shrinks out. Aim them to reinforce
   the **weak left/end zones**, not the already-strong near sideline.
3. **Do not chase the centre.** The centre already gets 1–2 holds; the ends are where coverage
   collapses to <1. Point FL/FR's attention at the under-covered **keys and corners**.

### Honest limit of angle-only changes
Re-aiming **cannot** fix far-baseline resolution — a side camera at fixed height sees a distant
player as a handful of pixels no matter the angle. Angle changes can recover a **mis-aimed or
obstructed** camera (likely the case for NL on the left) and squeeze more court into FL/FR's
frames, but the end/corner gaps are partly bound by where the cameras are mounted. Expected
effect: fixing NL's left-side view is real and specific; the rest is incremental. The larger,
more reliable gains are the **software levers** (jersey re-seed, masklet-split) — pair them.

---

## Execution order

1. ✅ SAM3 single-object + geometric fusion (engine proven: referee 96% / 8 cm).
2. ⏳ **#1 jersey-confident seeding** (AWS, running) → re-seed-on-drift loop.
3. ⏳ **#2 long-clip duration test** (clip pulled, queued).
4. ▢ #2 masklet-splitting + #3 basketball appearance model (software).
5. ▢ **Hardware proposal to client** — one overhead camera (Recommendation A), backed by the
   measured per-camera data.
6. ▢ Full-game validation once the software levers land.
