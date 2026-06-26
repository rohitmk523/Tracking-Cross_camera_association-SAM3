# 08 · Spatial World Model

The world model is the **substrate the VLM reasons over**. It is the deterministic, structured
representation of "what is happening on the court," built entirely by the CV pipeline. The
design principle (from the research): **LLMs are good at *consuming* structured state and bad
at *generating* it → we build the state deterministically and let the LLM only reason over it.**

## Two layers
1. **Per-frame state** — the raw spatial truth (positions, pose, ball, possession). High
   volume; lives on the edge / in a store; NOT all shipped to the LLM.
2. **Event stream** — discrete, semantically meaningful moments distilled from the per-frame
   state (shot, make/miss, possession change, screen, block, turnover, drive). Low volume;
   this is what drives narration.

## Per-frame schema (illustrative)
```json
{
  "court_id": "venue_07", "game_id": "...", "frame": 1234, "t_sec": 41.1,
  "court_cm": [2143.7, 1426.4],
  "ball": { "court_xy_cm": [1180, 690], "z_est": 240, "owner_id": 7 },
  "players": [
    { "id": 7, "team": "A", "number": 11, "name": "Dioscar Gomez",
      "court_xy_cm": [1175, 705], "velocity_ms": 2.6,
      "seen_by": ["FL","FR"], "members": [["FL",4],["FR",9]],
      "pose_2d": { "FL": { "r_wrist": [1322,751,0.93], "...": [] } } }
  ],
  "referees": [ { "id": "R1", "court_xy_cm": [300, 200] } ]
}
```
(Same shape as the sample already produced in DEMO_UBALL — see `spatial_state_sample.json`.)

## Event schema (illustrative)
```json
{ "event": "shot_attempt", "t_sec": 41.5, "shooter_id": 7, "shot_type": "fadeaway?",
  "defenders_near": [12, 5], "contest": true, "outcome": "blocked",
  "blocker_id": 12, "possession_team": "A", "frame_window": [1230, 1265],
  "confidence": 0.72, "needs_frame_check": true }
```
`needs_frame_check` flags events where pixels are required to confirm (shot type, contact,
who-touched-last) → triggers the VLM frame-clarifier ([09](09_vlm_narration.md)).

## Event detection (CV side, deterministic)
Derived from positions + ball + pose:
- **Possession**: nearest player to ball, sustained; possession changes.
- **Shot attempt / make / miss**: ball trajectory toward hoop + rim region + outcome
  (reuse the existing shot-detection capability; ~95% in prior work).
- **Screen, drive, pass, turnover, block/contest**: relational + pose heuristics (e.g.
  defender vertical extension near a shot = contest/block candidate).
- Each event carries a `confidence` and `needs_frame_check`.

## Volume control (critical for the VLM)
At ~12 players × 17 joints × 30 fps the per-frame state is large (~12 KB/frame in the sample).
**Do not stream raw per-frame to the LLM.** Instead:
- Keep per-frame state in a buffer/store (edge → object store).
- Feed the LLM the **event stream + a compact rolling possession context**, not every frame.
- **Per-entity decomposition** when narrating (avoid the documented "role confusion" failure
  where the LLM mixes up which player did what).

## Storage
Edge writes per-frame state (compact binary/parquet) + an event log (JSON) per game. The
event log + on-demand frame handles are what the cloud narration layer consumes.

## Why this is the leverage point
The closer the world model mirrors reality, the more the VLM can reason from state alone —
approaching "a VLM watching every frame" at a fraction of the latency/cost. Pose + accurate
cross-camera positions are what make descriptive (not just classification) narration possible.

## Implementation status (2026-06-26)
**Event deriver built** (`src/uball_cc/fusion/events.py`, `scripts/derive_events.py`): consumes
the fused world-state and emits a deterministic event JSON — `possession` (nearest-player-to-ball,
temporally voted), `pass` / `turnover` (possession transfer same/cross team), `transition`
(team-centroid sweep) — each with `confidence` + `needs_frame_check`. Ball-optional: with no ball
trace it emits only team-spatial events and a caveat (honest degradation, never guesses possession).

**The ball is the dependency, and it's hard here.** The tracker output carries only player(0)/ref(1),
no ball(2). The detector *fires* on the ball in ~92% of frames but at very low confidence (full-frame
NR: max score 0.52; only 37% of frames ≥0.05) — the small-object signature (ball ≈13 px @1080p).
- `scripts/extract_ball.py` — full-frame ball → project → fuse. On e6 gave **16/358 frames (4.5%)** — unusable.
- `scripts/track_ball.py` + `fusion/ball.py` — **SAHI tiling** (per-tile detection ≈3× effective ball
  resolution) feeding a **constant-velocity Kalman ball tracker** that gates false positives and
  interpolates gaps. Near cameras only (they see the ball + own the court). The Kalman tracker is
  unit-tested (`tests/test_events.py`): given a real trajectory it follows it, gates FPs, interpolates.

**SAHI result on e6 NR (2026-06-26) — machinery works, detector doesn't (yet).** SAHI raised raw ball
frames 4% → **61%** and the tracker filled to 100%, BUT the raw candidates are **95% clustered within
300 cm of one fixed point** at **median confidence 0.059** (max 0.30): the "ball" is a **persistent
false positive** at top-center, not the moving ball. So the event deriver produces possession/pass
events but they're proximity-flips around a pinned point — **not trustworthy**. **Conclusion:** the
pipeline (SAHI → Kalman → events) is built and validated; the blocker is **ball-detection quality** —
the 3-class detector can't localize this small/fast ball at usable confidence. **Next = a dedicated
ball model** (fine-tune on ball crops from our footage, or a TrackNet-style ball-trajectory net), a
data+training task. Tracker/threshold tuning cannot fix a clustered false positive.

**Deferred (needs more than positions):** **shot make/miss** needs the rim *in image space* + the ball's
*flight* trajectory — the planar floor homography is wrong for an elevated ball, so a court-projected
ball can't tell a make from a miss. Rim detector exists (`runs/rfdetr-rim-near-v1`) but isn't wired in.
Possession/pass/turnover come first (ball-on-floor, where the homography holds); shots are the next build.
