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
