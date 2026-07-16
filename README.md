# Cross-Camera Basketball Tracking — 4-Angle Fusion, Possession & Events

Production multi-camera basketball understanding on a fixed 4-camera rig
(FL / FR / NL / NR): per-camera detection → jersey-anchored identity
tracking → **cross-camera fusion into one identity per player** → a
**ball-first possession state machine** (who holds the ball, every frame)
→ shot / rebound / turnover **events scored against human ground truth**,
plus 4-angle review renders that make every layer visible.

> New here? Read **[docs/GUIDE_POSSESSION_PIPELINE.md](docs/GUIDE_POSSESSION_PIPELINE.md)**
> — the hands-on walkthrough from raw game videos to "marker on the ball
> holder in all 4 angles".

## The production pipeline (current, 2026-07)

```
S3 game videos (4 angles, 1080p)
  └─ AWS prep (GPU): yolo26s detections + RTMPose + jersey anchors    scripts/aws_fullgame_prep_job.py
  └─ AWS ball job:   ball+hoop specialist cache                       scripts/aws_ballcache_job.py
LOCAL (CPU):
  1. camera sync offsets (audio / anchor-geometry sweep)              scripts/sync_anchor_sweep.py, src/uball_cc/fusion/audiosync.py
  2. kit tags for dual jersey numbers                                 scripts/annotate_anchor_kits.py
  3. per-camera tracking: ByteTrack + jersey claims                   scripts/hybrid_track.py
  4. cross-camera identity solve (per player, --pose --carry)         scripts/solve_player_xcam.py
  5. ball trajectories (Kalman+RTS per camera)                        scripts/ball_traj.py
  6. BALL-FIRST holder machine → possession timeline                  scripts/ballfirst_who.py
  7. shot detection + WHO + zones (frozen thresholds)                 scripts/detect_shots.py
  8. make/miss via frozen P3 transfer                                 scripts/shotdet_p1_adapter.py, scripts/shotdet_transfer_eval.py
  9. event assembly + rebounds/turnovers                              scripts/assemble_events.py, scripts/detect_possession_events.py
 10. 4-angle review render (all layers visible)                       scripts/render_fullgame_grid.py
```

Detectors: **yolo26s** (0 player / 1 referee / 2 ball, imgsz 1280) + a
**ball+hoop specialist** (0 Basketball / 1 Basketball Hoop, imgsz 1280,
conf 0.15). Jersey identity: number localizer (yolo11n) + PARSeq OCR +
kit-shade classifier for numbers worn by both teams.

## Current results (blind protocol: thresholds frozen on e6, GT used once)

| metric (of GT shots) | e6 (dev) | c2a (blind) | 2c4 (blind) | 13e1 (blind) |
|---|---|---|---|---|
| Shot detected | 94% | 88% | 91% | 89% |
| WHO (of detected) | 66% | 38% | 62% | 34% |
| Zone / FT | 79% | 65% | 72% | 45% |
| Make/miss | 98% | 95% | 89% | 61% |
| Rebound detection | 93% | 81% | 67% (n=6) | 78% |

Blind-game analyses: [C2A](docs/C2A_BLIND_RESULTS.md) ·
[2C4](docs/2C4_BLIND_RESULTS.md) · [13E1FFAD](docs/13E1FFAD_BLIND_RESULTS.md).
The weak columns have measured causes, not mysteries: shared jersey
numbers across teams (c2a: five duals) and, for 13e1, January footage
against March-era court calibration.

## Repository layout

```
scripts/          every pipeline stage + AWS job launchers (self-documenting headers)
src/uball_cc/     shared library (fusion/homography, audiosync, tracking, detection)
configs/          games registry (games.json), court zones, per-camera calibration
data/             (gitignored) rosters, GT plays, gt_players
runs/             (gitignored) all artifacts: caches, track dumps, ledgers, renders
docs/             design docs, plans, blind-run reports, session logs
```

Key artifacts per processed game (under `runs/`):
- `tracking/ledger/holders_<game>.json` — frame-by-frame ball holder (RLE segments)
- `tracking/ledger/events_v2_<game>.json` + `possession_events_<game>.json`
- `events_fg_*<tag>/` — per-player per-camera track dumps (the fused identities)
- `event_demo/fullgame_<game>.mp4` — the 4-angle review video

## Docs index

The numbered docs (`docs/00..15_*.md`) are the original design phase —
still correct for rig/calibration/eval concepts; superseded in places by
the hybrid (no-SAM3) pipeline above. Living documents:

| Doc | Contents |
|---|---|
| [GUIDE_POSSESSION_PIPELINE](docs/GUIDE_POSSESSION_PIPELINE.md) | **Start here** — end-to-end runbook |
| [BALL_FIRST_SPEC](docs/BALL_FIRST_SPEC.md) | possession state machine design + verdict log |
| [WHO_MASTER_PLAN](docs/WHO_MASTER_PLAN.md) | attribution work, every adopt/refute verdict |
| [STATUS](docs/STATUS.md) | client-facing status + limitations |
| [SESSION_CONTEXT](docs/SESSION_CONTEXT.md) | operational truths, traps, per-game constants |
| [14_games_and_clips](docs/14_games_and_clips.md) | game inventory (S3 paths, GT counts) |

## Ground rules

- **Secrets**: nothing sensitive lives in the repo. AWS launchers are gated
  behind `UBALL_AWS_CREDS_ROTATED=1`; anything that costs money needs
  approval first (hard cap without it: $5/run).
- **Licenses**: Ultralytics YOLO is AGPL-3.0-or-commercial; KPR weights are
  Hippocratic-3.0 — neither ships in a product until cleared.
- **GT discipline**: blind games are scored once; never tune against them.
- `data/` and `runs/` are gitignored by design — ask for the artifact drop
  if you need a processed game.
