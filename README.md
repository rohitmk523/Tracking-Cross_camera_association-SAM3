# Cross-Camera Basketball Tracking — Ball Possession (WHO)

Multi-camera basketball tracking on a fixed 4-camera rig (FL/FR/NL/NR):
per-camera detection → jersey-anchored identity tracking → **cross-camera
fusion into one identity per player** → a **ball-first possession state
machine** that knows **who has the ball at every frame** — rendered as a
glowing ring under the holder, correct in all four angles simultaneously.

That is the whole product right now. Shot detection, make/miss and event
assembly were deliberately removed from the working pipeline (2026-08-01
possession-core pivot — recoverable from git history).

> **Start here: [`game_plan.md`](game_plan.md)** — the current mission and
> plays. Then [`docs/GUIDE_POSSESSION_PIPELINE.md`](docs/GUIDE_POSSESSION_PIPELINE.md)
> for the hands-on runbook.

## The pipeline

```
4 angle videos (1080p)
  GPU  build_dets_cache_yolo.py     players/referees   (yolo26s @ 1280)
  GPU  build_ball_cache.py          ball + hoop        (specialist @ 1280, conf .15)
  GPU  extract_pose.py              ankles (court projection)
  GPU  extract_jersey_anchors.py    jersey reads = identity ground wire
  CPU  annotate_anchor_kits.py      kit split for numbers worn by both teams
  CPU  hybrid_track.py              per-camera ByteTrack + jersey claims
  CPU  solve_player_xcam.py         cross-camera fusion (--pose --carry)
  CPU  ball_traj.py → ballfirst_who.py → runs/tracking/ledger/holders_<game>.json
       render_fullgame_grid.py --ring        ← the deliverable
```

For a ~55-min game the GPU stages run on AWS (3 instances in parallel,
~2.2h wall, ~$9); short clips run fully local on any CUDA GPU. Camera sync
(`sync_anchor_sweep.py`, audio tools) and **per-era court calibration**
(`refit_calibration.py`) are the two silent killers — check both before
trusting fusion on new footage.

## Repository layout

```
game_plan.md      the current mission — read first
scripts/          the 8-stage chain + sync/calibration + AWS & training launchers
src/uball_cc/     shared library (homography, audiosync, tracking, detection)
configs/          games registry, per-camera calibration
data/             (gitignored) rosters, GT plays
runs/             (gitignored) caches, track dumps, holder ledgers, render sources
docs/             runbook, design docs, blind-run records, session logs
```

Per processed game (`e6fba750`, `c2a354fe`, `2c490f1a`, `13e1ffad` on disk):
- `runs/tracking/ledger/holders_<game>.json` — frame-by-frame ball holder (RLE)
- `runs/events_fg_*<tag>/` — fused per-player per-camera boxes
- `runs/event_demo/fullsrc_<game>_<ANG>.mp4` — low-res render sources

## Detectors & weights

| model | classes | imgsz |
|---|---|---|
| `runs/yolo26s-1280-ourdata-v1_fetch/.../best.pt` | player / referee / ball | 1280 |
| `runs/ball_yolo26s_fetch/.../best.pt` | Basketball / Basketball Hoop | 1280 |
| `runs/jersey/*` | number localizer + OCR + legibility | — |

`imgsz=1280` on every predict — the ball is ~6px at the default 640.
A unified 4-class retrain is planned (game_plan P2).

## Ground rules

- **Secrets**: never committed; AWS launchers gated behind
  `UBALL_AWS_CREDS_ROTATED=1`; spends need approval (cap $5/run without).
- **Licenses**: Ultralytics YOLO is AGPL-3.0-or-commercial — cleared before
  anything ships.
- **GT discipline**: blind-game ground truth is scored once, never tuned on.
- `data/` and `runs/` are gitignored — ask for the artifact drop.
- Removed ≠ destroyed: git history and `~/.Trash/uball_cleanup_2026-08-01/`
  (with manifest) hold everything the pivot removed.
