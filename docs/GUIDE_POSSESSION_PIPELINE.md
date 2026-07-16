# Runbook: from game videos to "who has the ball" in all 4 angles

Audience: new developer with repo access. Goal: understand and run the
chain that produces the fused 4-angle possession view — every player
boxed and named in every camera, the current **ball holder highlighted**,
and the event feeds — like `runs/event_demo/fullgame_<game>.mp4`.

Mental model, one line: *each camera gets identity-tracked on its own,
jersey numbers pin identities, the four cameras are fused into one player
per person, the ball gets its own smoothed story, and a state machine
decides who HOLDS it at every frame — everything else (shots, rebounds,
the yellow box) hangs off that.*

---

## 0. Setup

```bash
git clone <repo> && cd Tracking-Cross_camera_association-SAM3
python3.11 -m venv .venv
.venv/bin/pip install ultralytics opencv-python numpy scipy pandas pyarrow \
    scikit-learn boto3 supervision
```

- Everything runs from the repo root with `.venv/bin/python`.
- `data/` and `runs/` are **gitignored** — ask Rohit for the artifact drop
  of a processed game (or the S3 fetch commands below for a new one).
- AWS: only needed for NEW games. Launchers refuse to run without
  `UBALL_AWS_CREDS_ROTATED=1` exported. **Ask before launching anything
  that costs money.**

Two per-game constants files you must know:
- `scripts/game_meta.py` — `GAME_OFFS` (inter-camera frame offsets:
  `camera_frame = reference_frame + offs[angle]`) and `GAME_CHUNKS`
  (the 600s slicing grid, e.g. `0_600 ... 3000_144`).
- `data/rosters/<game>.json` — `{num, name, team}` per player, plus
  optional `"kit_team"` map when the light/dark kit → team mapping is
  inverted (e.g. e6, 13e1ffad: the LIGHT kit is team 1).

---

## 1. FAST PATH — a game that's already processed

Processed games: `e6fba750`, `c2a354fe`, `2c490f1a`, `13e1ffad`.

Who has the ball, frame by frame, is **already computed**:

```python
import json
doc = json.load(open("runs/tracking/ledger/holders_e6fba750.json"))
# doc["segments"] = [[start_frame, end_frame, "n11"], ...]  @ 29.97 fps
# stream id "n11" -> jersey #11; dual numbers carry a kit letter: "n6B"/"n6W"
```

Resolve stream id → player name via the roster (see `resolve()` in
`scripts/render_fullgame_grid.py` — copy it; it handles duals + kit_team).

Per-camera boxes for any player at any frame live in the track dumps:
`runs/events_fg_<pfx>_<tag>/<game>_<tag>__n11__FL.json`
(`{"frames": {"<cam_frame>": {"box": [x1,y1,x2,y2], ...}}}`, pixels at
1920×1080; `pfx` is empty for e6, else first 3 chars of the game id).
To look up "player P in camera A at reference frame f": camera_frame =
`f + GAME_OFFS[game][A]`, chunk = `f // (600*29.97)`.

Render the full review video (needs the low-res sources
`runs/event_demo/fullsrc_<game>_<ANG>.mp4`):

```bash
.venv/bin/python scripts/render_fullgame_grid.py --game e6fba750 \
    --out runs/event_demo/my_render.mp4          # add --t0 600 --dur 60 for a slice
```

That script is also the reference implementation for any custom graphic
(possession circle etc.): it reads holders + tracks + ball + hoop and
draws everything. Start your graphics work by copying it.

---

## 2. FULL PATH — process a NEW game end to end

### 2.0 Prerequisites
- Game registered in `configs/games.json` (`gid8`, `s3_prefix`).
- Roster + GT plays built into `data/rosters/<gid8>.json` and
  `data/plays/<gid8>_raw.json` (pull from Supabase `games.roster_team1/2`
  and `plays`; resolve placeholder names like "Player #0" to the unique
  roster owner).
- Add `GAME_CHUNKS[gid8]` to `scripts/game_meta.py`
  (`min(600, duration - start)` naming; e.g. 3144s → last chunk `3000_144`).

### 2.1 AWS caches (GPU, ~$9-10 per game, ask first)
```bash
export UBALL_AWS_CREDS_ROTATED=1
.venv/bin/python scripts/aws_fullgame_prep_job.py --gid8 <gid8> \
    --duration-s <secs> --instances 3 --i-rotated-creds      # dets+pose+anchors, ~2h
.venv/bin/python scripts/aws_ballcache_job.py --gid8 <gid8> \
    --weights runs/ball_yolo26s_fetch/runs/detect/runs/ball-yolo26s-1280-v1/weights/best.pt \
    --chunks <comma-chunks> --failsafe 9000 --i-rotated-creds # ball+hoop, ~2h
# when instances terminate:
.venv/bin/python scripts/aws_fullgame_prep_job.py --gid8 <gid8> --fetch
.venv/bin/python scripts/aws_ballcache_job.py   --gid8 <gid8> --fetch
cp -n runs/fullgame_<gid8>/runs/dets_cache/* runs/dets_cache/
cp -n runs/fullgame_<gid8>/runs/anchors/*    runs/anchors/
cp -n runs/fullgame_<gid8>/runs/pose_cache/* runs/pose_cache/
```
(Add the new game's offsets entry to the launcher's `OFFSETS` dict —
zeros are fine at prep time; real offsets are applied locally later.)

### 2.2 Camera sync offsets → `GAME_OFFS`
Wrong offsets silently corrupt the fusion (a past +1-vs-−8 error cost
real accuracy). Two instruments, use both:
- **Audio** (authoritative when confident): extract full-game audio
  in-region (`scripts/aws_audiograb_job.py`), then run
  `uball_cc.fusion.audiosync.audio_offset_seconds` on 600s slices at
  3-4 positions. Trust peaks ≥ 3; frames = `round(seconds * 29.97)`.
- **Anchor-geometry sweep** (`scripts/sync_anchor_sweep.py --game <gid8>`)
  — validates/adjudicates; also detects stale court calibration (median
  pair distance ≫ 100cm ⇒ the calibration doesn't match this footage era).
Write the result into `GAME_OFFS` in `scripts/game_meta.py`.

### 2.3 Kit tags for dual numbers (numbers worn by BOTH teams)
```bash
.venv/bin/python scripts/annotate_anchor_kits.py --game <gid8> --tag <tag> \
    --dual-numbers 6      # repeat per chunk; adds "kit": "B"|"W" to anchors
```
If clusters come back "not kit-opposed" on a chunk, that chunk's duals
stay untagged (they simply won't claim) — note it and move on.

### 2.4 Per-camera tracking (ByteTrack + jersey claims)
```bash
for tag in <chunks>; do
  .venv/bin/python scripts/hybrid_track.py --game <gid8> --tag $tag \
      --out-dir runs/events_fg_hyb<pfx>_$tag --dual-numbers 6
done
```

### 2.5 Cross-camera identity solve (the FUSION step)
One run per roster player per chunk; `--pose --carry` is the production
configuration (ankle-projected court positions + carry between reads):
```bash
.venv/bin/python scripts/solve_player_xcam.py --game <gid8> --tag <tag> \
    --sam3-dir runs/events_fg_hyb<pfx>_<tag> \
    --dump-dir runs/events_fg_<pfx>_<tag> --only "#11" --pose --carry
```
Loop players × chunks (4 parallel workers is fine; ~45s each). Player
tokens: every roster number, with duals split as `#6B` / `#6W`.
**Output = the fused per-player per-camera dumps** the rest consumes.

### 2.6 Ball trajectories + the holder machine
```bash
.venv/bin/python scripts/ball_traj.py --game <gid8>
.venv/bin/python scripts/ballfirst_who.py --game <gid8>   # writes holders_<gid8>.json
```
`ballfirst_who` is the possession state machine (BALL_FIRST_SPEC.md):
per-frame cross-camera votes (ball within 0.7 box-widths), FLIGHT when
the smoothed ball moves >14 px/frame with no candidate, switch hysteresis
6 frames (3 right after a flight), landing-point assignment on catches.

### 2.7 Events (optional for possession graphics, required for scoring)
```bash
.venv/bin/python scripts/detect_shots.py --game <gid8> \
    --plays data/plays/<gid8>_full.json \
    --tracks-glob "runs/events_fg_<pfx>_{tag}" --no-team-prior
# clock shift: median(arc_time - play_time) over nearest pairs -> write
# data/plays/<gid8>_full.json with the shift; build arc-anchored windows
# [arc_t-2.5, arc_t+1.5]; then:
.venv/bin/python scripts/shotdet_p1_adapter.py --gid8 <gid8> \
    --gt runs/shotdet_ab/arc_windows_<pfx>.json --out /tmp/p1tracks_<pfx>
.venv/bin/python scripts/shotdet_transfer_eval.py \
    --tracks /tmp/p1tracks_<pfx>/<full-id>.parquet --label ours_arcwin_<pfx>
.venv/bin/python scripts/assemble_events.py --game <gid8> \
    --plays data/plays/<gid8>_full.json \
    --p3-eval runs/shotdet_ab/eval_ours_arcwin_<pfx>.json \
    --arc-windows runs/shotdet_ab/arc_windows_<pfx>.json
.venv/bin/python scripts/detect_possession_events.py --game <gid8> \
    --plays data/plays/<gid8>_full.json
```

### 2.8 Render
Low-res 4-angle sources first (in-region transcode — local S3 streaming
is throttled to uselessness, don't try):
```bash
.venv/bin/python scripts/aws_transcode_lowres_job.py --games <gid8> --i-rotated-creds
.venv/bin/python scripts/aws_transcode_lowres_job.py --fetch --games <gid8>
.venv/bin/python scripts/render_fullgame_grid.py --game <gid8> \
    --out runs/event_demo/fullgame_<gid8>.mp4
```
Verify before trusting: extract a frame (`ffmpeg -ss <t> -frames:v 1`)
and LOOK at it. Boxes on players, names readable, holder highlighted in
all angles, ball dot present.

---

## 3. Traps that have each cost real hours

1. **`imgsz=1280` on every YOLO predict** — default 640 makes the ball invisible.
2. **Offsets sign**: `camera_frame = reference_frame + GAME_OFFS[game][ang]`.
   Track-dump frame keys and ball-cache keys are CAMERA-LOCAL.
3. **kit_team inversion**: if a team's light kit is team 1, the roster
   needs `"kit_team": {"B": 2, "W": 1}` or every dual-number player gets
   the wrong name (cost e6 11 WHO points before it was found).
4. **Partial MP4s**: a killed ffmpeg leaves a file with no moov atom that
   cv2 reads as black frames, silently. `ffprobe` duration ≥ expected
   before consuming any video you didn't just verify.
5. **Don't casually rerun `detect_shots`** on a game with a good ledger —
   it overwrites `runs/tracking/ledger/shots_<game>_full.json` in place.
6. Tail-chunk naming must match the prep job (`3000_144` vs `3000_145`
   class of bug) — align `GAME_CHUNKS` with what the job actually wrote.
7. Anchor `"frame"` fields are baked with the offsets the prep was given
   (zeros for new games) — hybrid must consume them with the SAME bake;
   real offsets belong to the solve/consumer layer.
8. Ball is detected in only ~50-70% of frames per camera. Any per-frame
   "nearest player" logic will flicker; always go through the holder
   machine (or its cache).

## 4. Rules
- Never commit anything under `data/`, `runs/`, or any credential.
- AWS: ask first; export `UBALL_AWS_CREDS_ROTATED=1`; budget cap $5/run
  without explicit approval.
- Blind-game GT is scored ONCE — never tune thresholds against blind games.
- No Supabase writes without sign-off.
- Ultralytics is AGPL-or-commercial — prototype freely, clear before shipping.
