# Ring Overlay — Working Notes & Handoff

Everything we've figured out about drawing the glowing **ball-possession ring**
on the 4-angle fusion output, plus how to run it and how to process new footage.
This is the memory doc — read it first when picking this back up.

Last updated: 2026-07-22.

---

## TL;DR — where things stand

- **The circle graphic is DONE and works on the real `e6fba750` fusion output.**
  It draws a glowing `Circle.mov` ground ring under the ball holder in **all four
  angles at once**, driven by the cross-camera fused identity (so it's the same
  physical player in every view). Runs **locally, no GPU, no AWS**.
- It's built **on top of** `scripts/render_fullgame_grid.py` (not from scratch),
  via a new `--ring` flag.
- To process **new footage** (e.g. a 20-second 4-angle clip): **your local GPU is
  enough — AWS is NOT required.** The only blocker is having **calibration that
  matches the footage** (+ optionally a roster for names).

---

## What was built (files in THIS repo)

| File | What it is |
|---|---|
| `scripts/ring_overlay.py` | **NEW.** Self-contained ring renderer: decodes `Circle.mov` → alpha cache, One-Euro stabiliser with vertical ground-lock (ring stays planted when the holder jumps), box-anchored ground ellipse, and box-level occlusion (ring tucks behind players in front). |
| `scripts/render_fullgame_grid.py` | **PATCHED.** Added `--ring` (draw the ring), `--ring-mov PATH` (default: `/home/akhilesh/Desktop/Uball/desgin/Circle.mov`), and `--keep-holder-box` (also keep the old thick yellow box). The ring replaces the yellow holder box; name labels stay. |

`Circle.mov` currently lives in the `desgin/` repo and is referenced by absolute
path. (A copy could be dropped into this repo if desgin ever moves.)

---

## How the ring attaches to the right player (the important part)

We do **NOT** use "nearest player to the ball" per frame — that flickers and jumps
during passes because the ball is only detected in 50–70% of frames.

Instead the ring rides the **fused holder identity** from
`runs/tracking/ledger/holders_e6fba750.json` (frame ranges → stream id like
`n11`, at 29.97 fps). That ledger is the output of the ball-first possession state
machine, which already applied all the smoothing (6-frame switch hysteresis,
hold-through-dropout, flight logic). So:

- **holder identity** comes from `holders_<game>.json` (already smoothed).
- **per-camera box** for that identity comes from the `runs/events_fg_*` dumps:
  `runs/events_fg_<tag>/<game>_<tag>__<sid>__<ANG>.json` → `{frames: {<cam_frame>: {box:[x1,y1,x2,y2], present:true}}}` in 1920×1080 px.
- **camera-offset convention** (a documented trap): `cam_frame = ref_frame + GAME_OFFS[game][ANG]`. `render_fullgame_grid.py` already handles this (`target = f + OFFS[ang]`). e6 offsets: `FL 0, FR -11, NL -1, NR -1`.

Because the holder id is one global identity, looking up its box in each camera
puts the ring on the same person in all four views automatically. That lookup
**is** the cross-camera association we're riding on.

---

## How to run it (e6fba750 — local, no GPU)

**1. Get the artifacts in place** (once). The handoff zip populates gitignored
`runs/` + `data/`:
```bash
# handoff_e6_artifacts.zip contains data/ and runs/ at top level
cp -a /home/akhilesh/Downloads/handoff_e6_artifacts/runs/. runs/
cp -a /home/akhilesh/Downloads/handoff_e6_artifacts/data/. data/
```
Provides: `holders_e6fba750.json`, `events_v2_*`, `possession_events_*`, the
`events_fg_*` per-camera dumps, `ball_cache/`, `ball_traj/`, the 4 `fullsrc_e6fba750_<ANG>.mp4`
low-res videos, rosters, plays.

**2. Baseline render (guide fast path):**
```bash
PY=/home/akhilesh/Desktop/Uball/my/bin/python   # torch+CUDA venv; cv2/ffmpeg
$PY scripts/render_fullgame_grid.py --game e6fba750 --out runs/event_demo/test.mp4 --t0 600 --dur 60
```

**3. With the ring (the circle graphic):**
```bash
$PY scripts/render_fullgame_grid.py --game e6fba750 --out runs/event_demo/ring.mp4 --t0 600 --dur 60 --ring
# drop --t0/--dur for the full game (~30 min render)
```
Verified: ring lands on `#1 Veritas` (the holder) in FL/FR/NL/NR simultaneously.

Env: use the `my` venv at `/home/akhilesh/Desktop/Uball/my/bin/python` (Python
3.14, torch 2.12+cu130, cv2 5.0, ultralytics, av, onnxruntime 1.27). The system
python 3.14 has none of these.

---

## Processing NEW footage (e.g. a ~20s 4-angle clip) — LOCAL, no AWS

**Key finding: for short clips you do NOT need AWS.** Every GPU stage reads
**local** clip files `data/clips/<game>_<ANG>_<tag>.mp4` and takes `--device cuda`.
The `scripts/aws_*` jobs are just wrappers that slice those same clips from S3 and
run the identical scripts on EC2 — worth it for a 55-min game (~6 GPU-hrs),
pointless for ~600 frames. Your GPU does a 20s clip in minutes, for $0.

Pipeline (all local):
```
data/clips/<gid>_<ANG>_<tag>.mp4   (put the 4 clips here; tag e.g. "0_20")
  GPU  build_dets_cache_yolo.py  --game <gid> --tag <tag> --weights <yolo26s_player_referee_ball.pt> --device cuda   (imgsz 1280!)
  GPU  build_ball_cache.py       --game <gid> --tag <tag> --weights <yolo26s_ball_hoop_specialist.pt> --device cuda
  GPU  extract_pose.py           --game <gid> --tag <tag> --device cuda           (rtmlib ONNX)
  GPU  extract_jersey_anchors.py --game <gid> --tag <tag>                          (auto-measures sync by audio)
  CPU  hybrid_track.py -> solve_player_xcam.py (--pose --carry) -> ball_traj.py -> ballfirst_who.py
  CPU  render_fullgame_grid.py --ring
```
Only new dependency: `pip install rtmlib` (pose). `ultralytics`, `onnxruntime`,
`av` already present in the `my` venv. onnxruntime here is CPU-only providers;
fine for 20s, or `pip install onnxruntime-gpu` for speed.

### Trained weights (from handoff_weights_yolo26s.zip, in ~/Downloads)
- `yolo26s_player_referee_ball.pt` — 0=player, 1=referee, 2=ball. Main detector.
- `yolo26s_ball_hoop_specialist.pt` — 0=Basketball, 1=Basketball Hoop. **Use this
  for the ball/rim; far better than the main model's ball class.**
- `yolo11n_jersey_number_localizer.pt` — jersey number detection (optional).
- **imgsz=1280 on EVERY predict** (trained at 1280 on 1080p; default 640 makes the
  ~6px ball vanish). Ball `conf=0.15`, players `conf=0.25`. Keep only best ball/frame.

### What's still needed to process a new clip
1. **The 4 clips** (~20s, FL/FR/NL/NR).
2. **Calibration matching the footage — the one real risk.** Cross-camera fusion
   projects players to court cm via per-camera homographies (`configs/calib/<ANG>.json`).
   - Same fixed court-a rig as e6, cameras unmoved → reuse e6 calib, works.
   - Fresh camera setup → calib won't match → fusion breaks (see 10-2 below). Must
     re-calibrate first.
3. **Roster** `data/rosters/<gid>.json` (jersey #→name per team) — optional; without
   it boxes show `#num` not names.
4. **game_meta.py**: add `GAME_OFFS[<gid>]` and `GAME_CHUNKS[<gid>]` (one tag for 20s).
   Sync offsets are auto-measured from audio by `extract_jersey_anchors.py`.
5. Sync offsets: auto — no need to provide.

### If AWS is still preferred
Possible but heavier: upload 4 clips to S3 at `court-a/<date>/<gid>/...`, run
`aws_fullgame_prep_job.py` + `aws_ballcache_job.py` (~$18–20, gated behind
`UBALL_AWS_CREDS_ROTATED=1` / `--i-rotated-creds`), `--fetch`, then the local
stages. Put creds in `~/.aws/credentials` (`aws configure --profile uball`) —
never paste secrets in chat, never edit `.env` by hand (user manages secrets).

---

## Calibration mismatch — the "10-2" clips (cautionary tale)

The `desgin/` folder's `10-2 *_5min.mp4` clips are a DIFFERENT game. The fusion
repo's `configs/calib/*.json` were built for e6/etc. and **do NOT match 10-2**:
- same player projected from two cameras lands **~460 cm apart** (should be <50 cm),
- the projected court boundary floats over players' waists, not on the floor lines.
This is the documented "median pair distance ≫ 100 cm ⇒ stale calibration" trap.
Lesson: **any new footage from a re-placed rig needs fresh calibration** before
cross-camera fusion will work.

---

## The `xring` side-repo (alternative approach — parked)

`/home/akhilesh/Desktop/Uball/xring` — a self-contained attempt to build the ring
+ cross-camera fusion from scratch on raw videos (geometric association, no
jersey/AWS). Fully working: audio sync, per-camera ByteTrack, world model,
possession, grid render. **Parked** because it needs correct calibration for the
10-2 footage. It includes `calibrate.py` — an assisted click-to-calibrate tool
(click ~8-13 court landmarks per camera → fits homography + lens term → writes
`calib_local/<CAM>.json`). Use this if we ever need to calibrate a new rig for
either repo. Fitting math validated (median reproj ~2 cm on synthetic clicks).

---

## Ground rules (from the handoff)

- Never commit anything under `data/` or `runs/`, or any credential.
- Do **not** run `scripts/aws_*` jobs without explicit approval (they cost money;
  ~$18-20/game; gated behind `UBALL_AWS_CREDS_ROTATED=1`).
- Do **not** rerun `detect_shots.py` on a processed game (overwrites scored ledgers).
- Blind-game GT is scored ONCE — never tune against it.
- Ultralytics = AGPL/commercial — prototype freely, clear the license before shipping.
- Ask about offsets/AWS rather than guessing.

---

## Open questions / next actions

- [ ] New clip: is it the **same court-a rig as e6** (reuse calib) or a **fresh
      setup** (re-calibrate first)? — blocks fusion quality.
- [ ] Provide the 4 clips + (optional) roster for the new game.
- [ ] `pip install rtmlib` before the pose stage.
- [ ] Optional ring polish: sine-alpha pulse + Gaussian-blur glow, or swap to
      `sv.EllipseAnnotator`; match the reference video's exact ring style.
- [ ] Full-game e6 render (drop `--t0/--dur`).
