# GAME PLAN — Possession Only

*The reset, 2026-08-01. One mission, one metric, one deliverable.*

**Mission:** track the ball and the players across all 4 cameras, know **WHO
has the ball at every frame**, and show it — the glowing ring under the
holder, correct in every angle. Everything else (shot detection, make/miss,
zones, event assembly) is REMOVED from the working pipeline. It lives on in
git history and `~/.Trash/uball_cleanup_2026-08-01/` (manifest inside) if we
ever want it back.

---

## The pipeline (all that remains — 8 stages)

```
4 angle videos
  GPU  1. build_dets_cache_yolo.py    players/refs  (yolo26s, imgsz 1280)
  GPU  2. build_ball_cache.py         ball + hoop   (specialist, conf 0.15)
  GPU  3. extract_pose.py             ankles for court projection
  GPU  4. extract_jersey_anchors.py   jersey reads = identity ground wire
  CPU  5. annotate_anchor_kits.py     kit split for dual numbers
  CPU  6. hybrid_track.py             per-camera ByteTrack + jersey claims
  CPU  7. solve_player_xcam.py        cross-camera fusion (--pose --carry)
  CPU  8. ball_traj.py -> ballfirst_who.py -> holders_<game>.json   ← THE PRODUCT
       render_fullgame_grid.py --ring     the ring, all 4 angles
```

Support: `game_meta.py` (offsets/chunks), `sync_anchor_sweep.py` +
`sync_audio_offsets.py` (camera sync), `annotate_anchor_kits.py`,
`refit_calibration.py`, AWS launchers (`aws_fullgame_prep_job`,
`aws_ballcache_job`, `aws_audiograb_job`, `aws_transcode_lowres_job`),
training launchers (`aws_ball_train_job`, `aws_yolo_train_job`).

**Processed games on disk** (caches + holder ledgers ready): `e6fba750`,
`c2a354fe`, `2c490f1a`, `13e1ffad`.

---

## The plays, in order

### P1 — Merge the ring (the deliverable exists)
PR `ring_under_v2` works and rides the fused holder identity correctly.
Before merge: commit `Circle.mov` into the repo (kill the absolute path),
move `RING_README.md` → `docs/`, sanity-run the base render without
`--ring`. Then merge into the cleaned main.

### P2 — Speed: 30h → ~2h → <1h  (the intern's blocker)
The 30h-on-g4 quote is our workload on the wrong GPU, serially.
1. **Now, no code:** g5 (A10G) not g4 (T4), 3 instances in parallel via
   `aws_fullgame_prep_job.py` → **~2.2h wall, ~$9/game** (measured).
2. **One unified 4-class detector** (player/ref/ball/hoop, single model) —
   we currently run TWO detectors over every frame. Halves GPU detection
   time AND finishes the ball specialist's interrupted training (v1 stopped
   at epoch 21/120; recall 0.72 is our ball-coverage ceiling). One ~$10 run,
   A/B vs v1 on ball coverage + holder stability before adopting.
3. **TensorRT FP16 + batched inference** in the cache builders → 2-3× more;
   same work doubles as the Jetson port groundwork.
Target: **under 1 GPU-hour per game** on A10G.

### P3 — Measure WHO directly (holder ground truth)  [IN FLIGHT: docs/POSSESSION_GT_BRIEF.md]
Shot-WHO was our proxy metric; it left with the shot pipeline. The real
metric now: **holder accuracy per frame**. Build the annotation loop —
review video with our prediction preloaded, click to correct the ring when
it's wrong (~1-2h per game). One annotated game = ~90k supervised frames
(vs 134 events before): enough to tune the holder machine's five knobs
honestly, and eventually to train a learned holder classifier.
Baseline to beat, from the shot-era numbers: holder-at-release correct
66/62/38/34% across the four games; track-exists oracle says 80-89% is
there to be picked.
**Scale agreed 2026-08-02: ~30 min total — c2a 10 (tune), e6 10 (tune),
2c4 10 (HELD OUT). 5x2-min windows per game beat one contiguous 10.
Phase 1 = the benchmarked c2a 600-780s window. Brief + tool spec for the
annotator: docs/POSSESSION_GT_BRIEF.md. Bar for "proper tracking":
>=85% holder accuracy on the held-out game.**

### P4 — Sync + calibration hygiene (fusion's two silent killers)
- **c2a re-sync re-solve**: FR measured ~8 frames off its recorded offset
  (two independent instruments). Free, local; directly improves c2a's
  holder votes.
- **Calibration per footage era**: 13e1 (Jan) vs the Mar calib = 5.4m
  cross-camera floor; the intern's 10-2 clips hit the same wall (460cm).
  Any new/re-placed rig needs `refit_calibration.py` (or the intern's
  click-to-calibrate tool from the `xring` side repo — absorb it) BEFORE
  fusion. This is the #1 onboarding risk for new footage.

### P5 — New footage intake (what the intern runs)
Short clips: fully local per the runbook (his GPU, minutes, $0).
Full games: the 3-instance AWS pattern until P2.3 lands local speed.
Docs: `docs/GUIDE_POSSESSION_PIPELINE.md` (stages 7-9 there are now
optional/removed).

---

## What "done" looks like this month
1. Ring merged; one **new** game processed end-to-end by the intern without
   help, ring correct in all 4 angles.
2. A game processes in **≤2h wall / ≤$10**, path to <1h visible.
3. Holder accuracy **measured directly** on one annotated game, with the
   confusion anatomy (defender-swap vs pass-smear vs gap) driving the next
   holder-machine change.

## Ground rules (unchanged)
Never commit `data/`/`runs/`/credentials · AWS spends need approval
(`UBALL_AWS_CREDS_ROTATED=1`, ~$5 cap without sign-off) · blind GT is
scored once · Ultralytics AGPL cleared before anything ships ·
deleted ≠ destroyed: git history + the Trash folder hold everything.
