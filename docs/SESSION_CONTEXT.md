# Session context — SAM3-free pipeline era (updated 2026-07-11)

Dense handoff for continuing work. Client-facing narrative lives in `docs/STATUS.md`
(read it for the story); THIS file is the operational truth: numbers, paths, knobs,
in-flight jobs, and traps already paid for.

## 1. Where the project stands

SAM3 is **retired** (user directive: cost/time — 0.65fps ≈ $25+/game AWS; ~500×
real-time deficit on the Jetson AGX venue box, which is the deployment target).
The production pipeline is SAM3-free:

```
RF-DETR detect (every frame, cached)            scripts/track.py → runs/dets_cache/
→ ByteTrack motion tracking                     scripts/hybrid_track.py → runs/hybrid_*/
→ jersey OCR checkpoints (every frame, cached)  scripts/extract_jersey_anchors.py → runs/anchors/
→ kit-shade tags for roster-dual numbers        scripts/annotate_anchor_kits.py (in-place)
→ RTMPose skeletons (cached)                    scripts/extract_pose.py → runs/pose_cache/
→ cross-camera correction + KPR ties + reacq    scripts/solve_player_xcam.py / kpr_assist_xcam.py
→ demo render                                   scripts/render_demo_all.py, render_sam3_player.py
```

## 2. Measured results (all strict all-angles vs human GT unless noted)

| Milestone | Score |
|---|---|
| e6 (tuned game) hybrid baseline | 77.6% |
| + zero-shot KPR ties +1, coverage +2, reacq +1.7 | 82.3% |
| + fine-tuned KPR (flywheel cycle 1, trained on c2a auto-labels only) | **83.2%** (#11 92 / #6 85 / #22 82 / #43 74) |
| SAM3 reference (retired) | 88% |
| c2a BLIND validation (kit-aware) | **81%** (#3W 86 / #5B 83 / #3B 73) |
| f66 FULLY-blind demo (no GT exists) | visual only — 11 identities, delivered |
| KPR pile-up pilot: zero-shot → fine-tuned | 69% → **75%** same-kit (chance 33%) |
| Ankle sole07 projection | cross-cam agreement 63→48cm (−23%) |

REFUTED (do not retry; see ledger + commits): joint mutual-exclusion assignment (3
variants: 26%/48-55%/65.5% vs xcam 78%), gate tightening (78→70 monotone), segment-
level KPR voting (no gain), GSI short-gap fill (no gain — remaining no-picks are
LONG absences → solved by KPR re-acquisition instead).

Error decomposition insight that drove everything: at 78%, 60% of misses were
NO-PICK (nothing shown) vs 40% wrong-identity → coverage + reacq first, then KPR.

## 3. Per-game constants (critical)

| game_tag | OFFS (frames, vs FL) | notes |
|---|---|---|
| e6fba750_44_60 / _44_180 | FR −11, NL −1, NR −1 | tuned game; 4 GT players |
| c2a354fe_300_60 | **FR +1, NL +2, NR −1** | audio-synced, GT-validated (77cm agreement). The old −4/−3/−4 was WRONG (450cm) and caused every early c2a failure |
| f66eb3b2_60_300 | FR −5, NL +3, NR 0 | audio-synced on instance; blind game |

`solve_player_xcam.py` + `hybrid_track.py` fall back to the anchors-file offsets for
unknown games (audio-synced at extraction) — trust that path; it's validated.

Rosters: e6 has two #22s/#11s/#6s contexts (kit-aware TODO for e6 — likely #22 gain).
c2a duals = 1,3,5 (B/W kits). f66 (White vs Black) duals = 2,3,7,8; #8 kit-split
declined (41 reads). Roster source: Supabase `games.roster_team1/2` (+team1_color/2),
project mhbrsftxvxxtfgbajrlc, plays table has `timestamp_seconds` per game.

## 4. KPR (appearance) infrastructure

- Working checkout: **/tmp/kpr — EPHEMERAL** (reboot wipes). Rebuild instructions:
  `third_party/kpr_integration/SETUP.md` (venv py3.10, patches, weight sources).
- Weights: original `kpr_occ_pt_IN_82.34_92.33_42323828.pth.tar`; fine-tuned
  `kpr_uball_ft.pth.tar` (from `runs/kpr_ft/logs/700042914/...` job-700042914_12).
- Loading: restricted unpickler (audited 9-globals allowlist, incl BOTH
  numpy.core/_core paths) — see kpr_pilot.py. User explicitly approved the weights.
- Fine-tune dataset builder: `scripts/build_kpr_dataset.py` (3,061 c2a crops, kit
  split only when clusters kit-opposed; labels audited 89-100% vs GT).
- Trainer: `third_party/kpr_integration/{kpr_uball_ft.yaml, train_uball.py}` +
  `scripts/aws_kpr_train_job.py` (~$2.5/cycle). TRAP: PK sampler needs
  batch_size/4 ≤ #identities (bs=48 with 8 ids silently trained 0 batches — verify
  loader length AND diff weight tensors after any "successful" training).
- kpr_assist_xcam.py env knobs: KPR_CFG (test yaml), KPR_TAG (emb cache suffix),
  SAM3_DIR, GATE_CM, MAX_INTERP=300, SOFT_GATE=300, REACQ_GAP=240, DUMP_DIR.
  Operating point: MAX_INTERP=300 SOFT_GATE=300 + fine-tuned KPR = 83.2%.

## 5. In-flight RIGHT NOW (2026-07-11 ~01:00)

- **yolo11s training**: i-0c2182567975d5860, healthy (epoch 2+, 642 it/ep, ~3h).
- **yolo11m training**: i-052852936a63ce1d8, batch=8 explicit (auto-batch trap: -1
  chose bs2 = 7h pace). 401 it/ep @ ~1.9it/s → may brush the 3.5h failsafe near
  epoch 55-60; incremental best.pt uploads every 10 min make that survivable.
- Watchers: b4rbaqpvm (11s + a dead old 11m id — its first fetch is garbage),
  bhaa5k06y (new 11m). Fetch: `aws_yolo_train_job.py --fetch --model yolo11{s,m}`
  → `runs/yolo11{s,m}-1280-ourdata-v1_fetch/`.
- Next after weights: plan in `Training_frameworks/Uball 4Cam Detection/
  YOLO_TRAINING_PLAN.md` — Level-1 mAP (test split; instance already ran val),
  Level-2 pipeline gate (rebuild e6 dets with YOLO → hybrid+xcam strict score must
  be within 1pt of 83.2%), Level-3 RACE vs RF-DETR-FP16 (one instance, 3 lanes,
  fps + $/game). Decision rule pre-agreed in the plan. YOLO license = AGPL (flag).
- FP16 for RF-DETR is ENABLED (uball_cc/detection/base.py, CUDA-only) — race
  baseline is fast RF-DETR. User wants FP16 kept regardless.

## 6. Cost & ops rules

- HARD RULE: no AWS run >$5 without explicit approval (memory: aws-budget-cap).
  Levers: failsafes in every userdata (2-3.5h), external kill-switch watchers,
  incremental uploads. Account has ZERO G-type spot quota (spot always fails —
  needs a Service Quotas request; don't retry spot until user raises it).
- Creds: export UBALL_AWS_CREDS_ROTATED=1; never --accept-unrotated.
- AWS job traps already paid for: shared S3 log/results keys across relaunches →
  ALWAYS gate on instance state (describe-instances), never on log content;
  DLAMI pip: pin "torch==$TV", uninstall torchaudio (ABI break), apt ffmpeg for
  audio-sync, ultralytics+pytorch-lightning+nltk for the jersey stack; py3.12
  dataclass mutable-default fix in occluded_posetrack21.py; `yolo` CLI not on PATH
  (use Python API); bundle clips must land where REPO-relative code expects
  (symlink data/clips).
- Blind-game prep job: `scripts/aws_blindprep_job.py` (detection+OCR any window,
  ~$0.6-1, --anchors-only mode). Local chain pattern: scratchpad f66_chain.sh
  (fetch → kit-tag → hybrid → coverage-pick → xcam dumps → render).

## 7. Deliverables inventory (runs/tracking/)

demo_allplayers_f66eb3b2.mp4 (fully blind, 5min, 11 ids) · demo_allplayers_c2a354fe.mp4
(blind, 81%) · demo_allplayers_e6fba750.mp4 (tuned) · sam3player_n{11,22,43,6}_sam3free_
e6fba750.mp4 · sam3player_n{11,22,43,6}_final88_e6fba750.mp4 (SAM3 era) ·
pipeline_failure_heatmap.jpg (green center, red edges → bigger-jersey-numbers rec) ·
ledger/master_scorecard.jsonl (every scored experiment).

## 8. THE AIM (user-stated, 2026-07-11): as REAL-TIME as possible, as ACCURATE as possible

**User's detection thesis (adopted):** the failure mode is far-angle small-player
detection; since every player is covered by 2-4 cameras, high PRECISION with
good-enough recall at usable player sizes is the right trade — a far-cam miss is
survivable (redundancy covers it), a false box is not (it steals claims/ties/reacq).
Consequence for judging: the strict all-angles metric over-punishes far-cam misses
relative to the product; the detector winner is judged on BOTH strict AND fused
(≥1-cam) coverage, plus speed.

**Detector race: DONE (2026-07-11). E2E race replaced the e6 gate** — user correctly
called the e6 Level-2 gate contaminated (jersey anchors were RF-DETR-derived and
shared across lanes = "cheating"); the E2E race rebuilds EVERYTHING per lane from
raw video on the c2a GT minute (scripts/e2e_race.py + aws_e2e_race_job.py; grading
via GT_DETS_DIR; kit-copy step required for dual streams — n3→n3B/n3W).

**FINAL TABLE (c2a354fe_300_60, A10G):**
| lane | strict | fused | e2e 1min | det fps | reads |
|---|---|---|---|---|---|
| RF-DETR-S FP16 | .810 | .856 | 1765s | 24.5 | 6191 |
| yolo26s | .789 | .846 | 1568s | 63.4 | 6983 |
| yolo11s | .766 | .844 | 1578s | 67.2 | 6857 |
| yolo11m | .762 | .816 | 1560s | 60.3 | 6756 |

**EVENTS WORKSTREAM (2026-07-12, user-approved >$5 until live) — FIRST MILESTONE.**
Full-game e6 (55.7min): chunked cache prep (6×10min, detect+ball+pose+dense-s1
anchors, scripts/aws_fullgame_prep_job.py + aws_anchors_patch_job.py; traps paid:
bare-`wait` on infinite uploader loop $3.4; offsets single-quote bug ate anchors,
grep swallowed traceback). Court zones: configs/court_zones_court-a.json (3PT
model=white paint ±1cm; 4PT red-paint arc fit L949.5/R939.8cm, scripts/
extract_court_zones.py). Events engine scripts/detect_events.py v1.5 (possession:
ball-player proximity + x-cam vote + hysteresis + STICKY; attribution at GT play
timestamps; REBOUND/STEAL forward windows; zone at logged-ts position). GT: 219
plays data/plays/e6fba750_*.json (per-chunk). **RESULTS: dev(1-4) WHO 50%/zone 61%;
HELD-OUT(5-6) 45%/57%; full game 48%/60%; FTs ~solved.** REFUTED: ball interp at
3-12% recall (2/9→0/9). CEILING = ball detection recall (1,257 train balls) →
fork: ball retrain (+4th-class job) vs phase-2 possession-change rules. Ball cache
builder: scripts/build_ball_cache.py (conf 0.08 barely helps — data problem).
Names on videos: data/rosters/*.json + render_demo_all labels. Events total ~$15.

**PHASE 1+2 COMPLETE (2026-07-11 evening).** Phase 1 (batching, GATED, commit
78c1fd1): jersey_stack.read_crops (batched 3-stage), anchors sequential-decode +
inline kit shades, annotate_anchor_kits fast path — OCR+kits 1230s→82s (15×);
kit clustering 0.12s. Phase 2 verdicts (ALL work-reduction FAILED offline gates,
5 consistent density datapoints): triggered OCR (scripts/extract_jersey_anchors_
triggered.py, state machine) −2.9 strict at 78% crops → LIVE profile only;
stride-2 detection (scripts/make_stride2_dets.py sim) 0.821/0.862 vs 0.831/0.892
→ rejected; pose-subset skipped (untracked picks lose sole07). **OPERATING POINT
(offline): yolo26s + dense STRIDE-1 anchors everywhere — e6 81.5/85.5 (ft-KPR),
c2a 0.831/0.892 = NEW champion (beats RF-DETR lane 0.810/0.856). Laptop ~9.4
min/game-min.** #6 e6 outlier (−8 vs RF-DETR) = wrong-pick in FR(0.633)/NR(0.747),
same cells as #43/#22 → identity discrimination in right-side crowds, not
detection. TRAP: sweep first run scored 0.22 everywhere — forgot GT_DETS_DIR in
solve_player_xcam call (the classic artifact; env var now supported there too).
NEXT: Phase 3 = CoreML/ANE exports (detect+OCR 2-3×, target ~4-5 min/game-min),
4th-class retrain (~$3, ball gate ≥0.855), AGX TensorRT (awaiting engineer
answers to docs/JETSON_STREAMING_QUESTIONS.md). Accuracy: kit-aware e6 #22,
#6/FR-NR fix, flywheel 2.

**DECISION (user, 2026-07-11): yolo26s IS the production detector from now on.**
Optimization roadmap: docs/OPTIMIZATION_PLAN.md (phases: waste removal -> triggered
OCR/stride gates -> CoreML/TensorRT + 4th-class; accuracy track in parallel).
**Verdict:** RF-DETR = accuracy reference (wins both metrics). yolo26s = edge/live
candidate (best YOLO, fused −1.0pt, 2.6× det speed, DeepStream/TensorRT drop-in on
the new AGX streaming box, MOST jersey reads). User to visually confirm
(runs/tracking/demo_allplayers_yolo26s_c2a354fe.mp4 vs demo_allplayers_c2a354fe.mp4).
Stage split (any YOLO lane): OCR 53% + kits 27% + pose 11% + detect 7% → OCR/kits
are the realtime frontier, NOT detection. Laptop (M1): yolo26s 45.7fps measured;
full pipeline ≈30-45 min/game-min unoptimized. Old e6 gate numbers (81.8/80.0)
SUPERSEDED. yolo26s L1 valid: player .960 / ball .855 (−4.2). mAP lesson holds
(11m best mAP, worst pipeline).

**Jetson AGX live streaming (new workstream):** new box runs DeepStream 7.1 + NDI
ingest (uspaces-docker → bauersan/jetson-ndi-yolo image; old GoPro/dual-Nano stack
in gopro-automation-linux is the record→S3→cloud flow). Design = TWO-PASS: live
pass on AGX (TensorRT winner + tracker + TRIGGERED OCR per user's assisted-OCR
idea: new/low-conf/heartbeat-only reads) + finalization pass (full anchor
interpolation, needs future reads → rolling delay or post-game on recording).
Engineer questionnaire: docs/JETSON_STREAMING_QUESTIONS.md (21 Qs; key: NDI vs
GigE path, stream sync/audio, recording co-exists?, roster at stream start,
1 AGX per court?).

**Post-winner agenda (in order):**
1. Adopt winner into all prep jobs + dets caches; re-baseline e6/c2a numbers.
2. REAL-TIME push: detect every 2nd-3rd frame (tracker bridges), batching,
   TensorRT export; target full-game prep < 30 min, then Jetson AGX port.
3. ACCURACY push: kit-aware identity on e6 (#22 same-number opponent — free win);
   #43 far-court cell diagnosis; flywheel cycle 2 (e6+f66 auto-labels into KPR);
   bigger jersey numbers with the venue (highest-leverage physical change).
4. OCR-detection consolidation (user-directed, post-winner):
   - Number-box YOLO labels EXIST: Training_frameworks/"Uball OCR Master" annotates
     player/referee/NUMBER boxes in YOLO format (HF dataset-backed); the current
     localizer (YOLO11n, 0.99 mAP50 on crops) was trained from this line.
   - Options in order: (a) 4TH-CLASS PILOT — add "number" to the winner detector →
     one pass does players+refs+ball+number boxes, deletes the localizer stage
     (caveat: full-frame numbers only resolvable at near/mid sizes — acceptable per
     the precision thesis); (b) retrain localizer in the WINNER's family/size for
     stack consistency — honest note: 0.99 on crops is saturated, size-up buys
     ~nothing; the win is family consistency + more data via the active-learning
     loop, not capacity; (c) batch all three OCR stages + stride-2 reads (validated)
     = the actual real-time multiplier.
5. Ball workstream (detection labels exist; tracking/possession scoped not started).

## 8b. Backlog (ordered)

1. YOLO Level-1/2/3 when weights land (watchers will fire).
2. Update STATUS with f66 fully-blind result + YOLO outcome table.
3. Kit-aware identity on e6 (#22 has same-number opponent) — likely +points, free.
4. #43 FR cell (56%) diagnosis — biggest single defect.
5. Flywheel cycle 2 (add e6 windows + f66 auto-labels to KPR training).
6. USER-side: KPR HL3 license commercial check; YOLO AGPL decision; NL camera
   physical check (left-basket under-coverage); bigger jersey numbers with venue.

## 9. STATUS.md structure (don't reorder — user-specified narrative)

Part 1 foundations → Part 2 SAM3 proven (88%) → Part 3 why SAM3 can't ship
(AGX+AWS) → "the bet" paragraph → Part 4 pipeline explained + comparison + evidence
list → Part 5 blind validation + flywheel + next → Part 6 edge deployment + detector
swap + ball status → Appendices (method / heatmap+venue recs / files). No weekday
names anywhere. Ball: detection labels exist (~1,580 boxes, class 2 already in dets
caches, unconsumed); tracking/possession scoped not started; shots = separate repo.
