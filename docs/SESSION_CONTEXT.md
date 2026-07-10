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

## 8. Backlog (ordered)

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
