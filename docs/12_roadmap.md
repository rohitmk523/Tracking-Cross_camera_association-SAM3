# 12 · Roadmap (~3-week aggressive, LLM-accelerated)

Scope assumption: heavy LLM-assisted development + **reuse of existing annotated data and the
jersey-ResNet** lets us compress to ~3 weeks for **v1**. Genuinely time-bound items (GPU
training turnaround, single-annotator throughput, calibration, validation iterations) are
flagged — they don't speed up with code-gen.

**v1 = production tracking + cross-camera global IDs + world model + VLM batch narration.**
Follow-on = 3D pose, live/streaming narration, full Jetson real-time.

## Week 1 — Foundation, detection, per-camera tracking
- **D1–2 Foundation**: repo/docs (done) · uv env · AWS wired (gitignored creds, **rotate
  first**) · consolidate `Training_frameworks` annotations → one RF-DETR/COCO dataset
  (cross-court splits) · **eval harness** (detection mAP incl. endline band; tracking
  IDF1/MOTA; cross-cam ID-consistency; narration rubric).
- **D3–4 Detection**: train **RF-DETR-S @1280** on AWS (reuse data) + **SAHI** sliced
  fine-tune; endline-ROI tiling. A/B vs YOLO baseline (internal). *Gate: GPU run time.*
- **D5 Per-camera tracking**: ByteTrack/BoT-SORT via supervision + team (SigLIP) + jersey
  (reuse ResNet) + Torchreid embeddings.
- **Milestone 1**: every player detected (incl. endline) + stable per-camera tracklets with
  team/number/embedding, on a held-out court.

## Week 2 — Cross-camera fusion (core), pose, world model
- **D6–9 Fusion**: tracklab scaffold · homography→court · zone gate · identity-gated
  **Hungarian** (court dist + ReID + jersey + team) · **visibility-weighted global Kalman** ·
  re-entry · jersey-as-authority · cross-view consensus for ID changes. *The hard part — most
  of the build.*
- **D10 Pose + world model**: **RTMPose 2D** per tracked crop (ankle→foot-point feedback) ·
  world-model schema + event detection (possession, shot/make-miss, screen, block, turnover) ·
  per-frame store + event log.
- **Milestone 2**: one persistent global ID per player across all 4 cams through occlusion +
  a populated world-model/event stream on a full recorded game.

## Week 3 — VLM narration, calibration onboarding, hardening
- **D11–13 Narration (batch)**: LangGraph agent · LLM state-reasoner over event stream
  (per-entity decomposition) · **Gemini native-video frame-clarifier** gated on
  shot/contact/occlusion events · NVIDIA VSS pattern.
- **D14 Calibration onboarding**: assisted court-line tool + court-keypoint model for fast
  per-court setup.
- **D15 Hardening + demo**: configs, eval dashboards, run end-to-end on a new court, write
  results into docs. **Edge TensorRT export = start here, finish in follow-on.**
- **Milestone 3 (v1)**: descriptive narration on a recorded game from a previously-unseen
  court, end-to-end.

## Follow-on (post-v1)
- 3D triangulated pose; live/streaming narration (hard sync + when-to-speak gating); full
  Jetson real-time (TensorRT/INT8 budget across all modules); fine-tuned small narration
  model; auto-calibration polish; expand annotated data for accuracy.

## Critical path & risks
- **Cross-camera fusion (Week 2)** is the dominant risk/effort — protect it.
- **Hard camera sync** gates anything live (not v1, but plan now).
- **Annotation throughput** (single annotator) gates accuracy expansion — pre-labeling is
  essential; reuse existing data for v1 to avoid a cold start.
- **GPU turnaround** is wall-clock, not code-speed — parallelize training runs.
