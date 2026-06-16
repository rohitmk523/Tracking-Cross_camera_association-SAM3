# 11 · Edge Deployment & Infra (Hybrid)

## Topology (decided: hybrid)
- **Edge (Jetson AGX Orin, per court)**: ingest/sync → detect → track → pose → cross-camera
  fusion → world model. Emits compact world-state/event JSON.
- **Cloud**: VLM narration over the JSON; pulls frames on demand (event-gated).
Raw video stays local; only structured state (+ occasional frames) leaves the edge.

## Edge runtime
- **JetPack 6.x**, CUDA + **TensorRT** (FP16/INT8) for the per-frame GPU models
  (RF-DETR, RTMPose, Cutie). Export ONNX → TensorRT **on the device** (engines are
  arch/JetPack-specific, not portable).
- **Processes, not threads** (Python GIL): one process per camera for detect+track+pose, one
  fusion process consuming the 4. Frames via shared memory; metadata via ZMQ on-device.
- **Single shared GPU budget** is the real constraint — all per-frame models contend. Levers:
  consolidate where possible (e.g. one-stage pose), run pose at 10–15 fps + interpolate, INT8,
  reserve heavy VOS (Cutie/SAM2) for occlusion-only.
- v1 target: near-real-time tracking on edge for *recorded* games; full 30 fps live is
  follow-on (see hard-sync below).

## Hard camera sync (prerequisite for live)
Cross-camera fusion needs the 4 cameras frame-synchronized. For v1 (batch) we align by
timestamp/offset. **Live requires a genuine synchronized-capture path** (hardware
trigger / Z-cam-style ingest). This is the gating item for the live milestone and is called
out as real work, not a config flag.

## Cloud (narration)
- VLM APIs (Gemini native-video clarifier + LLM state-reasoner) OR on-prem **vLLM** serving
  Qwen3 / Qwen3-VL if data residency requires.
- **LangGraph** agent service consumes the event log + `get_frames` tool.

## AWS (GPU training) — credentials policy
Training runs on AWS GPU (RF-DETR/ReID/pose need CUDA; not Apple MPS). **Credentials are never
committed.**
- Real creds live in **`~/.aws/credentials`** (machine-global, used by the AWS CLI) and/or a
  **gitignored** `.env` / `.aws_local/` in this repo (both in `.gitignore`).
- `.env.example` documents the required variables without secrets:
  ```
  AWS_PROFILE=uball
  AWS_DEFAULT_REGION=us-east-1
  UBALL_S3_BUCKET=uball-videos-production
  ```
- Training scripts read from env/`~/.aws`; **no key material in any tracked file.**
- ⚠️ **Rotation**: the AWS keys used in prior projects were flagged for rotation. **Rotate
  before using them here**, then put the new keys only in `~/.aws` or the gitignored `.env`.

## Repo layout
```
configs/   # hydra/yaml: courts/<id>/ (homographies, zones), model + pipeline configs
src/       # detection, tracking, fusion, pose, worldmodel, narration, calibration
scripts/   # aws_train, bundle/pull, calibrate, run_pipeline, eval
data/      # gitignored (datasets, segcache)
runs/      # gitignored (weights, logs, experiments)
docs/      # this documentation
```
