# 02 · Tech Stack

## Ship-clean license policy
This is a **commercial product**. Shippable code/weights may only depend on **permissive**
licenses (Apache-2.0, MIT, BSD). **Excluded from shippable build**: AGPL (YOLO11/Ultralytics,
BoxMOT), GPL (sn-gamestate, XMem++), and any non-commercial / NDA datasets (those are
benchmark-only — see [10_data_and_licensing](10_data_and_licensing.md)).

> Two deliberate swaps from the DEMO_UBALL demo, forced by licensing + quality:
> - **Detection: YOLO11 → RF-DETR** (Apache-2.0; also better small-object/few-shot).
> - **Pose: YOLO11-pose → RTMPose/RTMO** (Apache-2.0 via MMPose).

## Component table

| Layer | Choice | License | Why |
|---|---|---|---|
| Detection | **RF-DETR-Small @1280** (`roboflow/rf-detr`) | Apache-2.0 | First real-time >60 mAP COCO; beats YOLO11-XL at ⅓ latency; DINOv2 → strong few-shot; clean license |
| Small-object | **SAHI** (`obss/sahi`) sliced fine-tune + endline-ROI tiling | MIT | +12–15 AP-small; supports RF-DETR ≥1.6.2 |
| Per-cam tracker | **ByteTrack / BoT-SORT** via `supervision` | MIT | Cheap, proven; MIT (avoids AGPL BoxMOT) |
| Masks / occlusion VOS | **Cutie** (`hkchengrex/Cutie`) | MIT | Best multi-object VOS per-FLOP; occlusion-robust; edge-portable |
| Offline quality ceiling | **SAM2.1-L + SAM2Long** | Apache-2.0 (SAM2) / verify SAM2Long | Max J&F on long occluded clips for ground-truth-grade passes |
| Occlusion rescue (1–2 tracks) | **SAMURAI / DAM4SAM** | Apache-2.0 / verify | Single-object motion/distractor-aware re-track during player-on-player occlusion |
| Team classification | `roboflow/sports` **TeamClassifier** (SigLIP→UMAP→KMeans) | MIT | Exact blog method; cleaner on fixed cams |
| Jersey number | **our ResNet** (reuse from DEMO_UBALL; retrain general) | ours | Already strong; improve over time |
| Pose | **RTMPose / RTMO** (MMPose) | Apache-2.0 | Top-down or one-stage; commercial-clean (replaces AGPL YOLO-pose) |
| Re-ID (cross-cam) | **Torchreid** OSNet (`KaiyangZhou/deep-person-reid`) | MIT | Battle-tested appearance embeddings |
| Court projection | `roboflow/sports` **ViewTransformer** + our homographies | MIT | Per-cam → court coords + top-down radar |
| Court keypoints | keypoint model trained on **our** per-court landmark annotations | ours | Per-court assisted calibration (no third-party court data — see [10](10_data_and_licensing.md)) |
| Pipeline scaffold | **tracklab** (`TrackingLaboratory/tracklab`) | MIT | Modular detect/track/reid/calib stages |
| Cross-cam fusion | **our code** (Hungarian + global Kalman) | ours | The core IP; mirrors MV3DT shape, fully open |
| VLM state-reasoner | **LLM API** (Gemini / Claude) or on-prem **Qwen3** | API / Apache-2.0 | Reasons over JSON; vision unnecessary here |
| VLM frame-clarifier | **Gemini (native video)** or on-prem **Qwen3-VL** | API / Apache-2.0 | Only Gemini ingests video natively; Qwen3-VL for on-prem |
| Agent orchestration | **LangGraph** + **vLLM** (if on-prem) | MIT / Apache-2.0 | "narrate-from-state vs fetch-frame" loop; serve open models |
| Edge runtime | **TensorRT / ONNX Runtime** on JetPack | proprietary-but-free | Required for edge real-time; not a license-share concern |
| Reference pattern | **NVIDIA VSS Blueprint** (structure only) | check | CV metadata → VLM narration; our exact pattern, prebuilt |

## Explicitly avoided (and why)
- **Ultralytics YOLO11 / YOLO11-pose** — AGPL-3.0 (commercial copyleft). Used only as an
  internal A/B baseline, never shipped.
- **BoxMOT** — AGPL-3.0. Reference only.
- **sn-gamestate, SportsLabKit** — GPL-3.0. Port *ideas* (PRTReid multi-task re-ID design),
  not code.
- **SAM3-as-tracker** — single per-frame segmenter, no occlusion memory; replaced by
  ByteTrack+Cutie. (SAM3/SAM2 still usable as a mask *seeder* offline.)
- **NVIDIA MV3DT** — proprietary `.so`, can't retrain or inject our jersey-ResNet; we mirror
  its architecture openly instead.

## Languages / infra
- Python 3.11 (uv-managed venv, as in DEMO_UBALL recovery).
- AWS GPU (g5/g6) for training; JetPack 6.x on AGX Orin for edge.
- Configs in `configs/` (YAML/Hydra via tracklab); experiments tracked in `runs/`.
