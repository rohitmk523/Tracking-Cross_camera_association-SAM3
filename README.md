# Cross-Camera Basketball Tracking + Spatial World Model

Production-grade, **general-purpose** multi-camera basketball tracking: 4 fixed cameras →
per-camera detection/tracking/pose → **cross-camera fusion into one persistent global
identity per player** on a top-down court → a structured **spatial world model** → a
**VLM/LLM narration layer** that turns world state into descriptive play-by-play
("dribbled through two defenders, pulled a fadeaway, got blocked").

This repo supersedes the overfit single-clip demo in `DEMO_UBALL/`. The goal here is a
system that generalizes across games and courts (same 4-camera rig).

## Documentation

Full architecture and plan live in [`docs/`](docs/):

| Doc | Contents |
|---|---|
| [00_overview](docs/00_overview.md) | Vision, goals, scope (v1 vs follow-on) |
| [01_architecture](docs/01_architecture.md) | System diagram, layers, data flow |
| [02_tech_stack](docs/02_tech_stack.md) | Every component + license + rationale |
| [03_cameras_and_calibration](docs/03_cameras_and_calibration.md) | 4-cam rig, geometry, per-court calibration |
| [04_detection](docs/04_detection.md) | RF-DETR, SAHI, small/endline players |
| [05_tracking_and_masks](docs/05_tracking_and_masks.md) | ByteTrack, Cutie, occlusion |
| [06_cross_camera_fusion](docs/06_cross_camera_fusion.md) | Homography + ReID + jersey + global ID |
| [07_pose](docs/07_pose.md) | 2D skeletons now, 3D triangulation later |
| [08_world_model](docs/08_world_model.md) | Spatial-state schema + events (JSON) |
| [09_vlm_narration](docs/09_vlm_narration.md) | State-reasoner + frame-clarifier |
| [10_data_and_licensing](docs/10_data_and_licensing.md) | Datasets, annotation, license hygiene |
| [11_edge_deployment](docs/11_edge_deployment.md) | Jetson, TensorRT, hybrid edge+cloud |
| [12_roadmap](docs/12_roadmap.md) | ~3-week aggressive build plan |
| [13_evaluation](docs/13_evaluation.md) | Metrics + acceptance criteria |
| [14_games_and_clips](docs/14_games_and_clips.md) | Which games/clips to pull (S3 + Supabase) |

## Status
Pre-implementation. Docs/architecture locked; build begins per [12_roadmap](docs/12_roadmap.md).

## License
Commercial product — **ship-clean dependency policy** (permissive only). See
[02_tech_stack](docs/02_tech_stack.md) and [10_data_and_licensing](docs/10_data_and_licensing.md).

## ⚠️ Credentials
AWS is used for GPU training only; credentials are **never committed** (see `.gitignore`,
`.env.example`, and [11_edge_deployment](docs/11_edge_deployment.md)). The AWS keys used
in prior projects are flagged for **rotation** — rotate before use here.
