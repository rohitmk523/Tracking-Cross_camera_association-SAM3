# 01 · Architecture

## Layered pipeline

```
                 ┌──────────────── EDGE (Jetson AGX Orin, per court) ────────────────┐
  4 fixed cams   │                                                                   │
  FL FR NL NR ──►│  A. INGEST/SYNC     4 synced 1080p streams (RTSP/file)            │
                 │       │                                                            │
                 │       ▼  per camera (×4, parallel processes)                       │
                 │  B. DETECT          RF-DETR-S @1280 (+SAHI endline ROI)            │
                 │       │             classes: player, referee, ball                 │
                 │       ▼                                                            │
                 │  C. TRACK           ByteTrack/BoT-SORT → per-cam track ids         │
                 │       │             (+ Cutie masks where occlusion needs it)       │
                 │       ▼                                                            │
                 │  D. ATTRIBUTES      team (SigLIP) · jersey # (ResNet) · pose (RTMPose)│
                 │       │                                                            │
                 │       ▼  all 4 cams →                                              │
                 │  E. CROSS-CAMERA FUSION                                            │
                 │       homography→court · ReID+jersey+team gate · Hungarian ·       │
                 │       visibility-weighted global Kalman → ONE global id/player     │
                 │       │                                                            │
                 │       ▼                                                            │
                 │  F. WORLD MODEL     per-frame/event spatial-state JSON +           │
                 │       │             event detection (shot, make/miss, possession,  │
                 │       │             screen, block, turnover)                       │
                 └───────┼────────────────────────────────────────────────────────────┘
                         │   compact world-state JSON / events  (NOT raw frames)
                         ▼
                 ┌──────────────── CLOUD ────────────────┐
                 │  G. VLM NARRATION (LangGraph agent)    │
                 │     state-reasoner LLM over JSON  ──┐  │
                 │     event-gated frame-clarifier ◄───┘  │  ← pulls frames only on
                 │       (native-video VLM)               │    shots/contact/occlusion
                 │       │                                │
                 │       ▼  descriptive play-by-play      │
                 └────────────────────────────────────────┘
```

## Why this split
- **Edge does the heavy per-pixel work** (detect/track/pose/fuse) and emits a tiny
  structured state. **Cloud does language reasoning** over that state. This is the
  client's "world-model-first" thesis: get most of the benefit of a VLM watching every
  frame, at a fraction of the latency/cost, because reasoning is over state not pixels.
- **Frames are fetched on demand**, gated by event type (shots, contact, occlusion) —
  exactly where pure-state reasoning is known to fail (see [09_vlm_narration](09_vlm_narration.md)).

## Per-camera vs cross-camera (the key boundary)
- **Per-camera (B–D)** runs ×4 in parallel; each camera independently detects, tracks, and
  attributes. Cheap, parallelizable, edge-friendly. Per-camera IDs are *local* and not
  trusted as global identity.
- **Cross-camera (E)** is the system's core value and the hard part. It consumes the 4
  per-camera tracklets + their court projections + attributes and produces the single
  global identity. This is what the Roboflow blog and most open repos do **not** do.

## Module → doc map
| Stage | Module | Doc |
|---|---|---|
| A | Ingest/sync, calibration | [03](03_cameras_and_calibration.md) |
| B | Detection | [04](04_detection.md) |
| C | Tracking + masks | [05](05_tracking_and_masks.md) |
| D | Team / jersey / pose | [05](05_tracking_and_masks.md), [07](07_pose.md) |
| E | Cross-camera fusion | [06](06_cross_camera_fusion.md) |
| F | World model + events | [08](08_world_model.md) |
| G | VLM narration | [09](09_vlm_narration.md) |

## Data contracts (interfaces between stages)
- **Detection → Tracking**: per frame, list of `{box, score, class}` per camera.
- **Tracking → Fusion**: per frame, per camera, list of `{cam, track_id, box, foot_xy,
  team, jersey#?, reid_embedding, pose?}`.
- **Fusion → World model**: per frame, list of `{global_id, court_xy, team, jersey#?,
  name?, members:[(cam,track_id)], pose}` + ball + events.
- **World model → VLM**: per possession/event, compact JSON (see [08](08_world_model.md))
  + a frame-fetch handle for the clarifier.

Keeping these contracts stable lets every stage be swapped/retrained independently.
