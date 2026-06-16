# 07 · Pose

Pose enriches the world model so the VLM can describe *how* a play happened (fadeaway,
crossover, block, contest) — not just positions. v1 = 2D per camera; 3D = follow-on.

## v1: 2D skeletons per camera
- **Model: RTMPose / RTMO (MMPose, Apache-2.0)** — top-down (on tracked player boxes) or
  one-stage RTMO. **Replaces YOLO11-pose** used in the demo (AGPL — not shippable).
- **Top-down on the tracked box** (the demo lesson): full-frame pose misses far/small
  players; running pose on each tracked crop guarantees a skeleton per player and maps it
  straight to the track id — no extra association.
- COCO-17 keypoints `[x, y, confidence]` per player per camera, attached to the global id via
  its per-camera members.
- Edge: RTMPose is lightweight and TensorRT-exportable; budget it as one pass per tracked
  crop, and/or run at reduced rate (pose changes smoothly — 10–15 fps + interpolation is
  enough, halving cost).

## Refines tracking too
The **ankle midpoint** from pose is a more accurate court foot-point than box-bottom-center →
feed it back into the cross-camera projection ([06](06_cross_camera_fusion.md)) for better
court positions, which directly reduces occlusion mis-association.

## Follow-on: 3D pose (triangulation)
With 4 calibrated cameras, lift the per-camera 2D keypoints to **3D court-space skeletons** via
multi-view triangulation:
- Open options researched: **mvpose** (Apache-2.0, epipolar match + triangulate),
  **VoxelPose** (MIT, 3D voxel — heavier), **SelfPose3d** (self-supervised, no 3D labels).
- Caveat: 3D quality is bounded by calibration — far cams (rectilinear) good, near cams
  (SuperView) weaker. Start with the center/far-cam-owned regions.
- Value: true 3D pose enables shot-form/biomechanics and unambiguous action description; it's
  the richest world-model signal but not required for v1 narration.

## Feeds the world model
Per player per frame: `pose_2d: {cam: {joint: [x,y,conf]}}` (v1) and later
`pose_3d: {joint: [x,y,z,conf]}`. Used by event detection (e.g. arm raised + ball release =
shot attempt; vertical extension of a nearby defender = contest/block) and by the VLM for
descriptive verbs. See [08](08_world_model.md).
