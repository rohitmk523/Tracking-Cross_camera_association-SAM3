# 05 · Per-Camera Tracking, Masks & Attributes

This stage runs **independently per camera** (×4). It produces stable *local* track IDs plus
attributes (team, jersey #, pose-ready crops). Global identity is NOT decided here — that's
[06_cross_camera_fusion](06_cross_camera_fusion.md).

## Why not SAM3 as the tracker (decision)
SAM3 is a strong per-frame *segmenter* but a **single CUDA-only model with no occlusion-aware
temporal memory** — exactly the dimension that breaks on ~12 mutually-occluding players over
long clips. We split **masks** from **tracking**:
- **Tracking = detect + ByteTrack/BoT-SORT** (motion + appearance), cheap and edge-friendly.
- **Masks = Cutie** (multi-object VOS) only where we actually need silhouettes/occlusion
  propagation. SAM3/SAM2 remain useful as offline **mask seeders**, not the tracker.

## Tracker
- **ByteTrack / BoT-SORT** via `supervision` (MIT). BoT-SORT adds appearance (ReID) +
  camera-motion compensation; ByteTrack is the lighter baseline. Fixed cameras → no global
  motion, so ByteTrack is often enough; BoT-SORT with ReID helps through brief occlusion.
- `minimum_consecutive_frames` debounce to suppress flicker births (blog uses 3).
- Output per camera: `{cam, track_id, box, foot_xy, score}` per frame.

## Masks & occlusion (layered, by need)
| Situation | Tool | License |
|---|---|---|
| Default multi-object masks (when needed) | **Cutie** | MIT |
| Offline "ground-truth-grade" long-clip quality | **SAM2.1-L + SAM2Long** | Apache / verify |
| 1–2 tracks lost in player-on-player occlusion | **SAMURAI** (motion) / **DAM4SAM** (distractor) on that track only | Apache / verify |

Masks are **optional** for the core tracking/identity path (boxes + foot-point + ReID +
jersey suffice). They add value for (a) occlusion propagation, (b) cleaner pose crops, (c)
visualization. Don't run heavy VOS on all 12 players every frame on the edge — reserve it.

## Attributes computed per track
- **Team**: `roboflow/sports` **TeamClassifier** (SigLIP embeddings → UMAP → KMeans, k=2),
  MIT. Fixed cameras give cleaner crops than broadcast → expect strong separation. Stored as
  A/B (+ REF from the detector's referee class).
- **Jersey number**: reuse our trained **ResNet** number reader (tight number-box crop →
  digit) + the e6 number-localizer pattern. Retrain general over time; it's the strongest
  same-uniform identity cue and the key to cross-camera ID (two tracks reading the same
  number on the same team are the same player).
- **ReID embedding**: **Torchreid OSNet** (MIT) appearance vector per track, for cross-camera
  matching and re-entry. Fine-tune on basketball crops (DeepSportradar-ReID, our crops).
- **Pose crop**: the track box feeds RTMPose ([07](07_pose.md)).

## Occlusion philosophy (important)
Per-camera tracking *will* lose/swap IDs under heavy occlusion — that is expected and OK,
because **cross-camera fusion repairs it**: when camera A loses a player behind a screen,
cameras B/C/D still see him, and the global identity persists. We do **not** over-engineer
single-camera occlusion; we lean on the multi-view redundancy ([06](06_cross_camera_fusion.md)).
Single-object rescue (SAMURAI/DAM4SAM) is a targeted offline tool, not the default.

## Outputs & contract
Per camera, per frame: `[{cam, track_id, box, foot_xy, team, jersey#?, reid_embedding,
pose?}]` → consumed by cross-camera fusion.
