# 00 · Overview

## What this is
A **general-purpose, production-grade** computer-vision system that tracks every player
across **4 fixed cameras** in a basketball game, fuses them into **one persistent identity
per player** on a top-down court, captures **pose**, builds a structured **spatial world
model**, and uses a **VLM/LLM** layer to produce **descriptive narration** of play.

## Why a new repo (vs DEMO_UBALL)
`DEMO_UBALL` produced an excellent *demo* but it is **overfit to one 60-second clip** of one
game (e6fba750): curated identities, a game-specific detector, hand-tuned spotlights. That
is the right thing for an investor demo and the wrong thing for a product. This repo rebuilds
the same capabilities to **generalize across games and courts** on the same 4-camera rig.

## Goals (v1, ~3 weeks)
1. **Detection** that finds every player/ref/ball across all 4 angles, including the hard
   case: small players at the far endline.
2. **Per-camera tracking** with stable per-camera IDs, occlusion-robust.
3. **Cross-camera fusion**: one global ID per player, consistent across all 4 cameras and
   through occlusions — IDs and jersey numbers resolved *in unison* across views.
4. **Pose** (2D skeletons) per player feeding the world model.
5. **Spatial world model**: per-frame/per-event JSON (positions, pose, ball, possession,
   team, IDs, events).
6. **VLM narration (batch)**: descriptive play-by-play from world state, pulling frames only
   when pixels are needed.

## Non-goals for v1 (explicit follow-on)
- **3D triangulated pose** (2D per camera first).
- **Live / streaming narration** (post-game/batch first).
- **Full Jetson real-time** at 30 fps on all modules (edge path designed for now, hardened
  later).
- **Auto-calibration for arbitrary venues** (we assume the same 4-cam rig; per-court
  calibration is an assisted onboarding step).

## Deployment shape (decided)
**Hybrid**: tracking + pose + world-model on the **Jetson edge**; the compact world-state
JSON is sent to **cloud VLM APIs** for narration. See [11_edge_deployment](11_edge_deployment.md).

## Commercial constraint (decided)
This ships as a **commercial product** → **permissive-license-only** dependency policy and
commercially-usable training data only. Non-commercial datasets are for benchmarking, never
in shipped weights. See [02_tech_stack](02_tech_stack.md) and [10_data_and_licensing](10_data_and_licensing.md).

## Generality target (decided)
**Same 4-camera rig, many courts.** Fixed camera layout; per-court calibration. This bounds
the cross-camera design (known camera topology) and the calibration problem (per-venue, not
per-frame).

## Success criteria (high level; metrics in [13_evaluation](13_evaluation.md))
- Player detection recall on a **held-out court**, incl. far-endline small players.
- Cross-camera **IDF1 / ID-consistency** through occlusions (one ID per player, end to end).
- World-model JSON that is faithful enough that VLM narration is correct on possessions.
- Runs on recorded games end-to-end (batch) with a clear path to edge real-time.

## The north star reference
Roboflow's basketball pipeline (https://blog.roboflow.com/identify-basketball-players/) —
detection → tracking → team → jersey number → identity. We adopt ~70% of it (detection,
team, jersey) and **build the 30% it omits and we most need: cross-camera fusion + court
projection + the world-model/VLM layer.**
