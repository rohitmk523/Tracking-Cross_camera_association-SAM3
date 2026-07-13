# Research: what the field does that we don't (2026-07-14)

## The honest architectural diagnosis
Our tracking associates by MOTION+IoU only (ByteTrack), then claims identities
with jersey reads, then corrects cross-camera. Appearance (KPR) is consulted
only downstream, never during track FORMATION — so in scrums the tracklet
itself goes wrong and every later layer inherits the error. Our measured
ceilings (80-89% "right track exists") and the refuted scorer-side
experiments all point at the same conclusion the literature reached:
**appearance must live inside association, and tracklets need offline
global refinement.**

## What the state of the art actually is (sports MOT, 2024-26)
1. **Deep-EIoU** — replaces Kalman/IoU assumptions with expanded-IoU
   association tuned for erratic sports motion; the standard online tracker
   for SportsMOT (HOTA 77.2). No official basketball-specific successor beats
   the family below.
2. **GTA / GTA-Link (arXiv 2411.08216; SoccerTrack-2025 winner "GTATrack")**
   — OFFLINE global tracklet refinement: per-tracklet ReID embeddings →
   split mixed tracklets (detect identity switches WITHIN a tracklet via
   embedding clustering) → merge fragments across time. **+10.2 HOTA over
   SORT, +3.8 over Deep-EIoU; SportsMOT SOTA 81.0-81.6.** This is the
   highest-leverage, pure-post-processing idea — fits our per-chunk caches
   exactly, zero new video compute.
3. **SportSORT (Springer MVA 2025)** — jersey COLOR + NUMBER inside the
   association cost + corrective matching after long occlusions +
   out-of-view re-association (HOTA 81.3/88.0). Validates our jersey-claim
   idea but does it AT MATCHING TIME, not after.
4. **KPR (ECCV 2024)** remains SOTA for our exact niche (occluded,
   multi-person-ambiguous ReID with keypoint prompts). No drop-in successor;
   the upgrade is not a better model but a better USE: per-tracklet averaged
   part-features for split/merge (GTA recipe), not just tie-breaking.
5. **SAM2-guided association (McByte, SAM2MOT, MoSAM 2025)** — masks as a
   guidance signal cut ID switches in crowds; SAM2 has memory + occlusion
   recovery and is ~10x lighter than SAM3. The "SAM in between" idea is
   alive: run SAM2 ONLY inside detected scrum windows (rebound fights,
   paint clusters ≈ 5% of frames) to hold identities through the pile-up,
   then hand back to the cheap tracker. Est. ~$1-2/game on A10G vs $25+ for
   full SAM3.
6. Jersey-number reading (Koshkina & Elder, CVPR-W 2024 framework): our
   dense read+legibility stack already matches this design.

## Recommended experiments, in order (each measurable on the 3-game board)
- **R1. GTA-Link-style tracklet split+merge with our ft-KPR embeddings**
  (offline, per chunk, local): split hybrid streams at intra-tracklet
  embedding breaks; merge fragments by part-based similarity + jersey/kit
  constraints. Attacks scrum identity = rebound WHO + paint FG + c2a chains.
  Expected from literature: +4-10 tracking points -> WHO ceiling AND pick.
- **R2. Jersey/kit cost inside association** (SportSORT-style): add
  kit-color histogram + last-confident-number as soft costs in ByteTrack
  matching (cheap; we already compute shades at anchor time).
- **R3. SAM2 scrum windows**: detect pile-ups (>=3 boxes overlapping near
  ball), run SAM2 masklets for the 2-3s window per cam, use masks to
  re-assign the tracklets through the scrum (McByte-style guidance).
- R4 (optional): swap ByteTrack -> Deep-EIoU wholesale if R2 insufficient.

Sources: arxiv.org/abs/2411.08216 (GTA), arxiv.org/abs/2602.00484
(GTATrack), Springer s00138-025-01756-y (SportSORT), arxiv.org/abs/2407.18112
(KPR), arxiv.org/abs/2504.04519 (SAM2MOT), arxiv.org/pdf/2505.00739 (MoSAM),
arxiv.org/pdf/2409.14220 (McByte), MCG-NJU/SportsMOT.
