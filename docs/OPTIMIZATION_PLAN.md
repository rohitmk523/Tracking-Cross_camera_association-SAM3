# Pipeline Optimization Plan — speed AND accuracy (post-yolo26s adoption)

**Decision (2026-07-11, user):** yolo26s is the production detector from now on.
RF-DETR-S FP16 is retained as the accuracy reference for A/B checks only.

**Goal:** the full pipeline runs on a MacBook Pro (M4 Pro) fast enough to be useful,
and on the Jetson AGX live path later — while accuracy goes UP, not down. Every
change ships only after re-scoring on the two GT windows (c2a blind minute +
e6 44_60) and logging to the ledger. No exceptions; this is how we caught every
regression so far.

---

## 0. Measured baseline (the thing we are optimizing)

One game-minute = 7,200 frames (4 cams × 1,800). E2E race, A10G, yolo26s lane:

| Stage | Seconds | Share | What it does today |
|---|---|---|---|
| Detect (yolo26s) | 115 | 7% | all 7,200 frames, one at a time |
| **Jersey OCR** | **834** | **53%** | ~19k crops → legibility → localizer(640) → PARSeq, ONE CROP AT A TIME |
| Pose | 183 | 12% | ~48k crops, batched (the proof batching works) |
| **Kit tags** | **430** | **27%** | RE-DECODES the video with random seeks to re-crop torsos |
| Track | 5 | 0% | ByteTrack + jersey claims |
| xcam + score | ~60 | 4% | correction math |
| **Total** | **~1,570** | | **26 min per game-minute** |

Laptop today (measured where noted): yolo26s detect **45.7 fps** (measured, MPS);
pose **~274 crops/s** (measured, ANE) — already GPU-class; OCR/kits est. 1.5-2×
slower than A10G. Laptop total est. **~35-45 min per game-minute**.

The logic of the whole plan in one line: **detection is 7%; OCR+kits are 80% and
almost all of it is waste** (unbatched inference, redundant reads, redundant
video decoding) — so we attack waste first (no accuracy risk), then reduce work
(measured accuracy gates), then port engines.

---

## Phase 1 — Remove pure waste (no accuracy change possible, by construction)

Same models, same crops, same reads — just executed properly. Output is
bit-identical or trivially equivalent; gate = scores unchanged on both GT windows.

1. **Batch the OCR stack.** Legibility ResNet18 batch-64 (it's a 160×96 classifier;
   per-crop Python calls are ~90% overhead). PARSeq batch-32. Localizer: batch, and
   drop imgsz 640→320 (number occupies most of the torso crop; verify on the
   OCR Master valid set first). Expected: OCR 834s → **~90-150s**.
2. **Merge kit-tagging into the OCR pass.** Kits re-decodes and re-seeks the same
   video to re-crop the same players. The OCR pass already holds the decoded frame
   and the crop; compute the torso shade there. Expected: kits 430s → **~10s**
   (the z-normalized clustering is milliseconds; it was 100% decode waste).
3. **Single-decode architecture.** Today each stage decodes the clips again
   (detect, OCR, pose, kits = up to 4 decodes of the same video). Refactor to one
   sequential decode per camera feeding all consumers (detector → crops for
   OCR/pose/kit from the in-memory frame). Also fixes the laptop's biggest
   I/O tax. Expected: saves ~60-120s and halves memory churn vs seeks.

**Phase 1 exit estimate: ~1,570s → ~450-550s per game-minute (A10G),
laptop ~8-12 min per game-minute.** Zero accuracy risk.

**PHASE 1 RESULT (2026-07-11, measured on the laptop, commit 78c1fd1):**
OCR 4-cam 813s(A10G) → **82s local** (~200-250 crops/s vs ~22/s unbatched local);
kit stage 417s → **0.12s** (inline shades + clustering, no re-decode). Combined
OCR+kits **15×**. GT gate PASSED: c2a strict 0.811 / fused 0.850 vs 0.810/0.856
reference (within established run-to-run noise; identical kit-split decisions;
FL event-level check 99.1% identical events, 99.7% number agreement).
New laptop total ≈ detect 160s + pose 150s + OCR 82s + track/xcam 65s ≈
**~7.5 min per game-minute** (from ~35-45). Single-decode consolidation deferred —
remaining decode cost is inside detect/pose passes, now a Phase-2 item alongside
stride-2.

## Phase 2 — Reduce work (each step has an accuracy gate)

4. **Triggered ("assisted") OCR** — the user-specified design, now formalized.
   Reorder: detect → track → OCR-on-demand (today OCR runs before tracking and
   reads blindly). Read a track's crop ONLY when:
   - the track is **unclaimed** (new, or claim lost);
   - its claim is **contested or low-confidence** (conflicting numbers, low read
     conf, recent identity flip);
   - a **collision/occlusion episode** just ended (box-overlap event — exactly
     where ByteTrack swaps IDs);
   - a **re-appearance** after a gap;
   - **heartbeat**: every ~2s per claimed track (tunable) to catch silent drift.
   Basis: today ~19k gated crops yield 6,983 confident reads for ~10 identities —
   the vast majority re-confirm an unchallenged claim. Expected crop volume −70-85%
   → OCR ~90-150s → **~25-50s**.
   **Accuracy gate:** the offline correction interpolates BETWEEN reads, so fewer
   anchors could widen error windows. Run c2a + e6 GT scoring at heartbeat = 1s /
   2s / 4s; adopt the cheapest setting that holds strict AND fused within 0.5pt of
   dense. If it doesn't hold, keep dense-batched for the finalization pass and
   triggered mode for the live path only — both profiles stay in the codebase.
5. **Detection stride-2 with tracker bridging.** Detect every 2nd frame; ByteTrack
   carries boxes between. Halves detect time (and helps every downstream stage that
   iterates detections). Gate: same GT scoring; far-court players are the risk
   (small, fast) — watch #43-e6 and #5B-c2a specifically.
6. **Pose only where it's used.** Pose currently runs on ALL detections; its
   consumers are sole07 projection (tracked players), kit torsos, and KPR prompts.
   Restrict to boxes attached to live tracks + OCR-trigger crops. Expected −30-40%
   pose time. Gate: cross-cam agreement (48cm baseline) must not regress.

**Phase 2 exit estimate: ~180-280s per game-minute (A10G) ≈ 3-4.5× real-time;
laptop ~4-7 min per game-minute.**

## Phase 3 — Engine ports + the 4th-class consolidation

7. **CoreML/ANE export of the hot models on the Mac** (yolo26s via ultralytics
   CoreML export; legibility + PARSeq via coremltools). Pose already proves the
   ANE path (274 crops/s). Expected 2-3× on detect + OCR locally.
8. **4th-class pilot: add "number" to yolo26s** (labels exist in Uball OCR Master,
   HF-backed). One inference pass then emits player+referee+ball+number boxes —
   the localizer stage is DELETED and number boxes come free at detection time.
   ~$3 retrain + the standard 3-level eval (the ball class must not regress:
   ≥0.855 valid mAP50). This is also the cleanest OCR trigger: a number box
   visible = a readable moment.
9. **Jetson AGX**: TensorRT engine of the (possibly 4-class) yolo26s inside the
   DeepStream container; two-pass design per docs/JETSON_STREAMING_QUESTIONS.md
   (live provisional identities; finalization pass for accuracy-grade output).

**Phase 3 exit target: laptop ~1.5-3 min per game-minute (≈1.5-3× real-time);
AGX live pass at stream rate with provisional IDs.**

## Accuracy track (runs in parallel — the other half of the mandate)

A. **Re-baseline on yolo26s everywhere** — DONE for e6+c2a (2026-07-11):
   e6 full stack (stride-1 anchors, ft-KPR) = **81.5 strict / 85.5 fused**
   (RF-DETR 83.2 strict). #11 94(+2), #43 77(+3), #22 78(−4), **#6 77(−8 —
   THE outlier: reads +14% yet worse; diagnose detection/claims in #6's zone)**.
   26s anchors are DENSER than RF-DETR's (9,461 vs 8,522; #43 +55%).
   KEY finding for Phase 2: stride-2 anchors cost −2.7 strict on e6 (81.5→78.8)
   — read DENSITY is a first-order accuracy knob; the heartbeat sweep must
   measure it, and the finalization pass likely keeps dense reads (now cheap:
   140s stride-1 4-cam local). f66 rebuild pending (no GT — low priority).
B. **Kit-aware identity on e6** (#22 has a same-number opponent; c2a showed
   kit-awareness is worth up to +26pts on affected players — likely ~+1pt mean).
C. **#43 far-court FR diagnosis** (worst cell in the failure heatmap).
D. **KPR flywheel cycle 2**: retrain on e6+c2a+f66 auto-labels from the 26s
   pipeline (cycle 1 gave +0.9 mean at $2.50).
E. **More OCR anchors at distance**: bigger jersey numbers with the venue
   (physical, highest leverage); 4th-class number detection may add mid-range
   reads the 90px crop gate currently drops.
F. Every phase-2/3 speed change re-scored on BOTH GT windows before adoption —
   speed never buys a silent accuracy cut.

## Execution order & effort

| # | Item | Cost | Risk | When |
|---|---|---|---|---|
| 1 | Batch OCR + merge kits + single decode | 1-2 days code | none (gated identical) | now |
| 2 | Re-baseline 26s full stack (A) | ~1h laptop + $0 | none | now, parallel |
| 3 | Triggered OCR + heartbeat sweep (4) | 1-2 days | gated | after 1 |
| 4 | Stride-2 + pose subset (5,6) | ~1 day | gated | after 3 |
| 5 | Kit-aware e6 + flywheel 2 (B,D) | ~$3 AWS | low | parallel |
| 6 | CoreML exports (7) | ~1 day | none | after 4 |
| 7 | 4th-class retrain + eval (8) | ~$3 AWS | ball-class gate | after 4 |
| 8 | AGX TensorRT + DeepStream (9) | blocked on engineer Qs | — | when answered |
