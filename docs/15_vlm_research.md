# 15 · VLM Research — picking the narration model (+ Twelve Labs verdict)

Research-grade comparison of the candidate VLM/LLM models for the narration layer
([09_vlm_narration](09_vlm_narration.md)), ranked on the one output that matters:
**jersey-grounded play-by-play** — e.g. *"red #7 dribbles through two defenders, pulls a
fadeaway 2, blocked by green #10."* Includes a verdict on the client's **Twelve Labs
(Pegasus / Marengo)** proposal and a concrete **clip bake-off plan** to validate empirically.

> Method: 18-agent web-verified research sweep (mid-2026 pricing/model lineups), each finding
> adversarially fact-checked against primary vendor docs, model cards, and benchmark papers.
> Knowledge as of **2026-06-22** — pin model versions and re-verify pricing before launch.

---

## TL;DR

1. **Identity is CV-owned, not VLM-owned.** Every benchmark and every working precedent says the
   same thing: out-of-box VLMs cannot reliably read jersey numbers (a basketball VLM scored
   **56%** vs **93%** for a CV classifier; SOTA ~90% jersey reading is achieved *only* by CV
   pipelines aggregating over a whole tracklet). The VLM is a **gated pixel-confirmer**, never the
   identity source. This is the architecture in [08](08_world_model.md)/[09](09_vlm_narration.md) —
   the research confirms it hard.
2. **Primary pick: Gemini 3 Flash** as the native-video frame-clarifier (Batch mode, Set-of-Marks
   grounding), with an LLM (Gemini 3.1 Pro *preview* or GPT-5.5) as the state-reasoner over event
   JSON. Cheapest viable cloud path, native video, 1M context, strict JSON, pixel-precise pointing.
3. **On-prem fallback: Qwen3-VL-8B (Apache-2.0)**, fine-tuned on the 41k `plays`. Keeps footage in
   your VPC; best open OCR; native video with timestamp grounding.
4. **Twelve Labs verdict: PILOT Marengo for semantic retrieval; Pegasus is at most a summary /
   chaptering layer and a *candidate* clarifier — neither is the play-by-play engine.** The client's
   framing ("enrichment layer on top of our models") is correct **provided identity always comes
   from CV.** Most of his specific beliefs are true-but-operationally-irrelevant; one (2-hour video)
   applies to a path you wouldn't use per play.

---

## 1. Comparison matrix

| Model (pin this version) | Video-native | Max video / context | Jersey-OCR & identity grounding | Fine-grained sports action | Pricing (our clip / game scale) | License / data-handling | On-prem? | Fit as frame-clarifier |
|---|---|---|---|---|---|---|---|---|
| **Gemini 3 Flash** (`gemini-3-flash`) / **3.5 Flash** | **Yes** — native; 1 fps default, configurable fps, start/end clip offsets | 1M ctx; ~1 hr @ default res, ~3 hr @ low. 10–40s clips trivial | **Weak on raw frame, strong with crops.** VideoZeroBench: 17.0%→**41.2%** with temporal-segment + spatial-crop. Strong general OCR (OmniDocBench 1.5 = 0.115). **Not the identity source.** | Strong high-level (Video-MMMU 87.6%); block-vs-foul needs higher fps + crops | **Cheapest cloud.** ~70 tok/s video @ low res → **<$0.01/clip**, ~**$0.05–0.40/game** gated in Batch (50% off) | Paid API. **Paid tier contractually NOT trained on** (incl. video); Vertex AI **ZDR** for enterprise; free tier IS used — never send footage on free tier | No | **Primary.** Native video + 1M ctx + `responseSchema` JSON + Set-of-Marks via 0–1000 normalized boxes |
| **OpenAI GPT-5.5** (`gpt-5.5`) / 5.4-Mini | **No** — Responses API **rejects video**; DIY ffmpeg frame-sampling only | ~1.05M ctx, 128K out; frames eat context | Weak-to-moderate; broadcast jersey-# **unvalidated** | Mediocre for generalists (DeepSport live-commentary **22.76**) | $5/$30 (5.5); ~$0.08–0.30/clip input-only | Closed API; no-train by default; **30-day** abuse retention; ZDR needs enterprise | No | **Best as the narration LLM**, weak clarifier (no native video, loses motion/audio) |
| **Twelve Labs Pegasus 1.5** (SaaS) / **1.2** (Bedrock `twelvelabs.pegasus-1-2-v1:0`) | **Yes** — native; URL/asset/base64, no pre-index for `/analyze` | Sync `/analyze` **4s–1 hr** (*not* 2h/12h); 2 hr only via indexed path | **Weak / not unaided.** OCR claims **NAMES on broadcast jerseys only** — no small-number / wide-cam claim, no accuracy figures. Output is generic ("player in red"). | Moderate/unproven; block/steal/who-touched not independently benchmarked | Per-clip **$0.005–0.018**; full-game 4-angle **indexing ≈ $10–15 = cost trap** | SaaS US-hosted, SOC 2 T2; privacy policy permits "improve the Services" (no explicit no-train). **Prefer Bedrock 1.2** (video stays in your S3) | Via Bedrock (in-account, not on-prem weights) | OK per-clip narrator; **poor identity source**; no shippable weight |
| **Qwen3-VL-8B-Instruct** (Apache-2.0; 32B for higher acc.) | **Yes** — native dynamic-fps, ~768 frames, text-timestamp grounding | 256K → 1M ctx; ~2 hr video. 10–40s clips ideal | **Best open OCR** (OCRBench 89.6). Still confirm digits on **CV crops**, not full frame | MVBench 68.7 (8B); no basketball benchmark — **fine-tune on the 41k plays** | On-prem GPU: 8B on g6.xlarge (L4) **$0.369/hr (3yr)–$0.805 OD**; 32B on g6e (L40S) $0.80–1.86/hr → ~$0.10–0.50/game at high util | **Apache-2.0, fully permissive.** Footage never leaves your VPC. vLLM ≥0.11.0 | **Yes** | **On-prem fallback.** Native video + timestamps + best open OCR + fine-tunable (InternVL3.5-8B Apache-2.0 is the clean alt) |
| **LLM-only state-reasoner** (narrate over event JSON): **Gemini 3.1 Pro preview** or **GPT-5.5** | N/A (text in) | 1M ctx | N/A — identity arrives **pre-resolved** in the JSON | N/A — reasons over CV events, not pixels | Output-dominated; Gemini 3.1 Pro $2/$12 (≤200k), batch halves | Paid API, no-train on paid/enterprise tier | No | **This is the narration core** — calls the VLM only on gated events |

**Corrections applied during verification** (don't repeat the originals):
- Pegasus synchronous `/analyze` caps at **1 hour**, not 2h/12h. The 2-hour figure is real but only on
  the **indexed / segmentation** path — irrelevant to per-clip calls. Any "~12 h" claim is third-party, not vendor.
- "60 fps documented for sports/games" is **not vendor-supported** — Gemini docs only say *raise fps for
  fast-action / high-speed motion*. Raise fps on gated clips, but don't cite "60 fps for sports."
- Gemini **3.1 Pro is a preview** model (`gemini-3.1-pro-preview`).
- Gemini **2.0 Flash hard-shutdown 2026-06-01**; **2.5 Flash EOL slated 2026-10-16** → pin versions.
- Google **paid** Gemini API + Vertex AI **contractually do not train** on prompts/responses (incl. video files);
  Vertex offers **zero-data-retention** — a *stronger* privacy posture than "cloud API" implies.

---

## 2. Ranked recommendation

### Primary — **Gemini 3 Flash as the gated frame-clarifier (Batch mode) + Set-of-Marks grounding**
Native-video clarifier (`gemini-3-flash`) with `gemini-3.1-pro-preview` or GPT-5.5 as the LLM narrator
over event JSON — one API surface, native video, 1M context, strict JSON via `responseSchema`, function
calling. The grounding pattern is **non-negotiable** (see §5): overlay CV-derived marks (numeric ID badge +
colored mask + bbox) on the **CV-cropped, event-segmented** clip, and ask Gemini only to *confirm the action
against the labels* — never to read the jersey or set identity. Directly validated: VideoZeroBench
**17.0%→41.2%** with temporal-segment + spatial-crop; Set-of-Marks hits 75.6 mIoU on RefCOCOg (beats
fine-tuned PolyFormer 67.2). Batch (50% off, 24h SLA) fits batch-v1 and keeps a game in the **low
single-digit dollars**.

### Runner-up — **Twelve Labs Pegasus 1.5 per-clip (preferably Bedrock 1.2)**
A managed, purpose-built video-QA layer with schema-driven timestamped JSON, used **per gated clip only**
($0.005–0.018/clip) — **never full-game 4-angle indexing** (~$10–15/game just to index). Genuinely
video-native and operationally simple, but a **weaker identity grounder than Gemini** (no Set-of-Marks pixel
pointing; OCR claims only large broadcast *names*). Run it as **Bedrock Pegasus 1.2** so footage stays in
your own S3, and validate action recognition against the 41k-play eval before trusting any event call.

### On-prem fallback — **Qwen3-VL-8B-Instruct (Apache-2.0), fine-tuned on the 41k plays (32B for higher accuracy)**
When data residency or a permissive shippable weight is required: Apache-2.0 at every size, best-in-class
open OCR (OCRBench 89.6), native dynamic-fps video with text-timestamp grounding aligned to event windows,
first-class fine-tuning (LLaMA-Factory / ms-swift). Runs the gated clarifier on a single g6.xlarge (L4) or
g6e (L40S for 32B). The open-vs-closed video-QA gap (~68–73 vs ~85+ Video-MME) is acceptable precisely
because identity comes from CV and the VLM only confirms short crops; fine-tuning on your plays closes most
of the practical gap. **Avoid MiniCPM-V** unless legal clears its separate Model License; **InternVL3.5-8B**
(Apache-2.0) is the clean alternative.

---

## 3. What NOT to rely on the VLM for (hard constraints)

- **Identity / jersey numbers.** Out-of-box VLMs hallucinate impossible numbers ("011", "3000"); Roboflow's
  basketball VLM hit **56%** vs a **93%** CV classifier. SOTA jersey reading (~90%) needs CV tracklet
  consensus + legibility filter (numbers legible on ~5% of frames). **Any number not on the team roster is
  rejected.** ICCV 2025 LLM-IAVC: inject identity from a dedicated CV network because model-inferred identity
  "is often wrong."
- **Precise temporal localization.** Gemini **8.0%** on temporal grounding (VideoZeroBench); GPT-4o **38.5%**
  on TemporalBench (humans 67.9%). The CV/event-stream owns *when*; the VLM confirms *what* on a pre-segmented clip.
- **Ordering / who-shot-first / who-touched-last.** Derive from CV tracking + ball/event detection.
- **Free-form fine-grained action.** Fadeaway vs pull-up, block vs clean contest vs foul is the failure
  frontier (SCBench best **5.44/10**). Give the VLM a **closed set** of CV-proposed hypotheses to choose
  among; treat low confidence as "unconfirmed."
- **Whole-game single-pass narration.** Accuracy degrades on long video (Sports is the weakest Video-MME
  subdomain, 68.6%) and cost balloons. Keep the VLM on short gated clips (10–40s) at the relevant angle.
- **BLEU/CIDEr for narration quality.** Near-useless (SCBench BLEU<1.0). Score against the 41k `plays` with a
  rubric/LLM-judge **plus a hard identity-attribution accuracy metric** vs CV ground-truth IDs.

---

## 4. Twelve Labs verdict (the client's proposal)

The client's instinct — *"an enrichment layer on top of the basketball models we'd be running ourselves"* —
is **correct**, with one rule: **identity always comes from CV.** Detail:

### 4a. Myth-check of his three beliefs
| Belief | Verdict | Reality |
|---|---|---|
| "Analyze videos up to **2 hours**" | **PARTLY TRUE** | 2 h only on the **indexed/segmentation** path; the per-clip `/analyze` you'd actually call caps at **1 h** (Bedrock 1.2: 1 h, <2 GB). Your 10–40s plays are trivially inside limits either way — so it's impressive but operationally irrelevant. |
| "A **VLM call per clipped play** (score/block/steal)" | **TRUE — strongest fit** | `/analyze` takes URL/asset/base64 with **no pre-indexing**, supports `start_time`/`end_time` clipping, streams NDJSON, returns JSON, batches up to 1,000 requests. ~$0.005–0.018/clip. Caveat: confirms *what*, not *who*. |
| "Send **far-side angles** and they **summarize key moments**, flipping by side" | **PARTLY TRUE / mostly FALSE as reliable** | Summaries/chapters/highlights are real features, but: (1) far-side = small distant players, **unproven** regime, no published accuracy; (2) there is **no side-flipping feature** — that's *your* pre-processing; (3) the summary is **jersey-blind** ("a player in red drives and scores"), not the named play-by-play he pictures. |

### 4b. Core limitation — can Pegasus/Marengo produce jersey-grounded play-by-play unaided?
**No — decisive, true for both.** **Marengo** is embedding/search only (output modality = "Embedding"); it
returns ranked clip relevance, never a jersey number bound to a player — it cannot narrate. **Pegasus** can
narrate but **not with grounded identity**: its OCR only claims reading large broadcast *names*, with zero
accuracy figures and no small-number/wide-cam claim; the only identity lever (`prompt_v2`, ≤4 reference
images/entity) doesn't scale to 10 players + numbers. This is a whole-class limitation (SCBench: every video
LLM fails, dominant failure = fabricated names + wrong attribution). Working precedents (ICCV 2025 LLM-IAVC,
VC-NBA-2022/KEANet, WSC "Roger") all **inject identity from a dedicated CV network**. If Pegasus ever sets or
overrides identity, it will corrupt global-ID fusion.

### 4c. Where it genuinely fits (net-new value vs our event stream)
| Capability | Owner | Twelve Labs additive? |
|---|---|---|
| Jersey #, team, global ID | **CV pipeline** | No — never delegate |
| Shot/make/miss/possession/block/steal/turnover with exact frames | **Deterministic event stream** | No — cheaper & more accurate from our pipeline |
| Jersey-grounded narration | **World-model LLM over event JSON** | No — Pegasus is identity-blind |
| **Fuzzy/semantic highlight retrieval** ("find all fadeaways over a contest", "and-ones") | — | **YES — Marengo.** Text→video search our symbolic JSON can't express; docs cite improved basketball-action recognition |
| **Cross-game / archive / dataset-curation search** across angles | — | **YES — Marengo** |
| **Candidate-gating** (cheap fuzzy pass → CV event stream + VLM confirm) | — | **YES — Marengo** as a pre-filter |
| **Coach/game-level summaries, chaptering, highlight reels** (jersey-agnostic) | — | **MAYBE — Pegasus**, pending far-side recall test |
| Per-gated-clip "confirm the action from pixels" | — | **Possible — Pegasus**, but competes with Gemini (cheaper, native, Set-of-Marks) and Qwen3-VL (in-VPC) |

### 4d. Verdict — **PILOT Marengo (retrieval); bench Pegasus as a candidate clarifier; DROP it as the play-by-play engine**
- **Marengo →** semantic highlight/clip retrieval + cross-game archive search; optional candidate-gating pre-filter.
- **Pegasus →** optional jersey-agnostic game/coach summaries & chaptering; optional per-clip action-confirmer
  (must out-compete Gemini native-video and Qwen3-VL first).
- **Data/privacy:** Pegasus per-clip `/analyze` is **transient (no index)** — fine for gated clips. **Do NOT
  index full 4-angle games** (~$15/game for 90 min × 4 angles before any search). Marengo **requires
  indexing** but async embeddings are retained server-side only **7 days** → persist vectors in your own
  store (Qdrant/Elasticsearch); Search API is **$4/1k queries**. Prefer the **Bedrock path** (embeddings →
  your S3; Pegasus reads your S3) to keep footage in-region, or negotiate enterprise no-train/retention.
  Neither ships as an on-prem weight — both are external paid-API dependencies with their own roadmaps.

---

## 5. The grounding pattern that makes "red #7 → blocked by green #10" work

The named line never comes from the VLM reading pixels. It comes from **CV identity fused into the prompt +
pixels**, with the VLM only confirming the action. Recommended pattern for the frame-clarifier:

1. **Segment** the clip to the event's `frame_window` ([08](08_world_model.md)) at the relevant angle — not
   the wide 4-camera frame.
2. **Crop** to the players involved (CV bboxes) so the action fills the frame (this is the 17%→41% lever).
3. **Set-of-Marks overlay**: draw each involved player's bbox + a numeric ID badge + colored team mask onto
   the frames (Gemini's 0–1000 normalized `[ymin,xmin,ymax,xmax]` makes this exact).
4. **Prompt**: pass the event hypothesis + the per-player roster/jersey map + a **closed set of candidate
   actions**; ask the VLM to *choose and confirm* ("is mark ②'s shot a fadeaway, and is it blocked by mark
   ⑤? yes/no + which"), **per entity** to avoid role confusion.
5. **Reject** any VLM identity claim not matching the CV-supplied marks; low confidence → "unconfirmed,"
   narrate from state.

This keeps the VLM doing what it's good at (confirming *what*, on a small labeled crop) and never trusts it
for *who* or *when*.

---

## 6. Cost model at game scale (mid-2026 prices)

| Path | Per clip | Per game (gated: ~80–150 events × 5–15s) | Notes |
|---|---|---|---|
| **Gemini 3 / 2.5 Flash, Batch, low res, gated** | **<$0.01** | **~$0.05–0.40** | Cheapest viable. Dominant lever is *VLM-seconds = events × clip length × fps*, not model choice |
| **Twelve Labs Pegasus per-clip** | $0.005–0.018 | ~$0.5–2 (per-clip) / **~$10–15 (full 4-angle index — avoid)** | Per-minute billing; never index full games |
| **On-prem Qwen3-VL-8B (g6.xlarge L4)** | — | **~$0.10–0.50** at high util | Removes per-call cloud cost; real ops burden + unbenchmarked throughput |
| **Marengo indexing (if piloted)** | — | ~$3.78/angle (90 min); **~$15.12 all 4 angles** + $4/1k queries | Scope to selective indexing (one canonical angle / flagged plays only) |

**Cost driver order:** (1) total VLM-seconds (gating discipline), (2) media resolution, (3) Batch vs sync,
(4) model choice. Gating beats model choice by 1–2 orders of magnitude — which is why [09](09_vlm_narration.md)'s
event-gating is the real cost control.

---

## 7. Empirical bake-off plan (validate the analysis on our own clips)

Run the shortlist against real clips and score vs the 41k `plays` ground truth ([14](14_games_and_clips.md)).
This either confirms or overturns §1–§2 with our own footage.

**Models to test:** Gemini 3 Flash (primary), Twelve Labs Pegasus 1.5 / Bedrock 1.2 (runner-up + the client's
ask), Qwen3-VL-8B (on-prem). Optional: GPT-5.5 frame-sampled as a narration-LLM check.

**Clip set (driven by `plays`, not random) — ~6–10 clips spanning:** clean FG make · **block/contest** ·
**steal/turnover** · 3PT make · fast break · **occlusion-heavy** play. Both an **easy** identity case
(`e6fba750` Blue/Green) and the **hard** one (`c2a354fe` Black/White, same-brightness kits). Slice around each
play's `start/end_timestamp` on the far-side angle (FL/FR) per the clip-window convention in [14](14_games_and_clips.md).

**Three conditions per model** (isolates where the grounded line comes from):
1. **Raw clip + minimal prompt** — standalone VLM capability (what Twelve Labs gives unaided).
2. **Clip + Set-of-Marks overlay** — bbox + global-ID/jersey# drawn on frames (grounding-via-pixels).
3. **Clip + event JSON / roster in prompt** — the production path.

Condition 1 vs 2 vs 3 is the experiment that proves whether jersey-grounded output comes from the VLM or from
**our IDs** — the whole architectural thesis.

**Scoring (blind, vs `plays`):** action correctness · outcome correctness · **jersey-ID accuracy** ·
hallucination rate · notes-similarity vs the human/Gemini `note` (LLM-judge in bulk + human spot-check).

**Pass bars:** primary must produce correct grounded narration in condition 3 on the majority of possessions
with frame fetches only on expected event types ([13](13_evaluation.md)); Pegasus must **beat or match** a
cheaper incumbent to justify a second vendor; Marengo must beat keyword/event-stream filtering on fuzzy queries
(recall@10).

> **Open question to size condition 2/3:** do we have world-model/event JSON + tracking-ID overlays for any of
> these games yet? If not, the first bake-off runs **condition 1 only** (raw clips) until the CV pipeline emits
> a test game, then conditions 2–3 follow.

---

## 8. Sources (primary, verified 2026-06-22)

- Gemini video understanding / media-resolution / pricing / structured output: `ai.google.dev/gemini-api/docs/{video-understanding,media-resolution,pricing,structured-output,gemini-3}`; model cards `deepmind.google/models/model-cards/`.
- Gemini pixel-pointing / Set-of-Marks: `blog.roboflow.com/gemini-3-pro/`; SoM paper `arxiv 2310.11441`.
- VideoZeroBench (zoom&crop 17→41%): `arxiv 2604.01569`. TemporalBench / SCBench: `arxiv 2412.17637`.
- OpenAI vision/video + pricing: `developers.openai.com/api/docs/{guides/images-vision,models/gpt-5.5,pricing}`; DeepSport `arxiv 2511.12908`.
- Twelve Labs Pegasus/Marengo: `docs.twelvelabs.io/{api-reference/analyze-videos/analyze,docs/concepts/models/...}`; `twelvelabs.io/{pricing,privacy-policy,blog/introducing-pegasus-1-5,blog/how-to-perform-video-ocr-...}`; Bedrock `docs.aws.amazon.com/bedrock/.../model-parameters-pegasus.html`, Marengo model card.
- Qwen3-VL: `github.com/QwenLM/Qwen3-VL`; OCRBench/MVBench leaderboards; AWS GPU pricing `instances.vantage.sh`.
- Jersey-OCR SOTA / identity injection: Koshkina & Elder `arxiv 2405.13896`; SoccerNet `arxiv 2309.06006`; Roboflow basketball `blog.roboflow.com/identify-basketball-players/`; LLM-IAVC `arxiv 2507.20163`; VC-NBA-2022/KEANet `arxiv 2401.13888`.

> This doc ranks and validates; [09_vlm_narration](09_vlm_narration.md) remains the canonical design.
> When the bake-off confirms the picks, fold the chosen model versions into [02_tech_stack](02_tech_stack.md)
> and [09](09_vlm_narration.md).

## Companion repo (research/testing demo)
The bake-off harness and a client-facing demo live in a **separate** repo:
**`VLM_Basketball`** (`../VLM_Basketball/`, sibling of this repo) — a FastAPI backend
(Supabase `plays` → S3 clip + events → Gemini `gemini-3.5-flash` / `gemini-3.1-pro-preview`)
and a React/Vite/Tailwind UI to run clips with/without events and compare. This research doc is
mirrored there as `docs/vlm_research.md`. Keep the two in sync; the production VLM layer is
integrated **here** (SAM3) later, not in that repo.
