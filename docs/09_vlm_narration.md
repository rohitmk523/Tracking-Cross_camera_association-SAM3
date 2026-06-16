# 09 · VLM / LLM Narration

Turns the world model into **descriptive play-by-play** ("dribbled through two defenders,
pulled a fadeaway, got blocked"), not just classification (make/miss). This realizes the
client's thesis: **reason over world state, not raw pixels** → near "VLM-watches-every-frame"
quality at low latency/cost, with **frame fetches only when pixels are truly needed.**

## Architecture validated (with precedent)
- **Frame-on-demand beats dense ingestion**: Stanford **VideoAgent** (ECCV'24) hits 54% on
  EgoSchema using ~8 frames via LLM-driven frame selection — ~20× fewer frames than dense.
- **Structured-state → language is deployed**: **WSC Sports'** "Large Sports Model" generates
  multilingual narration from structured play data **live in the NBA app**; academic
  **LLM-Commentator** fine-tuned Llama-7B on football events → 0.91 F1 event-to-text.
- **Novel ground**: existing *basketball* narration is video-captioning; pure
  tracking/state-first narration is largely unexplored — our differentiator.

## Two roles, two models
| Role | Job | Model (cloud, v1) | On-prem option | Notes |
|---|---|---|---|---|
| **State-reasoner** | reason over world-model **JSON/events** → narration | **LLM** (Gemini Flash/Pro or Claude Sonnet) | **Qwen3-32B (text)**, Apache-2.0 | Pure text/JSON job — a VLM adds nothing; cost compounds (called per possession) |
| **Frame-clarifier** | look at actual frames/clip when needed | **Gemini (native video)** | **Qwen3-VL-32B**, Apache-2.0 | Only Gemini ingests video natively; others are image-only (DIY frame sampling) |

Cloud APIs are fine for a commercial product (paid service, no code-license contamination).
On-prem (Qwen) only if a venue mandates data residency.

## Event-gated frame fetching (the key rule)
**Gate frame fetches on EVENT TYPE, not token budget.** The moments pure-state reasoning fails
are predictable, so fetch frames exactly there:
- **Always fetch**: shot attempts (type: fadeaway/floater/layup), contact/fouls, blocks/
  contests, turnovers, who-touched-last ambiguities, high-occlusion windows.
- **Never fetch** (state is enough): routine dribble/pass/movement, positions, spacing,
  possession flow.
Implemented via the `needs_frame_check` flag on events ([08](08_world_model.md)).

## Orchestration
- **LangGraph** agent (MIT): conditional edges = "narrate from state" vs "fetch a frame/clip
  then narrate". ToolNode exposes a `get_frames(event_window, cameras)` tool that pulls the
  relevant clip for the clarifier.
- Reference scaffold: **NVIDIA VSS Blueprint** — CV emits structured metadata + tracking IDs
  (Set-of-Marks overlay) → VLM narrates. Our exact pattern, prebuilt.
- **vLLM** (Apache-2.0) to serve the open models if on-prem.

## Documented pitfalls → mitigations (designed-in)
1. **Spatial reasoning over (x,y) JSON degrades 42–80% as scenes get complex** and is
   serialization-sensitive → keep geometric questions ("foot on the line?", "who screened?")
   to the frame-clarifier, not the state-reasoner.
2. **Fine-grained actions / shot type / fouls need pixels** (even pixel-native models ~50% on
   fine-grained sports) → those events always fetch frames.
3. **VLMs are weak at precise temporal localization** (~38% GPT-4o on TemporalBench) → don't
   ask the model "exactly when"; the CV event timestamps are authoritative.
4. **Cascade error**: tracker/ID mistakes become the LLM's ground truth → **per-entity
   decomposition** (narrate per player, not a dumped table) + confidence-aware phrasing;
   faithfulness does **not** improve with model size, so don't just buy a bigger model.

## Production
- **Volume**: event stream + rolling possession context, not per-frame state.
- **Cost/latency**: state-reasoning is cheap; frame fetches are the expensive path, so
  event-gating directly controls spend.
- **Fine-tune later**: a small open model (7–32B) fine-tuned on basketball event→text is
  cheaper, lower-latency, and *more faithful* than a frontier API for templated narration;
  reserve the frontier API for color/edge cases. (Follow-on; v1 prompts an API.)
- **Live (follow-on)**: streaming "when-to-speak" gating (MMDuet / VideoLLM-online /
  StreamingVLM patterns). v1 is batch/post-game.

## Output
Possession-/event-level descriptive narration, each line grounded in a CV event + (optionally)
a confirming frame, with the player(s) and outcome named.
