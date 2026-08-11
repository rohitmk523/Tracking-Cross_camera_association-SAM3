# Possession Pipeline → Ingestion Integration

**Target repo:** `gopro-automation-linux` (Flask on the Jetson + AWS Batch)
**Source repo:** `Tracking-Cross_camera_association-SAM3` (this one)
**Written:** 2026-08-11
**Status:** design doc — nothing built on the ingestion side yet

---

## 0. Read this first: the one design rule

The possession logic is **changing weekly right now** (holder accuracy went
24% → ~60% in a single day, and the referee fix expected to land is another
+15). The integration must therefore be built so that **improving the CV
never requires a code change in `gopro-automation-linux`.**

That is the entire point of this document. It is achieved by putting a
**versioned contract** between the two repos:

```
 THIS REPO  ────ships────>  a container image + an output JSON schema
                              ↑                        ↓
                          (the contract)          (the contract)
                              ↓                        ↑
 INGESTION  ────does────>  submit a Batch job + read that JSON
```

The ingestion side knows **three things only**: which env vars to set, where
the output lands, and what shape it has. It knows nothing about detectors,
trackers, hold hysteresis, or roster handling.

**Consequence:** when we improve possession logic, shipping it to production
is `docker push` + bump one env var (`POSSESSION_IMAGE_TAG`). No PR against
the Flask app. That is the "few prompts from our side" requirement, made
concrete.

---

## 1. Where this slots into the existing flow

The ingestion pipeline already has a working CV path (V1 shot detection).
**We mirror it exactly** rather than inventing a second pattern:

```
Jetson → S3 raw chapters → ffmpeg-nvenc transcode (existing AWS Batch)
   → 4 × 1080p in S3 (FL/FR/NL/NR)
   → 5-min cron → POST /api/cv/dispatch-pending      [EXISTS]
        ├─ 2 × cv-fusion GPU jobs  → cv-merge        [EXISTS: shots]
        └─ 1 × cv-possession GPU job                 [NEW: this doc]
   → artifacts in S3 + Firebase markers + Supabase
```

Deliberate design choices, and why:

| choice | rationale |
|---|---|
| **One job, not two** | Possession needs all 4 angles fused together (cross-camera identity is the whole point). The A/B side split used by shot detection would destroy it. |
| **Separate job def from `cv-fusion`** | Different resource profile and cadence. Possession can fail without taking shot detection down, and vice versa. |
| **Same queue, same dispatcher module, same marker convention** | Ops staff already know this pattern. One runbook, not two. |
| **Artifact-first output** | Possession is a *continuous timeline*, not discrete events. It does not fit the `actionType: score_added` log shape. See §3. |

---

## 2. The input contract (what the container receives)

Exactly mirrors `cv_batch_dispatch.CVBatchDispatcher.submit_fusion_job`, so
the new dispatcher is a near-copy of the existing one.

### Required environment variables

| var | example | meaning |
|---|---|---|
| `GAME_ID` | `026f49bb-f9ed-4353-b42f-b0e8b587e35e` | Supabase game UUID |
| `INPUTS_BUCKET` | `uball-videos-production` | bucket holding the 1080p angles |
| `FL_S3_KEY` | `court-a/2026-04-15/.../_FL.mp4` | far-left angle |
| `FR_S3_KEY` | … | far-right angle |
| `NL_S3_KEY` | … | near-left angle |
| `NR_S3_KEY` | … | near-right angle |
| `RESULTS_BUCKET` | `uball-cv-results` | where output is written |
| `RESULT_S3_PREFIX` | `cv-possession/court-a/2026-04-15/026f49bb/` | output prefix |
| `MODELS_BUCKET` | `uball-cv-models` | detector weights |
| `MODEL_VERSION` | `v1` | weights version |
| `ROSTER_JSON` | *(inline JSON or S3 key)* | **see §2.1 — this one is critical** |

### Optional environment variables (all have safe defaults in the image)

| var | default | meaning |
|---|---|---|
| `CAMERA_OFFSETS` | auto-measure | per-angle frame offsets, e.g. `{"FL":0,"FR":-8,"NL":2,"NR":-1}` |
| `CALIBRATION_KEY` | by-venue lookup | court calibration for this rig/era |
| `POSSESSION_PROFILE` | `balanced` | `fast` \| `balanced` \| `accurate` — see §5 |
| `WINDOW_START_S` / `WINDOW_DUR_S` | whole game | process a sub-window (for reprocessing/debug) |
| `EMIT_RING_VIDEO` | `false` | also render the 4-up ring video |

**Everything else is baked into the image.** Hold thresholds, release grace,
detector confidences, OCR stride, fusion parameters — none of these are
exposed. They change frequently and the ingestion side must never have an
opinion about them.

### 2.1 The roster is a hard input, not a nicety

We learned this the expensive way: a missing player in the roster silently
capped holder accuracy at 64% on one game, and a duplicate placeholder entry
in another split one real player into two identities. **A wrong roster
degrades output without any error being raised.**

Requirements:

- Roster must be supplied per game, from the ingestion system's own source of
  truth (Supabase `game_players` or equivalent) — **not** a file in our repo.
- Schema:
  ```json
  {
    "team1_color": "Black", "team2_color": "White",
    "players": [{"num": 2, "name": "Trent Wasser", "team": 1}, ...]
  }
  ```
- The container **validates and fails loudly** if: two entries share
  (num, team); any entry has an empty/placeholder name; or fewer than 8
  players are present. A hard failure is correct — a silently bad roster
  costs more than a retried job.
- If the roster is unavailable, the job should **not** run. Emit
  `status: "skipped", reason: "roster_unavailable"` rather than produce a
  low-quality timeline that looks valid.

---

## 3. The output contract (the part that must stay stable)

One artifact, written to `s3://{RESULTS_BUCKET}/{RESULT_S3_PREFIX}/possession.json`.

```json
{
  "schema_version": "1.0",
  "game_id": "026f49bb-f9ed-4353-b42f-b0e8b587e35e",
  "generated_at": "2026-08-11T10:04:00Z",
  "pipeline_version": "possession-2026.08.11",
  "model_version": "v1",
  "fps": 29.97,
  "frame_base": 0,
  "status": "ok",

  "segments": [
    {"start": 17987, "end": 18042, "holder": "n1W", "player_num": 1,
     "team": 2, "player_name": "Ana Li", "confidence": 0.81},
    {"start": 18043, "end": 18051, "holder": null, "reason": "flight"},
    {"start": 18052, "end": 18120, "holder": null, "reason": "referee"}
  ],

  "quality": {
    "frames_total": 107892,
    "frames_with_holder": 64000,
    "ball_coverage": 0.69,
    "mean_confidence": 0.74,
    "roster_players_seen": 16,
    "roster_players_expected": 18,
    "warnings": ["2 roster players never detected"]
  }
}
```

### Rules that make this contract survive our changes

1. **`segments` is the product.** Contiguous, non-overlapping, covers every
   frame. `holder: null` means nobody has the ball — a real answer, not
   missing data.
2. **`reason` on null segments is advisory** (`flight` / `referee` /
   `loose` / `unknown`). Consumers must tolerate new values appearing —
   we will add them. Never `switch` on it exhaustively.
3. **Additive changes only** within a major version. New fields may appear at
   any time; the ingestion side must ignore unknown keys.
4. **`schema_version` bumps major only on a breaking change**, which we will
   announce and dual-write for one release.
5. **`quality` is how the ingestion decides whether to trust the run** —
   see §6. It is not decoration.
6. **`status`** is one of `ok` | `degraded` | `skipped`. `degraded` means
   usable but flagged (e.g. a camera died mid-game — real occurrence: one
   venue loses NR at the 26-minute mark).

### Optional second artifact

If `EMIT_RING_VIDEO=true`, also writes `ring.mp4` (the 4-up grid with the
holder ring). Useful for QA and for the highlight pipeline. Not required.

---

## 4. What to build on the ingestion side

Four pieces, in dependency order. Each mirrors something that already exists.

### 4.1 `possession_batch_dispatch.py` — copy of `cv_batch_dispatch.py`

Same class shape, same boto3 config, same logging, same tag convention.
Differences: one job instead of three, four angle keys instead of two, no
`dependsOn` chain.

```python
class PossessionBatchDispatcher:
    def submit_possession_job(self, *, game_keys: GameVideoKeys,
                              roster: dict) -> Dict[str, Any]:
        env = [
            {"name": "GAME_ID",         "value": game_keys.game_uuid},
            {"name": "INPUTS_BUCKET",   "value": self.inputs_bucket},
            {"name": "FL_S3_KEY",       "value": game_keys.angle_keys["FL"]},
            {"name": "FR_S3_KEY",       "value": game_keys.angle_keys["FR"]},
            {"name": "NL_S3_KEY",       "value": game_keys.angle_keys["NL"]},
            {"name": "NR_S3_KEY",       "value": game_keys.angle_keys["NR"]},
            {"name": "RESULTS_BUCKET",  "value": self.results_bucket},
            {"name": "RESULT_S3_PREFIX","value": self.build_result_prefix(game_keys)},
            {"name": "MODELS_BUCKET",   "value": self.models_bucket},
            {"name": "MODEL_VERSION",   "value": self.model_version},
            {"name": "ROSTER_JSON",     "value": json.dumps(roster)},
        ]
        ...  # submit_job, tags={"stage": "possession", ...}
```

Reuse `GameVideoKeys` and `truncate_uuid` as-is — do not fork them.

**Requires all four angles.** `has_all_four_angles()` already exists; if it
returns False, skip the game and record why. Possession cannot be computed
from a partial rig.

### 4.2 Job definition — `cv-possession-job-def.json`

Register alongside the existing two in `deploy/batch-job-defs/`.

| setting | value | note |
|---|---|---|
| image | `…/uball-cv-possession:${POSSESSION_IMAGE_TAG}` | **tag comes from env** — this is the upgrade lever |
| GPU | 1 | |
| vCPU / RAM | 8 / 32 GB | 4 angles held in memory |
| timeout | **5400s** (90 min) | full game, measured ~3 GPU-h on A10G today, dropping |
| retries | 2 | |
| queue | `cv-shot-detection-queue` | reuse; no new compute env |

The timeout is deliberately generous. Our current full-game cost is ~3 GPU-h
and falling as speed work lands; a too-tight timeout will silently kill runs
as we change the profile.

### 4.3 Flask endpoint + cron

Extend the existing dispatch endpoint rather than adding a new one:

```
POST /api/cv/dispatch-pending
     {"firebase_game_id": "...", "stages": ["shots", "possession"]}
```

Default `stages` to `["shots"]` so behaviour is unchanged until possession is
explicitly enabled. Same 5-minute cron, same guard logic.

**Firebase markers** (mirroring `cv_dispatched_at` etc.):

```
possession_dispatched_at   ISO8601
possession_job_id          Batch jobId
possession_status          "ok" | "degraded" | "skipped" | "failed"
possession_result_key      s3://.../possession.json
possession_schema_version  "1.0"
disablePossession          bool — operator bypass, mirrors disableCv
```

Skip a game if `possession_dispatched_at` is set (unless a specific
`firebase_game_id` was passed — same override the CV path uses).

### 4.4 Consumption

Deliberately **not** through `plays_sync`. Possession is a timeline, not a
set of scoring events, and forcing it into `actionType` logs would lose the
information and pollute the plays table.

Start with the artifact only:

1. Store `possession_result_key` on the Firebase game.
2. Consumers (highlights, overlays, analytics, the annotation tool) read the
   JSON directly from S3.

Derived aggregates (time of possession per player, touches, passes) are a
**second step** — trivial to compute from `segments`, and best done once the
accuracy bar is met. Don't build it yet.

---

## 5. Profiles — the one tuning knob the ingestion may set

`POSSESSION_PROFILE` selects a bundle of internal settings. Ingestion picks
an intent; we decide what it means, and we can change the meaning without a
code change on their side.

| profile | what it means today | when |
|---|---|---|
| `fast` | unified detector, no pose, OCR stride 2 | previews, cost-sensitive |
| `balanced` *(default)* | unified detector, pose stride 2, OCR stride 1 | normal ingestion |
| `accurate` | full pose, OCR stride 1, all fusion passes | disputed games, QA |

Measured today (3-minute window, laptop): `fast` is 3.0× quicker than
`accurate` but costs ~5 points of holder accuracy. On A10G a full game is
roughly 1 GPU-h (`fast`) to 3 GPU-h (`accurate`).

These numbers will move. **That is exactly why the ingestion side names an
intent instead of setting parameters.**

---

## 6. Quality gating — do not publish a bad run

The `quality` block exists so the ingestion can refuse output automatically.
Recommended initial thresholds:

```python
q = result["quality"]
if result["status"] == "skipped":
    mark_failed(reason=result.get("reason"))
elif q["ball_coverage"] < 0.40 or q["mean_confidence"] < 0.50:
    mark_needs_review()          # store artifact, don't surface to users
elif q["roster_players_seen"] < 0.7 * q["roster_players_expected"]:
    mark_needs_review()          # roster/venue mismatch — the expensive bug
else:
    publish()
```

That third check is the one that would have caught the roster bug we spent
half a day finding. Keep it.

Alarm on a `needs_review` streak, mirroring `UBall-CV-NeedsReviewStreak`.

---

## 7. How an improvement ships (the whole point)

When possession logic improves in this repo:

1. We build and push `uball-cv-possession:2026.08.18`.
2. We tell you one line: *"bump `POSSESSION_IMAGE_TAG` to `2026.08.18`;
   schema unchanged."*
3. You bump the env var. Done.

If the output schema needs a **breaking** change, we:
- bump `schema_version` major,
- dual-write both shapes for one release,
- give you the diff and a migration note.

We will not change field meanings in place. A consumer written against
`1.0` keeps working against `1.x` forever.

**Rollback** is the same lever in reverse: point the tag at the previous
image. Because the contract is versioned and the artifact is immutable in S3,
rolling back never corrupts already-published data.

---

## 8. Build order (suggested)

| phase | work | done when |
|---|---|---|
| **0** | We publish `uball-cv-possession:v1` + a sample `possession.json` | ingestion can develop against a real artifact without running a GPU |
| **1** | `possession_batch_dispatch.py` + job def registered | a manual `submit_job` produces `possession.json` in S3 |
| **2** | `/api/cv/dispatch-pending` accepts `stages`, writes Firebase markers | one game dispatches end-to-end from the cron |
| **3** | Quality gate + `needs_review` + CloudWatch alarm | bad runs stop before users see them |
| **4** | Derived aggregates → Supabase | only after the accuracy bar is met |

Phases 0–2 are the integration. Phases 3–4 can lag.

---

## 9. Open items we owe you before phase 1

Tracked on our side, listed here so nothing is assumed:

1. **The container doesn't exist yet.** Today the pipeline is a chain of
   scripts driven by `game_meta.py`. Packaging it behind the env-var contract
   in §2 is our work, roughly a day.
2. **Camera offsets and calibration** are currently per-game constants in our
   repo. For arbitrary ingested games these must be auto-measured (we have
   `sync_anchor_sweep.py` and `refit_calibration.py`) or supplied per venue.
   **This is the biggest unknown for new footage** — a wrong calibration
   silently produces a 5-metre cross-camera error, and it is the number-one
   onboarding risk for a new or re-placed rig.
3. **Accuracy is ~60%, not production-grade.** Measured against human ground
   truth across three games. Integrate the plumbing now; gate user-visible
   features behind the quality block until the number is where we want it.

We are not blocked on any of these to start phases 0–1 on your side — the
contract in §2 and §3 is stable enough to build against today.
