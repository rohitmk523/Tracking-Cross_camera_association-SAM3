"""FastAPI pipeline server: upload 4 angles -> detect -> track -> fuse -> vlm.

Phased (inspect/fix each stage) + two no-checkpoint runs (CV-only, and full incl. VLM).
Phases run in background threads; poll GET /api/jobs/{id} for status + per-phase summaries.

  python scripts/pipeline_server.py            # http://127.0.0.1:8000
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .jobs import ANGLES, PHASES, JobStore
from .phases import RUNNERS

UI_HTML = Path(__file__).resolve().parent / "ui.html"
STORE = JobStore()
_running: set[str] = set()
_lock = threading.Lock()

_DESCRIPTION = """
## What this is

A system that turns **four fixed-camera videos of a basketball game** into **descriptive,
timestamped play-by-play** — automatically. You give it the raw footage; it works out
*who* is on the court (team + jersey number), *where* everyone is, and *what happened*
(shots, assists, rebounds, steals, blocks, turnovers), and writes it up like a commentator.

It runs on the **same 4-camera rig the courts already have** — far-left (FL), far-right
(FR), near-left (NL), near-right (NR) — and is built so **every step can be inspected and
improved on its own**.

## How it works — the four stages

The pipeline runs in four stages; each one takes the previous stage's output and adds to it:

| # | Stage | In plain English | Input → Output |
|---|---|---|---|
| 1 | **Detect** | Find every player, referee and ball in each camera, in every frame. | video → boxes per frame |
| 2 | **Track** | Follow each person over time, and label their **team** (from jersey colour) + a unique **appearance signature** (ReID). | boxes → tracked players + team |
| 3 | **Fuse** | Combine all four cameras onto **one top-down map of the court**, merging the four views of each player into **one identity**. | 4 camera tracks → one *world-state* (who / where) |
| 4 | **Narrate (VLM)** | Hand that world-state + the video to **Google Gemini**, which watches the play and writes the **play-by-play + summary**, naming players by their real identity. | world-state + video → play-by-play |

**Why the split?** Stages 1–3 are pure computer vision — they own *who* and *where*. The
VLM (stage 4) owns *what happened*. Because the VLM is handed the already-verified
identities, it never has to guess who a player is; it just describes the action. This is
what makes the narration accurate.

## The two-step workflow

Using it is always two steps:

1. **Create a job** — upload the four camera videos (`POST /api/jobs`). This only *stores*
   the videos and returns a **job id**. Nothing is processed yet, so it returns instantly.
2. **Run the pipeline on that job** — either the whole thing in one call, or one stage at a
   time so you can look at each result.

The split means you upload the (large) videos **once**, then run / re-run / tune the
processing as much as you like **without re-uploading**.

## Two ways to run it

- **`run_full`** — the complete pipeline (detect → track → fuse → **narrate**). Four videos
  in, **play-by-play out**. This is the end-to-end result.
- **`run_cv`** — everything **except** the narration. Produces the computer-vision
  **world-state** (the top-down map of identified players) with **no AI-language step** —
  useful to judge the tracking/fusion on its own.
- **A single stage** (`/phase/{stage}`) — run just one stage to inspect or improve it.

All runs happen **in the background** (the call returns immediately); poll
`GET /api/jobs/{id}` to watch each stage go `pending → running → done`.

## What you get back

- After **fuse / run_cv** → a **world-state**: the list of players (global id, team, jersey,
  court position) and their position every frame (the data behind the top-down radar).
- After **narrate / run_full** → a **play-by-play**: a list of plays — each with a start/end
  timestamp, a description, the players involved, the action and outcome — plus a
  one-paragraph game summary.
"""

TAGS = [
    {"name": "Jobs", "description": "**Step 1 + monitoring.** Create a job by uploading the four "
     "camera angles, list past jobs, check a job's live status, and download each stage's output."},
    {"name": "Pipeline", "description": "**Step 2, in one call.** Run the whole pipeline on a job — "
     "`run_full` for the complete result (with play-by-play), or `run_cv` for the computer-vision "
     "world-state only (no AI narration)."},
    {"name": "Phases", "description": "**Step 2, stage by stage.** Run a single stage on its own — to "
     "see exactly what it produces, or to re-run it after a change. The 'improve each factor one by "
     "one' loop."},
]

app = FastAPI(title="uball cross-camera pipeline", version="0.1.0",
              description=_DESCRIPTION, openapi_tags=TAGS)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _phase_kwargs(ph: str, cfg: dict) -> dict:
    if ph == "detect":
        return {k: cfg[k] for k in ("max_frames", "stride", "threshold") if k in cfg}
    if ph == "vlm":
        m = {"narration_angle": cfg.get("narration_angle"), "api_key": cfg.get("api_key"),
             "model": cfg.get("vlm_model"), "fps": cfg.get("vlm_fps")}
        return {k: v for k, v in m.items() if v is not None}
    if ph == "fuse":
        return {k: cfg[k] for k in ("ref_angle", "max_assoc_dist", "audio_sync") if k in cfg}
    return {}


def _run(jid: str, phases: list[str], cfg: dict) -> None:
    try:
        for ph in phases:
            job = STORE.load(jid)
            st = job.phases[ph]
            st.status, st.started, st.error = "running", time.time(), ""
            STORE.save(job)
            try:
                summary = RUNNERS[ph](job, STORE, **_phase_kwargs(ph, cfg))
                st.status, st.summary = "done", summary
            except Exception as e:  # noqa: BLE001
                st.status, st.error = "error", f"{type(e).__name__}: {e}"[:600]
                st.finished = time.time()
                STORE.save(job)
                return
            st.finished = time.time()
            STORE.save(job)
    finally:
        with _lock:
            _running.discard(jid)


def _launch(jid: str, phases: list[str], cfg: dict) -> None:
    with _lock:
        if jid in _running:
            raise HTTPException(409, "job already has a phase running")
        _running.add(jid)
    threading.Thread(target=_run, args=(jid, phases, cfg), daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return UI_HTML.read_text()


@app.get("/api/jobs", tags=["Jobs"], summary="List all jobs")
def list_jobs() -> dict:
    """Every analysis job created so far (newest first), with its per-stage status."""
    return {"jobs": STORE.list()}


@app.post("/api/jobs", tags=["Jobs"], summary="Create a job — upload the 4 camera angles")
async def create_job(
    FL: UploadFile | None = File(None), FR: UploadFile | None = File(None),
    NL: UploadFile | None = File(None), NR: UploadFile | None = File(None),
    narration_angle: str | None = Form(None),
) -> JSONResponse:
    """**Start here.** Upload the game-window videos for the four fixed cameras — far-left
    (FL), far-right (FR), near-left (NL), near-right (NR). At least one is required; the
    full cross-camera fusion needs all four. `narration_angle` (default FL) is the camera
    the VLM narrates over.

    This call **only stores the videos** — no processing happens yet, so it returns
    instantly with a **job id** that every other endpoint uses. Send it as
    `multipart/form-data` (one file field per angle).

    **Example response** — note every stage starts at `pending`:

    ```json
    {
      "id": "e6fba750-1c2d-4f8a-9b3e-...",
      "angles": ["FL", "FR", "NL", "NR"],
      "narration_angle": "FL",
      "phases": {
        "detect": {"status": "pending"},
        "track":  {"status": "pending"},
        "fuse":   {"status": "pending"},
        "vlm":    {"status": "pending"}
      }
    }
    ```
    """
    uploads = {a: f for a, f in (("FL", FL), ("FR", FR), ("NL", NL), ("NR", NR)) if f is not None}
    if not uploads:
        raise HTTPException(400, "upload at least one angle (FL/FR/NL/NR)")
    angle_bytes = {a: await f.read() for a, f in uploads.items()}
    job = STORE.create(angle_bytes, narration_angle=narration_angle)
    return JSONResponse(job.to_dict())


@app.get("/api/jobs/{jid}", tags=["Jobs"], summary="Job status + per-stage results")
def get_job(jid: str) -> dict:
    """The live status of each stage (`pending` / `running` / `done` / `error`) plus its
    result summary (detection counts, track counts, fused player count, play count). **Poll
    this** while a stage runs to watch progress.

    Each finished stage carries a short `summary` so you can sanity-check it at a glance —
    e.g. *fuse* reports how many distinct players it resolved and the average on-court at
    once; *vlm* reports how many plays it wrote.

    **Example (mid-run — detect/track done, fuse running):**

    ```json
    {
      "id": "e6fba750-...",
      "phases": {
        "detect": {"status": "done",    "summary": {"FL": {"frames": 600, "detections": 7421}, "...": {}}},
        "track":  {"status": "done",    "summary": {"FL": {"tracks": 18}, "...": {}}},
        "fuse":   {"status": "running"},
        "vlm":    {"status": "pending"}
      }
    }
    ```

    When `fuse` finishes its summary looks like
    `{"n_global_ids": 16, "n_frames": 600, "avg_players_per_frame": 15.8}`, and `vlm` like
    `{"n_plays": 11, "model": "gemini-3.5-flash", "summary": "..."}`.
    """
    try:
        return STORE.load(jid).to_dict()
    except FileNotFoundError as e:
        raise HTTPException(404, f"job {jid} not found") from e


_BODY_PHASE = {"requestBody": {"required": False, "content": {"application/json": {"example": {
    "api_key": "AIza...your-gemini-key...", "vlm_model": "gemini-3.5-flash", "vlm_fps": 1.0,
    "stride": 6, "threshold": 0.25, "max_frames": 600}}}},
    "responses": {"200": {"description": "Stage launched in the background.", "content":
                  {"application/json": {"example": {"job": "e6fba750-...", "launched": ["vlm"]}}}}}}
_BODY_CV = {"requestBody": {"required": False, "content": {"application/json": {"example": {
    "stride": 6, "max_frames": 600, "threshold": 0.25, "audio_sync": True}}}},
    "responses": {"200": {"description": "CV pipeline launched in the background.", "content":
                  {"application/json": {"example": {"job": "e6fba750-...",
                                                    "launched": ["detect", "track", "fuse"]}}}}}}
_BODY_FULL = {"requestBody": {"required": False, "content": {"application/json": {"example": {
    "api_key": "AIza...your-gemini-key...", "vlm_model": "gemini-3.5-flash", "vlm_fps": 1.0,
    "narration_angle": "FL", "stride": 6, "audio_sync": True}}}},
    "responses": {"200": {"description": "Full pipeline launched in the background.", "content":
                  {"application/json": {"example": {"job": "e6fba750-...",
                                                    "launched": ["detect", "track", "fuse", "vlm"]}}}}}}


@app.post("/api/jobs/{jid}/phase/{phase}", tags=["Phases"],
          summary="Run ONE stage (detect | track | fuse | vlm)", openapi_extra=_BODY_PHASE)
async def run_phase(jid: str, phase: str, request: Request) -> dict:
    """Run a single pipeline stage on its own (runs in the background — poll
    `GET /api/jobs/{id}`), so each can be inspected and improved independently:

    - **detect** — find players/referees/ball in each camera (output: per-frame boxes).
    - **track** — link them into tracklets + assign **team** and **ReID** per player.
    - **fuse** — combine the 4 cameras into one **top-down world-state** (one id per player).
    - **vlm** — narrate the video, grounded on that world-state → **play-by-play + summary**.
      (Send the Gemini key in the JSON body: `{"api_key": "..."}`; optional `vlm_model`,
      `vlm_fps`.) Earlier stages must be `done` first (each reads the previous one's output)."""
    if phase not in PHASES:
        raise HTTPException(400, f"unknown phase {phase}; one of {PHASES}")
    cfg = await _json_body(request)
    _launch(jid, [phase], cfg)
    return {"job": jid, "launched": [phase]}


@app.post("/api/jobs/{jid}/run_cv", tags=["Pipeline"],
          summary="Run the full CV pipeline (no VLM)", openapi_extra=_BODY_CV)
async def run_cv(jid: str, request: Request) -> dict:
    """One call, no stops: **detect → track → fuse**. Produces the cross-camera
    **world-state** (every player's global identity, team and court position) — the
    computer-vision output, **without** any Gemini/VLM call. Use this to evaluate the
    tracking + fusion quality on its own.

    Runs in the background; poll `GET /api/jobs/{id}`, then download the result with
    `GET /api/jobs/{id}/artifact/fuse`. The **world-state** looks like:

    ```json
    {
      "n_global_ids": 16,
      "players": [
        {"global_id": 3, "team": "A", "jersey": 7,  "court_xy": [812.4, 640.1]},
        {"global_id": 5, "team": "B", "jersey": 23, "court_xy": [1190.7, 533.9]}
      ],
      "frames": [
        {"frame": 0, "tracks": [
          {"global_id": 3, "team": "A", "jersey": 7,  "court_xy": [...]},
          {"global_id": 5, "team": "B", "jersey": 23, "court_xy": [...]}
        ]}
      ]
    }
    ```

    `players` is the roster (one row per resolved identity); `frames` is the per-frame
    top-down position of each — the data behind the radar view. Court coordinates are in
    centimetres on a corner-origin court (≈2144 × 1426 cm)."""
    _launch(jid, ["detect", "track", "fuse"], await _json_body(request))
    return {"job": jid, "launched": ["detect", "track", "fuse"]}


@app.post("/api/jobs/{jid}/run_full", tags=["Pipeline"],
          summary="Run the FULL pipeline (with VLM play-by-play)", openapi_extra=_BODY_FULL)
async def run_full(jid: str, request: Request) -> dict:
    """The complete system in one call: **detect → track → fuse → VLM**. Four camera
    angles in, **descriptive play-by-play + a summary** out, with players named by the
    identity the CV pipeline resolved. Needs a Gemini API key in the JSON body:
    `{"api_key": "..."}` (optional `vlm_model`, `vlm_fps`, `narration_angle`).

    Runs in the background; poll `GET /api/jobs/{id}`, then download with
    `GET /api/jobs/{id}/artifact/vlm`. The **play-by-play** looks like:

    ```json
    {
      "summary": "Blue controls the half-court and gets a clean look from the wing.",
      "play_by_play": [
        {
          "description": "Blue #7 initiates with a hard drive and kicks to the wing.",
          "players": ["Blue #7", "Blue #23"],
          "action": "drive-and-kick",
          "court_location": "right wing",
          "assisted_by": "Blue #7",
          "outcome": "made 3-pointer",
          "timestamp": "00:12",
          "confidence": 0.82
        }
      ],
      "caveats": "Jersey numbers on the far camera were partly unreadable."
    }
    ```

    Each play carries who was involved, the action, where on the court, the outcome, an
    advisory timestamp and a confidence. The player labels (e.g. *Blue #7*) come from the
    fused world-state, so the narration names real identities rather than guessing."""
    _launch(jid, list(PHASES), await _json_body(request))
    return {"job": jid, "launched": list(PHASES)}


@app.get("/api/jobs/{jid}/artifact/{phase}", tags=["Jobs"],
         summary="Download a stage's output (JSON)")
def artifact(jid: str, phase: str, angle: str | None = None):
    """The raw output of a finished stage, as a downloadable JSON file:

    - **detect** (`?angle=FL`) → per-frame boxes: `{"angle","n_frames","detections":[[{"b":[x1,y1,x2,y2],"s":score,"c":class}]]}` (class `0`=player, `1`=referee, `2`=ball).
    - **track** (`?angle=FL`) → tracklets with team + ReID: `{"angle","tracks":[{"track_id","team","jersey","frames":[...]}]}`.
    - **fuse** → the top-down **world-state** (`players` roster + per-`frames` positions; see *run_cv*).
    - **vlm** → the **play-by-play + summary** (see *run_full*).

    The per-camera stages (detect / track) produce one file **per angle** — pass
    `?angle=FL` (or FR / NL / NR). `fuse` and `vlm` are global (one file each)."""
    base = STORE.root / jid
    path = {"detect": base / "detect" / f"{angle or 'FL'}.json",
            "track": base / "track" / f"{angle or 'FL'}.json",
            "fuse": base / "fuse" / "worldstate.json",
            "vlm": base / "vlm" / "narration.json"}.get(phase)
    if not path or not path.exists():
        raise HTTPException(404, f"no {phase} artifact for {jid}")
    return FileResponse(str(path), media_type="application/json")


async def _json_body(request: Request) -> dict:
    try:
        return await request.json()
    except Exception:  # noqa: BLE001
        return {}


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)


_ = ANGLES  # re-exported for the UI/help
