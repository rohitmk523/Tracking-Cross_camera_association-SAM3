"""VLM narration via the VLM_Basketball backend (the canonical narrator we're using).

Rather than re-implement, the VLM phase imports VLM_Basketball/backend/app's
`narrate_clip` — native-video Gemini, the rich PlayLine schema, events/context
grounding — and calls it with our narration-angle clip + the cross-camera fusion
world-state injected as grounding `context`. Keeps VLM_Basketball the single source
of truth (set VLM_BASKETBALL_BACKEND to override the path).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

VLM_BACKEND = Path(os.environ.get(
    "VLM_BASKETBALL_BACKEND",
    "/Users/rohitkale/Cellstrat/GitHub_Repositories/VLM_Basketball/backend"))


def _narrate_clip():
    if not (VLM_BACKEND / "app" / "gemini.py").exists():
        raise RuntimeError(
            f"VLM_Basketball backend not found at {VLM_BACKEND}. "
            "Set VLM_BASKETBALL_BACKEND to the repo's backend/ dir.")
    if str(VLM_BACKEND) not in sys.path:
        sys.path.insert(0, str(VLM_BACKEND))
    from app.gemini import narrate_clip  # VLM_Basketball's canonical narrator  # noqa: PLC0415
    return narrate_clip


def _grounding(world_state: dict | None) -> dict | None:
    """Map our fused world-state -> VLM_Basketball `context` (identities the VLM trusts)."""
    if not world_state:
        return None
    players = world_state.get("players") or []
    return {
        "source": "cross-camera CV fusion (trust these identities; read actions from video)",
        "tracked_players": [
            {"id": p.get("global_id"), "team": p.get("team"), "jersey": p.get("jersey"),
             "court_xy_cm": p.get("court_xy")}
            for p in players
        ],
    }


def narrate(video_path, world_state: dict | None = None, *, model: str | None = None,
            fps: float | None = None, api_key: str | None = None,
            media_resolution: str = "medium") -> dict:
    """Grounded narration via VLM_Basketball. Returns a JSON-friendly dict."""
    narrate_clip = _narrate_clip()
    context = _grounding(world_state)
    # docs/08: let the VLM narrate OVER the deterministic CV event stream (possession/pass/
    # turnover/transition), not just positions — the structured "what happened" the pipeline built.
    events = None
    ev = (world_state or {}).get("events")
    if isinstance(ev, dict) and ev.get("events"):
        events = ev["events"]
    result = narrate_clip(str(video_path), api_key=api_key, model=model, fps=fps,
                          media_resolution=media_resolution, events=events, context=context)
    d = result.model_dump()
    return {
        "model": d["model"], "fps": d["fps"], "media_resolution": d["media_resolution"],
        "used_grounding": d["used_events"], "summary": d["narration"]["summary"],
        "play_by_play": d["narration"]["play_by_play"], "caveats": d["narration"].get("caveats", ""),
        "rendered": d["rendered"], "usage": d["usage"],
    }
