#!/usr/bin/env python3
"""End-to-end pipeline demo on one 4-camera window (the whole thing, tied together).

Reuses an already-fused world-state (players on their correct A/B teams) and adds the last two
stages live: full-court MOTION ball -> possession/pass/turnover EVENTS -> VLM play-by-play.
Prints a client-readable summary and saves runs/tracking/<tag>_demo_e2e.json.

  python scripts/demo_e2e.py                                  # e6fba750, 47:12 window
  python scripts/demo_e2e.py --worldstate runs/tracking/e6_worldstate.json \
      --clips-glob 'data/clips/e6fba750_{ang}_47_12.mp4' --ref FL --tag e6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worldstate", default="runs/tracking/e6_worldstate_v4.json")
    ap.add_argument("--clips-glob", default="data/clips/e6fba750_{ang}_47_12.mp4",
                    help="clip path template with {ang} for FL/FR/NL/NR")
    ap.add_argument("--calib", default="configs/calib")
    ap.add_argument("--ref", default="FL")
    ap.add_argument("--tag", default="e6")
    ap.add_argument("--no-vlm", action="store_true", help="skip the Gemini narration stage")
    ap.add_argument("--no-audio-sync", action="store_true",
                    help="EXPLICITLY fuse unsynced (audit 2026-07-02: FR is ~-13 frames; never skip silently)")
    ap.add_argument("--ball", choices=("none", "motion"), default="none",
                    help="'motion' = EXPERIMENTAL motion-fusion ball (audit: follows players, "
                         "not the ball); default derives events honestly without a ball")
    a = ap.parse_args()

    os.chdir(REPO)
    try:
        from dotenv import load_dotenv
        load_dotenv(".env")
    except ImportError:
        pass
    key = "" if a.no_vlm else os.environ.get("GOOGLE_API_KEY", "")

    from uball_cc.fusion.ball_fuse import multicam_ball_trace
    from uball_cc.fusion.events import derive_events

    print(f"=== END-TO-END DEMO — {a.tag} ({a.ref} ref, 4 cameras) ===\n", flush=True)
    ws = json.loads(Path(a.worldstate).read_text())
    teams = Counter(p.get("team") for p in ws["players"])
    print(f"[1] PLAYERS (cross-camera fusion): {ws['n_global_ids']} global IDs | teams {dict(teams)}", flush=True)

    clips = {ang: a.clips_glob.format(ang=ang) for ang in ("FL", "FR", "NL", "NR")}
    ball_xy: dict[int, tuple] = {}
    if a.ball == "motion":
        trace = multicam_ball_trace(clips, a.calib, ref=a.ref, audio_sync=not a.no_audio_sync,
                                    zone={"FL": 0.6, "FR": 0.6, "NL": 1.0, "NR": 1.0})
        ball_xy = {int(k): tuple(v) for k, v in trace.items()}
        print(f"[2] BALL (motion fusion, EXPERIMENTAL): {len(ball_xy)} frames "
              "(audit: follows players — don't trust possession)", flush=True)
    else:
        print("[2] BALL: none (honest default; pass --ball motion for the experimental tracker)",
              flush=True)
    for fr in ws["frames"]:
        fr["ball"] = ball_xy.get(fr["frame"])
    ws["ball"] = {str(k): list(v) for k, v in ball_xy.items()}

    events = derive_events(ws, ball_by_frame=ball_xy or None)
    ws["events"] = events
    s = events["summary"]
    print(f"[3] EVENTS: {s['n_events']} total | {s['n_passes']} passes | {s['n_turnovers']} turnovers", flush=True)
    for e in [e for e in events["events"] if e["event"] in ("possession", "pass", "turnover")][:10]:
        if e["event"] == "possession":
            print(f"      t={e['t_sec']:>5}s  possession  {e['player']}", flush=True)
        else:
            print(f"      t={e['t_sec']:>5}s  {e['event']:9} {e.get('from')} -> {e.get('to')}", flush=True)

    print("\n[4] VLM PLAY-BY-PLAY (grounded on the identities + events) ...", flush=True)
    if not key:
        print("      (no GOOGLE_API_KEY / --no-vlm — skipping narration)", flush=True)
    else:
        from uball_cc.pipeline.vlm import narrate
        res = narrate(clips[a.ref], ws, api_key=key)
        print(f"      model {res['model']} | grounded={res['used_grounding']}", flush=True)
        print(f"\n      SUMMARY: {res['summary']}", flush=True)
        for pl in res.get("play_by_play", [])[:12]:
            print(f"        • {pl.get('description', '')}", flush=True)
        if res.get("caveats"):
            print(f"      caveats: {res['caveats']}", flush=True)
        out = REPO / f"runs/tracking/{a.tag}_demo_e2e.json"
        out.write_text(json.dumps({"players": ws["players"], "events": events, "narration": res}, indent=2))
        print(f"\n      saved -> {out}", flush=True)

    print("\n=== DEMO COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
