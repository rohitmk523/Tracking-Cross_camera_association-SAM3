#!/usr/bin/env python3
"""Set-of-marks overlay: draw each track's box + an EVOLVING label onto its own camera, so the
VLM grounds on the marks instead of hallucinating who's who (the video we hand Gemini).

The label folds identity in as we learn it -- NOT a separate number box (docs/05):
    A id7            (team + track id, always)
 -> A #22            (once the jersey recogniser fills track.jersey)
 -> A #22 Rohit      (once ids are matched to a roster: --roster {track_id:{jersey,name}})

Boxes come from the saved per-camera tracks JSON (track.py output) -- these are the camera's OWN
2D detections, so they sit on players correctly (unlike court->pixel projection on a partly
calibrated far camera). Reuses the render_frame label convention from tracking/run.py.

  python scripts/render_marks.py --tracks runs/tracking/e6fba750_FL_47_12_teams.json \
      --video data/clips/e6fba750_FL_47_12.mp4 --out runs/tracking/e6_FL_marks.mp4
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

TEAM_BGR = {"A": (0, 140, 255), "B": (255, 120, 40), "REF": (0, 255, 255)}  # match tracking/run.py
NONE_BGR = (170, 170, 170)


def _ident(t: dict, roster: dict, show_team: bool) -> tuple[str, str | None]:
    """Return (label, team). roster entry may override team/jersey/name (step-4 fused id-map).
    Label folds identity in as we learn it: id -> #jersey -> name (docs/05). Team is dropped when
    show_team is False -- far cameras (FL/FR) can't separate teams, so we don't feed the VLM a
    label we don't trust; near-cam voting / the fused id-map supplies team + jersey/name."""
    r = roster.get(str(t.get("track_id")), {})
    team = (r.get("team", t.get("team")) or None) if show_team else None
    tag = f"{team} " if team else ""
    jersey = r.get("jersey", t.get("jersey"))
    name = r.get("name")
    if name:
        return f"{tag}{name}", team
    if jersey is not None:
        return f"{tag}#{jersey}", team
    return f"{tag}id{t.get('track_id')}", team


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True, help="per-camera tracks JSON (track.py output)")
    ap.add_argument("--video", required=True, help="the matching source clip")
    ap.add_argument("--out", default="runs/tracking/marks.mp4")
    ap.add_argument("--roster", default=None,
                    help="optional {track_id:{team,jersey,name}} JSON overriding the label (step-4 id-map)")
    ap.add_argument("--no-team", action="store_true",
                    help="drop team from labels (use for far cams FL/FR where team is unreliable)")
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()

    import cv2

    data = json.loads(Path(a.tracks).read_text())
    tracks = data["tracks"] if isinstance(data, dict) else data
    roster = json.loads(Path(a.roster).read_text()) if a.roster else {}
    by_frame: dict[int, list] = defaultdict(list)
    for t in tracks:
        by_frame[t["frame"]].append(t)

    cap = cv2.VideoCapture(a.video)
    W, H = int(cap.get(3)), int(cap.get(4))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    fi, named = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for t in by_frame.get(fi, []):
            x1, y1, x2, y2 = (int(v) for v in t["box_xyxy"])
            lbl, team = _ident(t, roster, show_team=not a.no_team)
            col = TEAM_BGR.get(team, NONE_BGR)
            named += ("#" in lbl or "id" not in lbl)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            yt = max(14, y1 - 6)
            cv2.rectangle(frame, (x1, yt - 13), (x1 + 9 * len(lbl), yt + 3), (0, 0, 0), -1)
            cv2.putText(frame, lbl, (x1 + 2, yt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
        vw.write(frame)
        fi += 1
    cap.release()
    vw.release()
    print(f"wrote {fi} marked frames ({len(tracks)} boxes, {named} with #/name) -> {a.out}", flush=True)
    print("feed this to the VLM instead of the raw clip -> grounded narration, fewer hallucinations", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
