#!/usr/bin/env python3
"""Sweep FusionEngine params on a synced 4-cam window and rank configs (audit follow-up).

The shipped TUNED values (cluster_dist=600cm!) were tuned while the cameras were believed
unsynced — 6m grouping absorbs sync+calibration error but is LARGER than player spacing,
forcing ambiguous associations. With strict audio-sync in place this sweep re-tunes on the
aligned timeline. No GT exists yet, so configs are ranked by PROXY quality (known truth on
this window: ~10 players + 2 refs => ppf median ~12, ids floor 13, no subs):

  ids close to 13 | stable(>=50%) ~12-13 | ghosts(<10%)=0 | few frames >13 or <11

  python scripts/tune_fusion.py            # -> runs/tracking/fusion_tune.json + table
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CAMS = ("FL", "FR", "NL", "NR")
ZONE = {"FL": 0.6, "FR": 0.6, "NL": 1.0, "NR": 1.0}
TEAM_CAMS = {"NL", "NR"}
FPS = 29.97
N_PEOPLE = 13                      # 10 players + refs on this window (no subs in 12s)


def _load_obs(tracks_tpl: str, calib_dir: str, clip_tpl: str, ref: str, region_pad: float):
    """-> {frame: [Observation]} on the synced ref timeline (built once, reused per config)."""
    from uball_cc.fusion.audiosync import audio_offset_seconds
    from uball_cc.fusion.court import LENGTH, WIDTH
    from uball_cc.fusion.engine import Observation
    from uball_cc.fusion.homography import calib_hull, in_calib_region, load_calib, project_pixels
    from uball_cc.tracking import Track

    by_frame: dict[int, list] = defaultdict(list)
    for ang in CAMS:
        data = json.loads(Path(tracks_tpl.format(ang=ang)).read_text())
        tracks = [Track.from_record(r) for r in data["tracks"]]
        calib = load_calib(f"{calib_dir}/{ang}.json")
        hull = calib_hull(calib)
        court = project_pixels([t.foot_xy for t in tracks], calib)
        off = 0
        if ang != ref:
            off_s, _ = audio_offset_seconds(clip_tpl.format(ang=ref), clip_tpl.format(ang=ang))
            off = int(round(off_s * FPS))
        rp = Path(tracks_tpl.format(ang=ang).replace("_teams.json", "_reid.npz"))
        reid = ({int(i): v for i, v in zip(z["ids"], z["emb"])}
                if rp.exists() and (z := np.load(rp)) is not None else {})
        for t, c in zip(tracks, court):
            if not (-300 <= c[0] <= LENGTH + 300 and -300 <= c[1] <= WIDTH + 300):
                continue
            if not in_calib_region(c, hull, region_pad):
                continue
            team = t.team if (ang in TEAM_CAMS or t.team == "REF") else None
            by_frame[t.frame - off].append(Observation(
                ang, t.track_id, (float(c[0]), float(c[1])), team=team, jersey=None,
                reid=reid.get(t.track_id), score=t.score, zone_conf=ZONE.get(ang, 1.0)))
    return by_frame


def _metrics(per_frame_live: dict[int, list]) -> dict:
    frames = sorted(per_frame_live)
    core = frames[15:-15]
    ppf = [len(per_frame_live[f]) for f in core]
    life = Counter()
    for f in frames:
        for gid, *_ in per_frame_live[f]:
            life[gid] += 1
    n = len(frames)
    return {"ids": len(life),
            "stable": sum(1 for c in life.values() if c >= 0.5 * n),
            "ghosts": sum(1 for c in life.values() if c < 0.1 * n),
            "ppf_med": st.median(ppf), "ppf_mean": round(st.mean(ppf), 2),
            "over13": round(sum(1 for p in ppf if p > 13) / len(ppf), 3),
            "under11": round(sum(1 for p in ppf if p < 11) / len(ppf), 3)}


def _score(m: dict) -> float:
    """Lower is better. Penalise fragmentation, ghosts, and both count-error directions;
    reward stability. A proxy until the GT eval set exists."""
    return (0.4 * max(0, m["ids"] - N_PEOPLE) + 1.0 * m["ghosts"]
            + 8.0 * m["over13"] + 8.0 * m["under11"]
            + 1.0 * abs(m["ppf_med"] - 12) + 0.8 * max(0, 12 - m["stable"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="runs/tracking/e6fba750_{ang}_47_12_teams.json")
    ap.add_argument("--clips", default="data/clips/e6fba750_{ang}_47_12.mp4")
    ap.add_argument("--calib", default="configs/calib")
    ap.add_argument("--ref", default="FL")
    ap.add_argument("--region-pad", type=float, default=800.0)
    ap.add_argument("--out", default="runs/tracking/fusion_tune.json")
    a = ap.parse_args()

    from uball_cc.fusion.engine import TUNED, FusionEngine

    by_frame = _load_obs(a.tracks, a.calib, a.clips, a.ref, a.region_pad)
    frames = sorted(by_frame)
    print(f"loaded {sum(len(v) for v in by_frame.values())} obs over {len(frames)} synced frames")

    grid = {"cluster_dist": [250.0, 350.0, 450.0, 600.0],
            "gate_cost": [6.0, 7.0],
            "min_hits": [3, 4],
            "w_a": [0.0, 3.0]}                     # reid off vs on (audit: camera-biased)
    rows = []
    for cd, gc, mh, wa in itertools.product(*grid.values()):
        eng = FusionEngine(max_assoc_dist=max(600.0, cd), gate_cost=gc, w_t=TUNED["w_t"],
                           w_a=wa, min_hits=mh, cluster_dist=cd)
        live = {f: [(t.id,) for t in eng.step(f, by_frame[f])] for f in frames}
        m = _metrics(live)
        rows.append({"cluster_dist": cd, "gate_cost": gc, "min_hits": mh, "w_a": wa,
                     **m, "score": round(_score(m), 2)})
    rows.sort(key=lambda r: r["score"])
    hdr = ["cluster_dist", "gate_cost", "min_hits", "w_a", "ids", "stable", "ghosts",
           "ppf_med", "ppf_mean", "over13", "under11", "score"]
    print("  ".join(f"{h:>11}" for h in hdr))
    for r in rows[:12]:
        print("  ".join(f"{r[h]:>11}" for h in hdr))
    cur = next((r for r in rows if r["cluster_dist"] == TUNED["cluster_dist"]
                and r["gate_cost"] == TUNED["gate_cost"] and r["min_hits"] == TUNED["min_hits"]
                and r["w_a"] == TUNED["w_a"]), None)
    print(f"\ncurrent TUNED ranks #{rows.index(cur) + 1 if cur in rows else '?'} of {len(rows)}"
          f" (score {cur['score'] if cur else '?'} vs best {rows[0]['score']})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"grid": grid, "results": rows}, indent=2))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
