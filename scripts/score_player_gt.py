#!/usr/bin/env python3
"""Score a tracking version against operator per-player ground truth.

GT (data/gt_players) = for one physical player, per frame per camera, the detection
box the operator clicked. Boxes are version-independent (cached detections), so this
scores every worldstate ever produced on the same window.

Per player, against a version's fused worldstate:
  observed_coverage  — of GT frames where the operator selected >=1 camera, fraction
                       where the version has a fused track within match-cm of the
                       GT court position (the player existed; did we track him?)
  id_purity          — dominant global-id share among matched frames (1.0 = one
                       identity the whole minute; the operator's churn complaint)
  n_identities       — distinct gids that ever matched (ideal: 1)
  switches           — dominant-id changes over the matched timeline

  python scripts/score_player_gt.py --game e6fba750 --tag 44_60 \
      --worldstate runs/tracking/e6fba750_44_60_worldstate_v2_2.json --version v2.2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ANGLES = ("FL", "FR", "NL", "NR")
# sync offsets for e6 44_60 measured earlier (FL ref); recomputed if absent
OFFS = {"e6fba750_44_60": {"FL": 0, "FR": -11, "NL": -1, "NR": -1}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--worldstate", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--match-cm", type=float, default=150.0)
    a = ap.parse_args()

    from uball_cc.fusion.homography import load_calib, project_pixels

    key = f"{a.game}_{a.tag}"
    offs = OFFS.get(key)
    if offs is None:
        from uball_cc.fusion.audiosync import audio_offset_seconds
        offs = {"FL": 0}
        ref = str(REPO / f"data/clips/{a.game}_FL_{a.tag}.mp4")
        for ang in ("FR", "NL", "NR"):
            s, _ = audio_offset_seconds(ref, str(REPO / f"data/clips/{a.game}_{ang}_{a.tag}.mp4"))
            offs[ang] = int(round(s * 29.97))

    dets = {}
    for ang in ANGLES:
        z = np.load(REPO / f"runs/dets_cache/{a.game}_{ang}_{a.tag}_small_1280_t0.25.dets.npz")
        m = {}
        for b, s, c, f in zip(z["boxes"], z["scores"], z["classes"], z["frame_idx"]):
            if int(c) in (0, 1) and float(s) >= 0.25:
                m.setdefault(int(f), []).append([float(v) for v in b])
        dets[ang] = m
    calib = {ang: load_calib(str(REPO / f"configs/calib/{ang}.json")) for ang in ANGLES}

    gt = json.loads((REPO / f"data/gt_players/{key}.json").read_text())
    ws = json.loads(Path(a.worldstate).read_text())
    ours = {fr["frame"]: [(t["global_id"], t["court_xy"]) for t in fr["tracks"]]
            for fr in ws["frames"]}

    report = {}
    for player, rec in gt.items():
        sel_frames = {int(f): sels for f, sels in rec.get("frames", {}).items()
                      if sels and str(f) in rec.get("approved", {})}
        if len(sel_frames) < 30:
            continue
        matched = {}                                   # frame -> gid
        n_obs = 0
        for f, sels in sorted(sel_frames.items()):
            pts = []
            for ang, bi in sels.items():
                boxes = dets[ang].get(f + offs[ang], [])
                if bi < len(boxes):
                    b = boxes[bi]
                    (cx, cy), = project_pixels([((b[0] + b[2]) / 2, b[3])], calib[ang])
                    pts.append((cx, cy))
            if not pts:
                continue
            n_obs += 1
            gx, gy = float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))
            cands = ours.get(f, [])
            if not cands:
                continue
            d, gid = min((np.hypot(x - gx, y - gy), g) for g, (x, y) in cands)
            if d <= a.match_cm:
                matched[f] = gid
        if not n_obs:
            continue
        gids = Counter(matched.values())
        dom = gids.most_common(1)[0] if gids else (None, 0)
        # switches: changes in gid along the matched timeline
        seq = [matched[f] for f in sorted(matched)]
        switches = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
        report[player] = {
            "gt_frames": n_obs,
            "observed_coverage": round(len(matched) / n_obs, 3),
            "id_purity": round(dom[1] / max(1, len(matched)), 3),
            "dominant_gid": dom[0],
            "n_identities": len(gids),
            "switches": switches,
        }

    out = {"version": a.version, "window": key, "players": report}
    print(json.dumps(out, indent=1))
    led = REPO / f"runs/tracking/ledger/playerGT_{a.version}_{key}.json"
    led.parent.mkdir(parents=True, exist_ok=True)
    led.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
