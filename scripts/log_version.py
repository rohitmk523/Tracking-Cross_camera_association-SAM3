#!/usr/bin/env python3
"""Version ledger: one structured record per tracking-version per window.

Everything a future plan could need: config fingerprint, per-camera fragmentation,
fused identity metrics, roster vs the official plays roster, churn indicators, and
(when available) SAM3 ground-truth scores. Appends to runs/tracking/version_ledger.jsonl
and pretty-writes runs/tracking/ledger/<version>_<game>_<tag>.json.

  python scripts/log_version.py --version v2_1 --game e6fba750 --tag 44_60 \
      --worldstate runs/tracking/e6fba750_44_60_worldstate_v2_1.json \
      --notes "min_cos 0.68, base-radius 50"
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--worldstate", required=True)
    ap.add_argument("--notes", default="")
    ap.add_argument("--sam3-scores", default=None, help="optional parity report json")
    a = ap.parse_args()

    ws = json.loads(Path(a.worldstate).read_text())
    frames = ws["frames"]
    n_frames = len(frames)
    life: Counter = Counter()
    first_seen: dict[int, int] = {}
    coast = 0
    total_rows = 0
    for fr in frames:
        for t in fr["tracks"]:
            g = t["global_id"]
            life[g] += 1
            first_seen.setdefault(g, fr["frame"])
            coast += 1 if t.get("coasting") else 0
            total_rows += 1
    ppf = st.median([len(f["tracks"]) for f in frames[15:-15]]) if n_frames > 40 else None

    per_cam = {}
    for ang in ANGLES:
        p = REPO / f"runs/tracking/{a.game}_{ang}_{a.tag}_teams.json"
        if p.exists():
            d = json.loads(p.read_text())
            tids = {r["track_id"] for r in d["tracks"] if r["class_id"] in (0, 1)}
            jer = Counter(r["jersey"] for r in d["tracks"] if r.get("jersey") is not None)
            per_cam[ang] = {"tracklets": len(tids), "jersey_numbers": sorted(set(jer))}

    named = sorted([str(p_["team"]), p_["jersey"]] for p_ in ws["players"]
                   if p_.get("jersey") is not None)
    # roster check vs the official plays roster snapshot when available
    roster_hit = None
    snap = REPO / "data/plays/windows_snapshot.json"
    if snap.exists():
        d = json.loads(snap.read_text())
        official = {int(k) for k in d.get(f"roster_{a.game}", {})}
        if official:
            ours = {int(j) for _, j in named}
            roster_hit = {"read": sorted(ours), "on_roster": sorted(ours & official),
                          "hallucinated": sorted(ours - official)}

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None

    rec = {
        "version": a.version, "game": a.game, "tag": a.tag, "commit": commit,
        "notes": a.notes,
        "fused": {
            "n_global_ids": ws["n_global_ids"],
            "stable_ids": sum(1 for c in life.values() if c >= 0.5 * n_frames),
            "ghost_ids": sum(1 for c in life.values() if c < 0.1 * n_frames),
            "late_born_ids": sum(1 for g, f0 in first_seen.items()
                                 if f0 > n_frames * 0.5),
            "people_per_frame_median": ppf,
            "coasting_rate": round(coast / max(1, total_rows), 4),
        },
        "per_camera": per_cam,
        "named_players": named,
        "roster_check": roster_hit,
        "sam3_scores": (json.loads(Path(a.sam3_scores).read_text())
                        if a.sam3_scores and Path(a.sam3_scores).exists() else None),
        "artifacts": {
            "worldstate": a.worldstate,
            "demo_video": f"runs/tracking/demo_track_{a.version}_{a.game}.mp4",
        },
    }
    led = REPO / "runs/tracking/version_ledger.jsonl"
    with open(led, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    det = REPO / f"runs/tracking/ledger/{a.version}_{a.game}_{a.tag}.json"
    det.parent.mkdir(parents=True, exist_ok=True)
    det.write_text(json.dumps(rec, indent=1))
    print(f"ledger <- {a.version} {a.game} {a.tag}: ids {rec['fused']['n_global_ids']} "
          f"stable {rec['fused']['stable_ids']} late {rec['fused']['late_born_ids']} "
          f"named {len(named)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
