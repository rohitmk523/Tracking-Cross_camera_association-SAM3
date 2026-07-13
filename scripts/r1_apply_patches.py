#!/usr/bin/env python3
"""R1-full stage 3 — apply foreign-segment patches to the identity dumps:
move box entries from the wrong stream's file to the right stream's file for
the patched (cam, frame-range), then the event chain re-runs on clean streams.

  .venv/bin/python scripts/r1_apply_patches.py --game c2a354fe [--dry]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from game_meta import GAME_CHUNKS

FPS = 29.97


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    game = a.game
    patches = json.loads((REPO / f"runs/r1_samples_{game[:3]}/patches.json").read_text())
    moved = 0
    for tag in GAME_CHUNKS[game]:
        t0f = round(float(tag.split("_")[0]) * FPS)
        t1f = t0f + round(float(tag.split("_")[1]) * FPS)
        tdir = REPO / (f"runs/events_fg_{tag}" if game == "e6fba750"
                       else f"runs/events_fg_{game[:3]}_{tag}")
        for p in patches:
            if p["gf1"] < t0f or p["gf0"] >= t1f:
                continue
            src = tdir / f"{game}_{tag}__{p['sid']}__{p['ang']}.json"
            dst = tdir / f"{game}_{tag}__{p['new']}__{p['ang']}.json"
            if not src.exists():
                continue
            sdoc = json.loads(src.read_text())
            ddoc = (json.loads(dst.read_text()) if dst.exists()
                    else {"player": f"#{p['new'][1:]}", "cam": p["ang"],
                          "hybrid": True, "frames": {}})
            lo, hi = p["gf0"] - t0f, p["gf1"] - t0f
            take = [fr for fr in list(sdoc["frames"])
                    if lo <= int(fr) <= hi and sdoc["frames"][fr].get("present")]
            for fr in take:
                ddoc["frames"][fr] = sdoc["frames"].pop(fr)
            moved += len(take)
            if not a.dry and take:
                src.write_text(json.dumps(sdoc))
                dst.write_text(json.dumps(ddoc))
    print(f"{'DRY: would move' if a.dry else 'moved'} {moved} frame-boxes "
          f"across {len(patches)} patches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
