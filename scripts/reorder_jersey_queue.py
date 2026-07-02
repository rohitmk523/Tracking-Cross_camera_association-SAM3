#!/usr/bin/env python3
"""Reorder the jersey-annotation queue to INTERLEAVE all games/cameras (2026-07-02 audit fix).

items.json was alphabetical -> game-grouped, so sequential labeling burned every click on ONE
game/camera (213/213 labels were 0fa23810/NL) and the next 400 would cover ~4 of 25 games.
Cross-game generalization -- the whole point of the trained recogniser -- can be neither
trained nor measured that way.

New order: round-robin one crop per (game, cam) group, hashlib-ordered within each group.
Deterministic (no RNG), idempotent, and safe: labels are keyed by crop NAME, so existing
labels survive any reorder; the annotator resumes at the first unlabeled item of the new order.

Also purges invalid empty labels ({"number":"","box":null}) so those items return to the queue.

  python scripts/reorder_jersey_queue.py --pool data/jersey_pool
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def _group(crop: str) -> str:
    parts = crop.split("_")              # <gid8>_<cam>_<event>_t<sec>_f<idx>_p<n>.jpg
    return f"{parts[0]}_{parts[1]}" if len(parts) > 1 else parts[0]


def _h(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def interleave(items: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for it in items:
        groups.setdefault(_group(it["crop"]), []).append(it)
    for g in groups.values():
        g.sort(key=lambda it: _h(it["crop"]))
    order = sorted(groups, key=_h)
    rounds = max(len(v) for v in groups.values())
    return [groups[g][j] for j in range(rounds) for g in order if j < len(groups[g])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/jersey_pool")
    a = ap.parse_args()
    pool = Path(a.pool)

    items = json.loads((pool / "items.json").read_text())
    lab_path = pool / "labels.json"
    labels = json.loads(lab_path.read_text()) if lab_path.exists() else {}

    # purge invalid empties (they blocked items from the unlabeled queue)
    bad = [k for k, v in labels.items()
           if isinstance(v, dict) and not str(v.get("number", "")).strip()]
    for k in bad:
        del labels[k]
    if bad:
        lab_path.write_text(json.dumps(labels))

    new_items = interleave(items)
    assert {i["crop"] for i in new_items} == {i["crop"] for i in items}, "reorder lost items"
    (pool / "items.json").write_text(json.dumps(new_items))

    # coverage forecast: what do the operator's NEXT labels touch?
    unlabeled = [i["crop"] for i in new_items if i["crop"] not in labels]
    nxt = Counter(_group(c).split("_")[0] for c in unlabeled[:400])
    print(f"reordered {len(new_items)} items across {len({_group(i['crop']) for i in items})} "
          f"(game,cam) groups | purged {len(bad)} empty labels | {len(labels)} labels kept")
    print(f"next 400 unlabeled now cover {len(nxt)} games (was ~4): "
          f"{dict(sorted(nxt.items(), key=lambda kv: -kv[1])[:8])} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
