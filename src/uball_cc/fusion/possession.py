"""Image-space possession (docs/08, post-audit design): a held ball overlaps its holder's
box in the CAMERA IMAGE, so possession is attributed in pixels — the flat-court projection
error for elevated balls never enters.

Measured against operator ground truth: this attribution is precision-clean (zero wrong
possessions on both scored windows) with coverage limited by ball VISIBILITY — hence the
production default pairs it with the near-basket appearance detector (ball AP 0.80) and
says "no ball data" elsewhere rather than guessing.
"""
from __future__ import annotations

BOX_EXPAND = 0.20            # a held ball sits at the body edge; expand boxes a little
NEAR_FRAC = 0.5              # no containment: accept nearest box centre within this * box_h


def attribute_ball(ball_xy, players) -> int | None:
    """players: [(track_id, (x1,y1,x2,y2))]. Containment first (smallest box wins =
    nearest player), else nearest box centre within NEAR_FRAC of its box height."""
    bx, by = ball_xy
    containing = []
    for tid, (x1, y1, x2, y2) in players:
        ex, ey = (x2 - x1) * BOX_EXPAND, (y2 - y1) * BOX_EXPAND
        if x1 - ex <= bx <= x2 + ex and y1 - ey <= by <= y2 + ey:
            containing.append((abs((x2 - x1) * (y2 - y1)), tid))
    if containing:
        return min(containing)[1]
    best = None
    for tid, (x1, y1, x2, y2) in players:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        d = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
        lim = NEAR_FRAC * (y2 - y1)
        if d <= lim and (best is None or d < best[0]):
            best = (d, tid)
    return best[1] if best else None


def holder_stream_from_votes(votes: dict[int, list], gmap: dict[int, dict],
                             search: int = 5) -> dict[int, int]:
    """votes: {ref_frame: [(conf, cam, local_track_id)]} -> {ref_frame: holder global_id}.
    Highest-confidence ball wins the frame; the (cam, local) pair maps to the fused
    identity via the per-frame members map (searching +-search frames for a fresh one)."""
    out: dict[int, int] = {}
    for f, vs in votes.items():
        _, cam, lid = max(vs)
        for df in sorted(range(-search, search + 1), key=abs):
            gid = gmap.get(f + df, {}).get((cam, lid))
            if gid is not None:
                out[f] = gid
                break
    return out
