"""Tracklet-level association (fusion/tracklets.py): merge fragments, never real people."""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")
from uball_cc.fusion.tracklets import apply_merges, merge_map  # noqa: E402


def _ws(frames):
    players = {}
    for fr in frames:
        for t in fr["tracks"]:
            players.setdefault(t["global_id"], {"global_id": t["global_id"],
                                                "team": t.get("team"), "jersey": None})
    return {"frames": frames, "players": list(players.values()), "n_global_ids": len(players)}


def _fr(f, *tracks):
    return {"frame": f, "tracks": [{"global_id": g, "court_xy": [x, y], "team": tm}
                                   for g, x, y, tm in tracks]}


def test_fragment_across_gap_merges():
    """One player: id1 for 30 frames, 20-frame dropout, reappears as id2 nearby."""
    frames = [_fr(f, (1, 500 + 3 * f, 700, "A")) for f in range(30)] \
        + [_fr(f, (2, 590 + 3 * (f - 50), 700, "A")) for f in range(50, 80)]
    m = merge_map(frames)
    assert m == {2: 1}
    out = apply_merges(_ws(frames), m)
    assert out["n_global_ids"] == 1
    assert {t["global_id"] for fr in out["frames"] for t in fr["tracks"]} == {1}


def test_different_teams_never_merge():
    frames = [_fr(f, (1, 500, 700, "A")) for f in range(30)] \
        + [_fr(f, (2, 510, 700, "B")) for f in range(40, 70)]
    assert merge_map(frames) == {}


def test_far_apart_never_merges():
    frames = [_fr(f, (1, 500, 700, "A")) for f in range(30)] \
        + [_fr(f, (2, 1900, 200, "A")) for f in range(40, 70)]
    assert merge_map(frames) == {}


def test_temporal_overlap_never_merges():
    """Two ids alive at the same time are two PEOPLE, however close."""
    frames = [_fr(f, (1, 500, 700, "A"), (2, 520, 700, "A")) for f in range(60)]
    assert merge_map(frames) == {}


def test_ref_only_merges_with_ref():
    frames = [_fr(f, (1, 500, 700, "REF")) for f in range(30)] \
        + [_fr(f, (2, 510, 700, "A")) for f in range(40, 70)] \
        + [_fr(f, (3, 515, 705, "REF")) for f in range(80, 110)]
    m = merge_map(frames)
    assert m.get(3) == 1 and 2 not in m


def test_transitive_chain_merges_in_order():
    """id1 -> gap -> id2 -> gap -> id3: all one player."""
    frames = [_fr(f, (1, 500, 700, "A")) for f in range(20)] \
        + [_fr(f, (2, 520, 700, "A")) for f in range(30, 50)] \
        + [_fr(f, (3, 540, 700, "A")) for f in range(60, 80)]
    m = merge_map(frames)
    assert m == {2: 1, 3: 1}
