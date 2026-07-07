"""Image-space possession attribution (fusion/possession.py) — the shipped honest partial."""
from __future__ import annotations

from uball_cc.fusion.possession import attribute_ball, holder_stream_from_votes


def test_containment_picks_smallest_box():
    players = [(1, (0, 0, 400, 800)), (2, (100, 200, 300, 600))]     # 2 is inside 1's area
    assert attribute_ball((200, 400), players) == 2                  # nearest player wins


def test_near_fallback_within_half_box_height():
    players = [(1, (100, 100, 200, 300))]                            # box_h=200 -> limit 100
    assert attribute_ball((150, 470), players) is None               # 270 below centre: too far
    assert attribute_ball((230, 250), players) == 1                  # ~85px from centre: ok


def test_no_players_abstains():
    assert attribute_ball((100, 100), []) is None


def test_holder_stream_maps_via_members():
    votes = {5: [(0.9, "NR", 3)], 6: [(0.4, "NL", 7), (0.8, "NR", 3)]}
    gmap = {5: {("NR", 3): 11}, 6: {("NR", 3): 11, ("NL", 7): 4}}
    hs = holder_stream_from_votes(votes, gmap)
    assert hs == {5: 11, 6: 11}                                       # highest-conf ball wins


def test_holder_stream_searches_nearby_frames():
    votes = {10: [(0.9, "NR", 3)]}
    gmap = {12: {("NR", 3): 11}}                                      # members only 2 frames later
    assert holder_stream_from_votes(votes, gmap) == {10: 11}
