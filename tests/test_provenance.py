"""Tests for frame provenance parsing + game resolution (drives the splits)."""
from __future__ import annotations

import pytest

from uball_cc.data.provenance import (
    ALIASES,
    GameInfo,
    parse_stem,
    resolve_game,
    validate_aliases,
)

GAMES = {
    "e6fba750": GameInfo("e6fba750", "train", "2026-03-18"),
    "c2a354fe": GameInfo("c2a354fe", "test", "2026-03-19"),
    "d0a9faef": GameInfo("d0a9faef", "train", "2026-04-17"),
}


def test_parse_stem_angle():
    assert parse_stem("e6_NL_f01404") == ("e6", "NL")
    assert parse_stem("d0a9faef_NR_t000655.2_abc_rim") == ("d0a9faef", "NR")
    assert parse_stem("frame_01410_make_LEFT") == ("frame", None)


def test_resolve_alias_and_prefix():
    assert resolve_game("e6", GAMES).gid8 == "e6fba750"     # alias
    assert resolve_game("e6p", GAMES).gid8 == "e6fba750"    # perfect-set alias
    assert resolve_game("c2a", GAMES).gid8 == "c2a354fe"    # gid8-prefix
    assert resolve_game("d0a9faef", GAMES).gid8 == "d0a9faef"  # exact gid8
    assert resolve_game("frame", GAMES) is None             # old-facility, unknown


def test_ambiguous_short_prefix_is_rejected_not_misattributed():
    # 'cc5deb39' and 'cc1710c4' both start with 'cc' -> a 2-char key must NOT
    # silently resolve to one of them (review #14).
    games = {
        "cc5deb39": GameInfo("cc5deb39", "train", "2026-05-19"),
        "cc1710c4": GameInfo("cc1710c4", "train", "2026-05-19"),
    }
    assert resolve_game("cc", games) is None        # too short -> rejected
    assert resolve_game("cc5deb39", games).gid8 == "cc5deb39"  # full gid8 ok


def test_validate_aliases_catches_stale_alias():
    # every ALIASES target must exist in the real games map (review #15)
    full = {gid8: GameInfo(gid8, "train", None)
            for gid8 in set(ALIASES.values())}
    validate_aliases(full)                          # all present -> ok
    with pytest.raises(ValueError):
        validate_aliases({})                        # nothing present -> raise
