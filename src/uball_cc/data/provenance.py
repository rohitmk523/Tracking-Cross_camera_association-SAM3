"""Filename -> (game, angle) provenance + resolution against configs/games.json.

Our annotated frames are named like:
    e6_NL_f01404.jpg            (4Cam / E6 Demo: short alias prefix)
    c2a_FL_f00012.jpg
    e6p_NR_f00096.jpg           (E6 "perfect" set, same game as e6)
    d0a9faef_NR_t000655.2_*.jpg (Near-angle rim set: full gid8 prefix)

So the first underscore-token is a game key (a gid8, a gid8-prefix, or a short
alias) and the second (when present) is the camera angle.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ANGLES = ("FL", "FR", "NL", "NR")

# Short aliases used in legacy frame names -> gid8 (configs/games.json key).
ALIASES = {
    "e6": "e6fba750",
    "e6p": "e6fba750",   # the "perfect" re-annotation of the same game
    "c2a": "c2a354fe",
}


@dataclass(frozen=True)
class GameInfo:
    gid8: str
    split: str            # docs/14 whole-game split: train|val|test|fresh|unknown
    date: str | None
    court: str = "court-a"   # all current footage is court-a (one venue)


def load_games(games_json: Path) -> dict[str, GameInfo]:
    """gid8 -> GameInfo from configs/games.json working_games."""
    data = json.loads(Path(games_json).read_text())
    out: dict[str, GameInfo] = {}
    for g in data.get("working_games", []):
        out[g["gid8"]] = GameInfo(gid8=g["gid8"], split=g.get("split", "unknown"),
                                  date=g.get("date"))
    return out


def parse_stem(stem: str) -> tuple[str, str | None]:
    """Return (game_key, angle) parsed from a frame filename stem."""
    parts = stem.split("_")
    game_key = parts[0]
    angle = next((p for p in parts[1:4] if p in ANGLES), None)
    return game_key, angle


# Shortest key allowed to prefix-match a gid8. Below this, ambiguity is too
# likely (e.g. 'cc' matches both cc5deb39 and cc1710c4) -> force such names
# through the explicit ALIASES map instead (review #14).
_MIN_PREFIX_LEN = 3


def resolve_game(game_key: str, games: dict[str, GameInfo]) -> GameInfo | None:
    """Resolve a frame's game key to a known game (gid8). None if unknown/ambiguous."""
    key = ALIASES.get(game_key, game_key)
    if key in games:
        return games[key]
    if len(key) < _MIN_PREFIX_LEN:
        return None
    # gid8-prefix match (e.g. legacy 'c2a', or truncations). Ambiguous -> None.
    cands = [g for gid8, g in games.items() if gid8.startswith(key)]
    return cands[0] if len(cands) == 1 else None


def validate_aliases(games: dict[str, GameInfo]) -> None:
    """Every ALIASES target must be a real game (review #15) -- a stale alias
    would misattribute every frame of a game into the wrong split silently."""
    bad = {k: v for k, v in ALIASES.items() if v not in games}
    if bad:
        raise ValueError(f"ALIASES point to unknown gid8(s): {bad} -- "
                         "fix uball_cc.data.provenance.ALIASES or configs/games.json.")


_STEM_RE = re.compile(r"\.(jpg|jpeg|png)$", re.IGNORECASE)


def is_image(name: str) -> bool:
    return bool(_STEM_RE.search(name))
