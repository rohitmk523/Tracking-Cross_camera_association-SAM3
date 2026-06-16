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


def resolve_game(game_key: str, games: dict[str, GameInfo]) -> GameInfo | None:
    """Resolve a frame's game key to a known game (gid8). None if unknown."""
    key = ALIASES.get(game_key, game_key)
    if key in games:
        return games[key]
    # gid8-prefix match (e.g. legacy 'e6'/'c2a' before aliasing, or truncations)
    cands = [g for gid8, g in games.items() if gid8.startswith(key)]
    if len(cands) == 1:
        return cands[0]
    return None


_STEM_RE = re.compile(r"\.(jpg|jpeg|png)$", re.IGNORECASE)


def is_image(name: str) -> bool:
    return bool(_STEM_RE.search(name))
