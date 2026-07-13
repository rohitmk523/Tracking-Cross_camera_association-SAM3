"""Per-game constants shared by the events pipeline scripts.

Offsets are inter-camera frame offsets (audio-sync residuals) measured once
per game on its GT minute; chunks are the ffmpeg slicing grid used by every
AWS prep/cache job for that game.
"""

GAME_OFFS = {
    "e6fba750": {"FL": 0, "FR": -11, "NL": -1, "NR": -1},
    "c2a354fe": {"FL": 0, "FR": 1, "NL": 2, "NR": -1},
}

GAME_CHUNKS = {
    "e6fba750": ("0_600", "600_600", "1200_600", "1800_600", "2400_600", "3000_345"),
    "c2a354fe": ("0_600", "600_600", "1200_600", "1800_600", "2400_600", "3000_597"),
}
