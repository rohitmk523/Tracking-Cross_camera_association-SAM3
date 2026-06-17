#!/usr/bin/env python3
"""Launch the player/referee/ball annotation UI (FastAPI).

  python scripts/annotate.py                       # http://127.0.0.1:8000
  UBALL_ANNOT_POOL=data/annotate_pool python scripts/annotate.py
"""
from uball_cc.annotation.server import run

if __name__ == "__main__":
    run()
