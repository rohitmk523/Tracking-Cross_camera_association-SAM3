#!/usr/bin/env python3
"""Launch the player/referee/ball annotation UI (FastAPI).

  python scripts/annotate.py                       # http://127.0.0.1:8000
  python scripts/annotate.py --port 8001           # if 8000 is taken
  UBALL_ANNOT_POOL=data/annotate_pool python scripts/annotate.py
"""
import argparse

from uball_cc.annotation.server import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    run(host=a.host, port=a.port)
