#!/usr/bin/env python3
"""Launch the end-to-end pipeline server (4 angles -> detect -> track -> fuse -> VLM).

  python scripts/pipeline_server.py                 # http://127.0.0.1:8000
  python scripts/pipeline_server.py --port 8020

Needs the `baseline` + `tracking` + `fusion` + `vlm` deps (RF-DETR, trackers, OSNet,
scipy, google-genai) and GOOGLE_API_KEY for the VLM phase.
"""
import argparse

from uball_cc.pipeline.server import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    run(host=a.host, port=a.port)
