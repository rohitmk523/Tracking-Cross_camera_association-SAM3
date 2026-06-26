"""End-to-end pipeline: 4 angles -> detect -> track -> fuse -> VLM play-by-play.

Phased (inspect/fix each stage) + no-checkpoint runs (CV-only and full). See server.py.
"""
from .jobs import ANGLES, PHASES, Job, JobStore
from .phases import RUNNERS, run_detect, run_fuse, run_track, run_vlm

__all__ = ["Job", "JobStore", "PHASES", "ANGLES", "RUNNERS",
           "run_detect", "run_track", "run_fuse", "run_vlm"]
