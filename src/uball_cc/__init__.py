"""uball_cc -- general-purpose cross-camera basketball tracking system.

Layered pipeline (see docs/01_architecture.md):
  detection -> tracking -> cross-camera fusion -> world model -> VLM narration.

This package currently implements the Week-1 foundation: dataset consolidation
(`uball_cc.data`) and the evaluation harness (`uball_cc.eval`). Later stages are
added per docs/12_roadmap.md.
"""

__version__ = "0.1.0"
