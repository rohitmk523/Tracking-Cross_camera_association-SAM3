"""Pipeline job model + on-disk store.

A job = one game window across the 4 camera angles, run through the phases
(detect -> track -> fuse -> vlm). Each phase reads the previous phase's artifacts
and writes its own under data/pipeline_jobs/<id>/, so phases are independently
runnable and inspectable ("fix each phase one by one").
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

PHASES = ("detect", "track", "fuse", "vlm")
ANGLES = ("FL", "FR", "NL", "NR")
ROOT = Path(__file__).resolve().parents[3] / "data" / "pipeline_jobs"


@dataclass
class PhaseState:
    status: str = "pending"          # pending | running | done | error
    started: float | None = None
    finished: float | None = None
    summary: dict = field(default_factory=dict)
    error: str = ""


@dataclass
class Job:
    id: str
    angles: list[str]                # angles actually uploaded (subset of ANGLES)
    narration_angle: str             # which angle the VLM narrates over
    created: float
    phases: dict[str, PhaseState] = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return ROOT / self.id

    def to_dict(self) -> dict:
        return {**asdict(self), "phases": {k: asdict(v) for k, v in self.phases.items()}}


class JobStore:
    def __init__(self, root: Path = ROOT):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, angle_to_video: dict[str, bytes | Path], narration_angle: str | None = None) -> Job:
        jid = uuid.uuid4().hex[:12]
        job = Job(id=jid, angles=sorted(angle_to_video),
                  narration_angle=narration_angle or _first(angle_to_video, ("FL", "FR")),
                  created=time.time(),
                  phases={p: PhaseState() for p in PHASES})
        (job.dir / "videos").mkdir(parents=True, exist_ok=True)
        for ang, v in angle_to_video.items():
            dst = job.dir / "videos" / f"{ang}.mp4"
            if isinstance(v, (bytes, bytearray)):
                dst.write_bytes(v)
            else:
                dst.write_bytes(Path(v).read_bytes())
        self.save(job)
        return job

    def save(self, job: Job) -> None:
        (job.dir / "job.json").write_text(json.dumps(job.to_dict(), indent=2))

    def load(self, jid: str) -> Job:
        d = json.loads((self.root / jid / "job.json").read_text())
        d["phases"] = {k: PhaseState(**v) for k, v in d.get("phases", {}).items()}
        return Job(**d)

    def list(self) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("*/job.json")):
            try:
                out.append(json.loads(p.read_text()))
            except (OSError, ValueError):
                continue
        return sorted(out, key=lambda j: j.get("created", 0), reverse=True)

    def video(self, job: Job, angle: str) -> Path:
        return job.dir / "videos" / f"{angle}.mp4"


def _first(d, prefs):
    for p in prefs:
        if p in d:
            return p
    return next(iter(d))
