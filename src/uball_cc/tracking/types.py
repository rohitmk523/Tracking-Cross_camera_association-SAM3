"""Per-camera track types (Tracking->Fusion contract, docs/01 / docs/05).

A `Track` is one tracked detection in one camera at one frame, carrying the local
track id plus attribute slots (team / jersey / reid) that the attribute providers
fill in. Global identity is NOT decided here — that's cross-camera fusion (docs/06).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

CLASS_NAMES = {0: "player", 1: "referee", 2: "ball"}


@dataclass(frozen=True)
class Track:
    cam: str                                    # FL | FR | NL | NR
    frame: int
    track_id: int                               # LOCAL (per-camera) id
    box_xyxy: tuple[float, float, float, float]
    score: float
    class_id: int                               # 0=player 1=referee 2=ball
    team: str | None = None                     # "A" | "B" | "REF" (attribute stage)
    jersey: int | None = None                   # number reader (attribute stage)
    reid: tuple[float, ...] | None = None        # appearance embedding (attribute stage)

    @property
    def foot_xy(self) -> tuple[float, float]:
        """Bottom-centre of the box = the player's ground/foot point (court projection)."""
        x1, _, x2, y2 = self.box_xyxy
        return ((x1 + x2) / 2.0, y2)

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.class_id, str(self.class_id))

    def with_attrs(self, **kw) -> "Track":
        """Return a NEW Track with attributes set (immutable update)."""
        return replace(self, **kw)

    def to_record(self) -> dict:
        fx, fy = self.foot_xy
        return {
            "cam": self.cam, "frame": self.frame, "track_id": self.track_id,
            "box_xyxy": [round(v, 2) for v in self.box_xyxy],
            "foot_xy": [round(fx, 2), round(fy, 2)],
            "score": round(self.score, 4), "class_id": self.class_id,
            "class": self.class_name, "team": self.team, "jersey": self.jersey,
        }

    @classmethod
    def from_record(cls, r: dict) -> "Track":
        return cls(
            cam=r["cam"], frame=int(r["frame"]), track_id=int(r["track_id"]),
            box_xyxy=tuple(float(v) for v in r["box_xyxy"]), score=float(r.get("score", 0.0)),
            class_id=int(r["class_id"]), team=r.get("team"), jersey=r.get("jersey"),
        )
