"""Cross-camera fusion engine (docs/06): per-camera court observations -> ONE global
identity per player.

match-then-fuse: gate an association cost (court distance + ReID + team + jersey),
Hungarian-match each camera's observations to the live global tracks, fuse the matched
observations into one visibility-weighted Kalman update, and keep persistent global IDs
with jersey-as-authority, `(team, number)` uniqueness, and re-entry memory.

Operates on COURT-COORDINATE observations (cm); the per-camera homography (docs/03,
fusion/homography.py) produces them. Validated on synthetic multi-camera scenes until
the homographies are calibrated.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .kalman import CVKalman2D

_LARGE = 1e6

# The tuned 4-cam config (docs/06) — the SINGLE source of defaults for fuse_cams.py and
# pipeline/phases.run_fuse (the audit found three divergent parameter sets; the CLI silently
# reproduced the over-counting config). NOTE: tuned while NL/NR were unknowingly unsynced —
# re-tune (esp. cluster_dist=600) now that sync is strict and the clips carry audio.
TUNED = {"max_assoc_dist": 600.0, "gate_cost": 7.0, "w_t": 1.0, "w_a": 3.0,
         "min_hits": 4, "cluster_dist": 600.0}


@dataclass
class Observation:
    cam: str
    local_track_id: int
    court_xy: tuple[float, float]
    team: str | None = None
    jersey: int | None = None
    reid: np.ndarray | None = None
    score: float = 1.0
    zone_conf: float = 1.0           # camera's ownership confidence at this court location


def _cos(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


@dataclass
class GlobalTrack:
    id: int
    dt: float = 1 / 30.0
    jersey_min_votes: int = 3
    attr_decay: float = 0.98         # per-update decay of team/jersey votes: recent evidence
    kf: CVKalman2D = field(init=False)  # dominates, so early contamination (REF flicker, an ID
    members: dict[str, int] = field(default_factory=dict)  # switch) fades instead of persisting
    hits: int = 0
    time_since_update: int = 0
    last_frame: int = -1
    _team: Counter = field(default_factory=Counter)
    _jersey: Counter = field(default_factory=Counter)
    _reid_sum: np.ndarray | None = None
    _reid: np.ndarray | None = None

    def init(self, obs_list: list[Observation], frame: int) -> "GlobalTrack":
        c = np.mean([o.court_xy for o in obs_list], axis=0)
        self.kf = CVKalman2D(c, dt=self.dt)
        self._ingest(obs_list, frame, init=True)
        return self

    @property
    def pos(self) -> np.ndarray:
        return self.kf.pos

    @property
    def team(self) -> str | None:
        return self._team.most_common(1)[0][0] if self._team else None

    @property
    def jersey(self) -> int | None:
        if not self._jersey:
            return None
        num, cnt = self._jersey.most_common(1)[0]
        return num if cnt >= self.jersey_min_votes else None

    @property
    def reid(self) -> np.ndarray | None:
        return self._reid

    def predict(self) -> None:
        self.kf.predict()
        self.time_since_update += 1

    def update(self, obs_list: list[Observation], frame: int) -> None:
        w = [max(o.zone_conf * o.score, 1e-3) for o in obs_list]
        total = sum(w)
        z = np.sum([np.array(o.court_xy) * wi for o, wi in zip(obs_list, w)], axis=0) / total
        self.kf.update(z, weight=total)
        self._ingest(obs_list, frame)

    def _ingest(self, obs_list: list[Observation], frame: int, init: bool = False) -> None:
        for c in (self._team, self._jersey):
            for k in list(c):
                c[k] *= self.attr_decay
                if c[k] < 0.01:
                    del c[k]
        for o in obs_list:
            if o.team:
                self._team[o.team] += 1
            if o.jersey is not None:
                self._jersey[o.jersey] += 1
            if o.reid is not None:
                v = np.asarray(o.reid, float)
                v = v / (np.linalg.norm(v) + 1e-8)
                self._reid_sum = v if self._reid_sum is None else self._reid_sum + v
                self._reid = self._reid_sum / (np.linalg.norm(self._reid_sum) + 1e-8)
        self.members = {o.cam: o.local_track_id for o in obs_list}
        self.hits += 1
        self.time_since_update = 0
        self.last_frame = frame


class FusionEngine:
    """Per-frame cross-camera fusion. Call `step(frame, observations)` in order."""

    def __init__(self, *, dt: float = 1 / 30.0, w_d: float = 1.0, w_a: float = 1.5,
                 w_t: float = 4.0, w_j: float = 8.0, max_assoc_dist: float = 250.0,
                 gate_cost: float = 6.0, lost_buffer: int = 45, reentry_frames: int = 150,
                 reentry_dist: float = 350.0, merge_dist: float = 120.0, cluster_dist: float = 250.0,
                 min_hits: int = 3, jersey_min_votes: int = 3, attr_decay: float = 0.98,
                 reentry_min_score: float = 0.55):
        self.dt, self.w_d, self.w_a, self.w_t, self.w_j = dt, w_d, w_a, w_t, w_j
        self.max_assoc_dist, self.gate_cost = max_assoc_dist, gate_cost
        self.lost_buffer, self.reentry_frames = lost_buffer, reentry_frames
        self.reentry_dist, self.merge_dist, self.cluster_dist = reentry_dist, merge_dist, cluster_dist
        self.min_hits, self.jersey_min_votes = min_hits, jersey_min_votes
        self.attr_decay, self.reentry_min_score = attr_decay, reentry_min_score
        self.tracks: list[GlobalTrack] = []
        self.lost: list[GlobalTrack] = []
        self._next_id = 1

    # --- association cost (gated); on a point+attrs so groups & observations share it ---
    def _cost_xy(self, xy, team, jersey, reid, t: GlobalTrack) -> float:
        d = float(np.linalg.norm(np.array(xy) - t.pos))
        if d > self.max_assoc_dist:
            return _LARGE
        c = self.w_d * (d / 100.0)
        if team and t.team and team != t.team:
            c += self.w_t
        if jersey is not None and t.jersey is not None:
            if jersey == t.jersey and (not team or not t.team or team == t.team):
                c -= self.w_j                       # same (team,number): force the match
            else:
                return _LARGE                       # different confirmed number: never same player
        if reid is not None and t.reid is not None:
            c += self.w_a * (1.0 - _cos(reid, t.reid))
        return c

    def step(self, frame: int, observations: list[Observation]) -> list[GlobalTrack]:
        from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

        for t in self.tracks:
            t.predict()

        # 1. CLUSTER all cameras' observations into per-player groups (<=1 obs per camera)
        groups = self._cluster(observations)

        # 2. MATCH groups (centroid + aggregated attrs) to existing global tracks
        leftover = list(range(len(groups)))
        if self.tracks and groups:
            summ = [self._summarize(g) for g in groups]
            cost = np.array([[self._cost_xy(s["xy"], s["team"], s["jersey"], s["reid"], t)
                              for t in self.tracks] for s in summ])
            rows, cols = linear_sum_assignment(cost)
            taken = set()
            for i, j in zip(rows, cols):
                if cost[i, j] < self.gate_cost:
                    self.tracks[j].update(groups[i], frame)
                    taken.add(i)
            leftover = [i for i in range(len(groups)) if i not in taken]

        # 3. leftover groups -> re-entry or a new global track
        for i in leftover:
            if self._try_reentry(groups[i], frame) is None:
                t = GlobalTrack(self._next_id, dt=self.dt, jersey_min_votes=self.jersey_min_votes,
                                attr_decay=self.attr_decay).init(groups[i], frame)
                self._next_id += 1
                self.tracks.append(t)

        self._retire(frame)
        self._enforce_unique()
        return [t for t in self.tracks if t.hits >= self.min_hits and t.time_since_update == 0]

    # --- cluster ALL observations (across cameras) into per-player groups ---
    def _cluster(self, observations: list[Observation]) -> list[list[Observation]]:
        groups: list[list[Observation]] = []
        for o in sorted(observations, key=lambda x: -x.score):
            best, best_d = None, self.cluster_dist
            for g in groups:
                if any(x.cam == o.cam for x in g):          # one observation per camera per group
                    continue
                s = self._summarize(g)
                if o.team and s["team"] and o.team != s["team"]:
                    continue
                if o.jersey is not None and s["jersey"] is not None and o.jersey != s["jersey"]:
                    continue                                # different confirmed numbers: not one player
                d = float(np.linalg.norm(np.array(o.court_xy) - s["xy"]))
                if d <= best_d:
                    best, best_d = g, d
            if best is not None:
                best.append(o)
            else:
                groups.append([o])
        return groups

    @staticmethod
    def _summarize(g: list[Observation]) -> dict:
        teams = Counter(o.team for o in g if o.team)
        jers = Counter(o.jersey for o in g if o.jersey is not None)
        reids = [np.asarray(o.reid, float) for o in g if o.reid is not None]
        return {"xy": np.mean([o.court_xy for o in g], axis=0),
                "team": teams.most_common(1)[0][0] if teams else None,
                "jersey": jers.most_common(1)[0][0] if jers else None,
                "reid": np.mean(reids, axis=0) if reids else None}

    def _try_reentry(self, cl: list[Observation], frame: int) -> GlobalTrack | None:
        cen = np.mean([o.court_xy for o in cl], axis=0)
        jers = Counter(o.jersey for o in cl if o.jersey is not None)
        cj = jers.most_common(1)[0][0] if jers else None
        teams = Counter(o.team for o in cl if o.team)
        ct = teams.most_common(1)[0][0] if teams else None
        reid = next((o.reid for o in cl if o.reid is not None), None)
        best, best_score = None, -1.0
        for lt in self.lost:
            if frame - lt.last_frame > self.reentry_frames:
                continue
            if cj is not None and lt.jersey is not None and cj != lt.jersey:
                continue                                    # jersey rules it out
            if ct and lt.team and ct != lt.team:
                continue                                    # team rules it out (audit: reid alone
            if float(np.linalg.norm(cen - lt.pos)) > self.reentry_dist:  # can't — camera-biased)
                continue
            score = 0.0
            if cj is not None and lt.jersey == cj:
                score += 1.0                                # jersey match: strong
            if reid is not None and lt.reid is not None:
                score += _cos(reid, lt.reid)
            if score > best_score:
                best, best_score = lt, score
        # threshold: measured cross-cam same-rig reid cosine NEVER drops below 0.456, so the
        # old 0.3 gate could not reject anyone (identity theft on subs). 0.55 still accepts
        # true returns (same-person mean 0.69) and lets jersey (+1.0) override weak reid.
        if best is not None and best_score > self.reentry_min_score:
            self.lost.remove(best)
            self.tracks.append(best)
            best.update(cl, frame)
            return best
        return None

    def _retire(self, frame: int) -> None:
        keep = []
        for t in self.tracks:
            if t.time_since_update > self.lost_buffer:
                self.lost.append(t)
            else:
                keep.append(t)
        self.tracks = keep
        self.lost = [t for t in self.lost if frame - t.last_frame <= self.reentry_frames]

    def _enforce_unique(self) -> None:
        """At most one live track per committed (team, number). The weaker claimant KEEPS
        LIVING but its number votes are cleared (retraction) — the old behaviour silently
        DELETED the track, so one confidently-wrong jersey read could destroy a real
        identity (audit blast-radius finding). team=None tracks are exempt: (None, n)
        keys collide across teams and prove nothing."""
        best: dict[tuple, GlobalTrack] = {}
        for t in self.tracks:
            if t.jersey is None or t.team is None:
                continue
            key = (t.team, t.jersey)
            cur = best.get(key)
            if cur is None:
                best[key] = t
            elif t.hits > cur.hits:
                cur._jersey.clear()
                best[key] = t
            else:
                t._jersey.clear()


def fuse_sequence(frames: list[list[Observation]], **kw) -> list[list[dict]]:
    """Run the engine over a list of per-frame observation lists. Returns, per frame,
    the live global tracks as {global_id, court_xy, team, jersey, members}."""
    eng = FusionEngine(**kw)
    out = []
    for f, obs in enumerate(frames):
        live = eng.step(f, obs)
        out.append([{"global_id": t.id, "court_xy": [round(float(v), 1) for v in t.pos],
                     "team": t.team, "jersey": t.jersey, "members": dict(t.members)}
                    for t in live])
    return out
