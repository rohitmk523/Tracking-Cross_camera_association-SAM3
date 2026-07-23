#!/usr/bin/env python3
"""Glowing floor-ring overlay for the full-game renderer.

Draws the alpha `Circle.mov` ring under the ball holder. Unlike the standalone
`desgin` renderer (which had per-frame segmentation masks), here the holder is a
CROSS-CAMERA FUSED identity and we only have his BOX per camera -- so the ring is
anchored to the box's bottom-centre (feet) and sized from the box width. The
holder id is the same global stream across all four angles, so the ring lands on
the same physical player in every view automatically.

Self-contained (no dependency on the desgin repo): Circle.mov -> alpha cache,
a One-Euro stabiliser with vertical ground-lock (so the ring stays planted when
the holder jumps), and box-based ring compositing.
"""
import json
from pathlib import Path

import cv2
import numpy as np

_TWO_PI = 2.0 * np.pi


# ------------------------------ Circle.mov cache ---------------------------
def build_overlay_cache(mov_path, cache_path, max_dim=640):
    """Decode the alpha .mov ONCE into a stacked BGRA .npy. Returns (n, fps)."""
    cache_path = str(cache_path)
    meta = cache_path + ".json"
    if Path(cache_path).exists() and Path(meta).exists():
        m = json.loads(Path(meta).read_text())
        return m["n"], m["fps"]
    import av  # noqa: PLC0415
    frames = []
    with av.open(str(mov_path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        for frame in container.decode(stream):
            bgra = cv2.cvtColor(frame.to_ndarray(format="rgba"), cv2.COLOR_RGBA2BGRA)
            h, w = bgra.shape[:2]
            if max(h, w) > max_dim:
                s = max_dim / max(h, w)
                bgra = cv2.resize(bgra, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            frames.append(bgra)
    np.save(cache_path, np.stack(frames, 0))
    Path(meta).write_text(json.dumps({"n": len(frames), "fps": fps}))
    return len(frames), fps


# ------------------------------ Stabiliser ---------------------------------
class _OneEuro:
    def __init__(self, mincutoff, beta, dcutoff=1.0):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self.x_prev = None
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff):
        tau = 1.0 / (_TWO_PI * cutoff)
        return 1.0 / (1.0 + tau)

    def reset(self, x=None):
        self.x_prev = x
        self.dx_prev = 0.0

    def filter(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = x - self.x_prev
        a_d = self._alpha(self.dcutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        a = self._alpha(self.mincutoff + self.beta * abs(dx_hat))
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat


class RingStabilizer:
    """One-Euro smoothing of the ring's foot centre + width, plus a vertical
    ground-lock (foot-Y may fall freely but rises slowly) so a jumping holder's
    ring stays on the floor. A move beyond `snap_dist` snaps instead of sliding
    (possession switch / reappearance). Call reset() on frames with no ring."""

    def __init__(self, mincutoff=0.05, beta=0.007, snap_dist=90.0,
                 ground_down=0.5, ground_max_up=2.0, ground_release=15):
        self.snap_dist = snap_dist
        self.ground_down = ground_down
        self.ground_max_up = ground_max_up
        self.ground_release = ground_release
        self.fx = _OneEuro(mincutoff, beta)
        self.fy = _OneEuro(mincutoff, beta)
        self.fw = _OneEuro(mincutoff, beta)
        self.cx = self.cy = None
        self.ground_y = None
        self.rising = 0

    def reset(self):
        self.fx.reset(); self.fy.reset(); self.fw.reset()
        self.cx = self.cy = None
        self.ground_y = None
        self.rising = 0

    def _lock(self, cy):
        if self.ground_y is None:
            self.ground_y = float(cy); self.rising = 0
        elif cy >= self.ground_y:
            self.ground_y += self.ground_down * (cy - self.ground_y); self.rising = 0
        else:
            self.rising += 1
            rate = self.ground_max_up
            if self.rising > self.ground_release:
                rate = max(self.ground_max_up, self.ground_y - cy)
            self.ground_y = max(float(cy), self.ground_y - rate)
        return self.ground_y

    def apply(self, cx, cy, w):
        cyg = self._lock(cy)
        if self.cx is not None and abs(cx - self.cx) + abs(cyg - self.cy) > self.snap_dist:
            self.fx.reset(cx); self.fy.reset(cyg); self.fw.reset(w)
            self.ground_y = float(cy)
            self.cx, self.cy = float(cx), float(cyg)
            return int(round(cx)), int(round(cyg)), float(w)
        self.cx = self.fx.filter(cx)
        self.cy = self.fy.filter(cyg)
        return int(round(self.cx)), int(round(self.cy)), self.fw.filter(w)


# ------------------------------ Ring drawing -------------------------------
def _alpha_composite(bg, ov, cx, cy, occ=None):
    oh, ow = ov.shape[:2]
    bh, bw = bg.shape[:2]
    x1, y1 = cx - ow // 2, cy - oh // 2
    x2, y2 = x1 + ow, y1 + oh
    fx1, fy1, fx2, fy2 = max(x1, 0), max(y1, 0), min(x2, bw), min(y2, bh)
    if fx1 >= fx2 or fy1 >= fy2:
        return
    ox1, oy1 = fx1 - x1, fy1 - y1
    roi = bg[fy1:fy2, fx1:fx2]
    o = ov[oy1:oy1 + (fy2 - fy1), ox1:ox1 + (fx2 - fx1)]
    alpha = o[:, :, 3:4] / 255.0 if o.shape[2] == 4 else 1.0
    if occ is not None:
        alpha = alpha * (occ[fy1:fy2, fx1:fx2] == 0)[:, :, None]
    roi[:] = (1.0 - alpha) * roi + alpha * o[:, :, :3]


def draw_ring_box(cell, box, overlay_img, stab=None, occ=None,
                  scale=2.2, squash=0.40, feet_lift=0.10,
                  min_w_far_frac=0.06, min_w_near_frac=0.24):
    """Draw the ground ring under a holder given his BOX in cell pixels.

    box: (x1,y1,x2,y2) in the cell's coordinate system. The ring sits at the
    box bottom-centre (feet), width scales with the box, with a proximity floor
    (nearer the camera -> lower in the cell -> larger ring). `stab` smooths it;
    `occ` optionally hides it behind foreground boxes.
    """
    ch, cw = cell.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy, w = (x1 + x2) * 0.5, y2, max(1.0, x2 - x1)
    if stab is not None:
        cx, cy, w = stab.apply(cx, cy, w)
    frac = min(1.0, max(0.0, cy / ch)) if ch else 0.0
    min_w = (min_w_far_frac + (min_w_near_frac - min_w_far_frac) * frac) * cw
    ring_w = max(int(round(min_w)), int(round(w * scale)))
    ring_h = max(1, int(round(overlay_img.shape[0] * (ring_w / overlay_img.shape[1]) * squash)))
    resized = cv2.resize(overlay_img, (ring_w, ring_h), interpolation=cv2.INTER_AREA)
    _alpha_composite(cell, resized, int(round(cx)), int(round(cy - ring_h * feet_lift)), occ)


def occlusion_from_boxes(boxes, holder_box, shape):
    """Mask of players standing IN FRONT of the holder (lower foot line), so the
    ring tucks behind them. Coarse (box-level) but keeps depth ordering sane."""
    occ = np.zeros(shape[:2], np.uint8)
    hy2 = holder_box[3]
    for b in boxes:
        if b is holder_box or b[3] <= hy2:      # only players closer to camera
            continue
        cv2.rectangle(occ, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), 255, -1)
    return occ
