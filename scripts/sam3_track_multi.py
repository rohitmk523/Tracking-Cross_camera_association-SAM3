#!/usr/bin/env python3
"""SAM3 MULTI-OBJECT tracking — N players in ONE pass per camera (runs on the AWS GPU).

Same recipe as sam3_track_player.py (box prompts -> mask propagation) but all players
are seeded together at one common frame and share the image encoder, which is where
almost all the compute goes. 59 single-object passes become 4 multi-object passes.

Output: one JSON per player (same schema as sam3_track_player.py so every downstream
scorer/solver works unchanged).

  python scripts/sam3_track_multi.py --video clip.mp4 --seed-frame 120 \
      --seeds '{"#11": [x1,y1,x2,y2], ...}' --cam FR --out-dir out --tag e6fba750_44_60
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


def per_object_records(r, n_obj, np):
    """Demux one Results frame into n_obj records (box/score/area/present)."""
    recs = []
    masks = getattr(r, "masks", None)
    boxes = getattr(r, "boxes", None)
    mdata = masks.data.cpu().numpy() if masks is not None and masks.data is not None else None
    for i in range(n_obj):
        rec = {"box": None, "score": 0.0, "mask_area": 0, "present": False}
        if mdata is not None and i < len(mdata):
            m = mdata[i] > 0.5
            area = int(m.sum())
            if area > 40:
                ys, xs = np.where(m)
                sc = 1.0
                if boxes is not None and getattr(boxes, "conf", None) is not None and i < len(boxes.conf):
                    sc = round(float(boxes.conf[i]), 3)
                rec = {"box": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                       "score": sc, "mask_area": area, "present": True}
        recs.append(rec)
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--seed-frame", type=int, required=True)
    ap.add_argument("--seeds", required=True, help='JSON {"#11": [x1,y1,x2,y2], ...}')
    ap.add_argument("--cam", required=True)
    ap.add_argument("--tag", required=True, help="output stem: {tag}__{player}__{cam}.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--weights", default="sam3.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--max-frames", type=int, default=0, help="debug: stop after N frames")
    a = ap.parse_args()

    import cv2
    import numpy as np
    from ultralytics.models.sam import SAM3VideoPredictor

    seeds = json.loads(a.seeds)
    players = sorted(seeds)                      # stable order == obj_id order
    bboxes = [seeds[p] for p in players]
    n = len(players)

    src = cv2.VideoCapture(a.video)
    fps = src.get(cv2.CAP_PROP_FPS) or 29.97
    w, h = int(src.get(cv2.CAP_PROP_FRAME_WIDTH)), int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src.release()
    tdir = Path(tempfile.mkdtemp())

    def write_clip(path, frame_order):
        """Write frames (list of absolute indices, in stream order) to a temp video."""
        cap = cv2.VideoCapture(a.video)
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        # decode in chunks of ascending index, buffer, emit in requested order
        CH = 200
        cache = {}
        need = list(frame_order)
        for start in range(0, len(need), CH):
            want = need[start:start + CH]
            lo, hi = min(want), max(want)
            cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
            for f in range(lo, hi + 1):
                ok, img = cap.read()
                if not ok:
                    break
                if f in want:
                    cache[f] = img
            for f in want:
                if f in cache:
                    vw.write(cache[f])
            cache.clear()
        vw.release()
        cap.release()

    def propagate(clip_path, abs_of_stream, direction):
        pred = SAM3VideoPredictor(overrides=dict(
            conf=0.25, task="segment", mode="predict", imgsz=a.imgsz,
            model=a.weights, save=False, verbose=False))
        got = {}
        for fi, r in enumerate(pred(source=str(clip_path), bboxes=bboxes, stream=True)):
            absf = abs_of_stream(fi)
            got[absf] = per_object_records(r, n, np)
            if fi % 100 == 0:
                live = sum(1 for rec in got[absf] if rec["present"])
                print(f"[{a.cam} {direction}] frame {absf}: {live}/{n} present", flush=True)
        return got

    cap = cv2.VideoCapture(a.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    end = min(total, a.seed_frame + a.max_frames) if a.max_frames else total
    lo = max(0, a.seed_frame - a.max_frames) if a.max_frames else 0
    print(f"[{a.cam}] seeding {n} players at frame {a.seed_frame} "
          f"(forward to {end}, backward to {lo}): {players}", flush=True)

    fwd_clip = tdir / "fwd.mp4"
    write_clip(fwd_clip, list(range(a.seed_frame, end)))
    merged = propagate(fwd_clip, lambda i: a.seed_frame + i, "fwd")
    fwd_clip.unlink()

    if a.seed_frame > lo:                        # backward: reversed [lo..seed_frame]
        rev_clip = tdir / "rev.mp4"
        write_clip(rev_clip, list(range(a.seed_frame, lo - 1 if lo else -1, -1)))
        back = propagate(rev_clip, lambda i: a.seed_frame - i, "rev")
        rev_clip.unlink()
        back.update(merged)                      # forward wins at the seed frame
        merged = back

    frames = {p: {f: recs[i] for f, recs in merged.items()} for i, p in enumerate(players)}
    fi = len(merged)

    outd = Path(a.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    for p in players:
        safe = p.replace("#", "n").replace(" ", "")
        (outd / f"{a.tag}__{safe}__{a.cam}.json").write_text(json.dumps(
            {"player": p, "cam": a.cam, "seed_frame": a.seed_frame, "seed_box": seeds[p],
             "imgsz": a.imgsz, "multi": True, "frames": frames[p]}))
    print(f"[{a.cam}] wrote {n} masklets ({fi} frames each) -> {outd}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
