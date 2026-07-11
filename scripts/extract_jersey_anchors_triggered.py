#!/usr/bin/env python3
"""TRIGGERED jersey anchors — OCR as a service, not a firehose (Phase 2, user design).

Instead of reading every eligible crop (dense mode), a per-track state machine
decides which crops are worth reading:
  - UNCLAIMED track (new, or votes not yet committed): read every eligible crop.
  - CLAIMED track: heartbeat read every --heartbeat frames.
  - CONTACT burst: while a track overlaps another track (IoU >= 0.3) and for
    --burst frames after separation, read every frame — that is exactly where
    ByteTrack swaps IDs and where identity needs re-clamping.
  - CONTRADICTION: a confident read that disagrees with the committed claim puts
    the track back in burst mode.

Claims update after every batched flush (<=--flush frames of staleness).
Output schema is identical to the dense extractor (anchors + inline shade), so
every downstream consumer works unchanged. Adoption is gated on GT scoring vs
dense (docs/OPTIMIZATION_PLAN.md Phase 2).

  python scripts/extract_jersey_anchors_triggered.py --game c2a354fe --tag 300_60 \
      --heartbeat 60 --out-suffix .trigH60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ANGLES = ("FL", "FR", "NL", "NR")
OFFS = {"e6fba750_44_60": {"FL": 0, "FR": -11, "NL": -1, "NR": -1},
        "c2a354fe_300_60": {"FL": 0, "FR": 1, "NL": 2, "NR": -1}}
MIN_BOX_H = 90
CLAIM_VOTES, CLAIM_SHARE = 2, 0.6          # same commit rule as jersey_stack.vote_number
CONTACT_IOU = 0.3


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--heartbeat", type=int, default=60,
                    help="frames between reads on a claimed, uncontested track")
    ap.add_argument("--burst", type=int, default=15,
                    help="post-contact / post-contradiction dense-read frames")
    ap.add_argument("--flush", type=int, default=32,
                    help="frames per batched read flush (claim staleness bound)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--angles", default=",".join(ANGLES))
    ap.add_argument("--out-suffix", default="", help="suffix for output file")
    a = ap.parse_args()

    from uball_cc.detection.base import Detection
    from uball_cc.tracking.tracker import ByteTrackTracker
    from uball_cc.tracking.jersey_stack import JerseyStack
    from uball_cc.tracking.kit_shade import jersey_shade

    key = f"{a.game}_{a.tag}"
    offs = OFFS.get(key)
    if offs is None:
        offs = json.loads((REPO / f"runs/anchors/{key}.jersey_anchors.json").read_text())["offsets"]

    stack = JerseyStack()
    anchors = []
    tot_dense = tot_sel = 0
    for ang in a.angles.split(","):
        z = np.load(REPO / f"runs/dets_cache/{a.game}_{ang}_{a.tag}_small_1280_t0.25.dets.npz")
        by_f: dict[int, list] = {}
        for di, (b, s, c, f) in enumerate(zip(z["boxes"], z["scores"], z["classes"], z["frame_idx"])):
            if int(c) == 0 and float(s) >= 0.3:
                by_f.setdefault(int(f), []).append((di, [float(v) for v in b], float(s)))
        pose_p = REPO / f"runs/pose_cache/{a.game}_{ang}_{a.tag}.pose.npz"
        pose = None
        if pose_p.exists():
            p = np.load(pose_p)
            pose = {(int(f), int(d)): (k, s) for f, d, k, s in
                    zip(p["frame_idx"], p["det_idx"], p["kpts"], p["kscores"])}

        # --- pass 0: tracks from cached dets (cheap, no video) ---
        bt = ByteTrackTracker(ang)
        det2track: dict[tuple[int, int], int] = {}      # (frame, di) -> track_id
        contact_until: dict[int, int] = defaultdict(int)  # track_id -> frame
        for f in sorted(by_f):
            dets_f = [Detection(tuple(b), s, 0) for _, b, s in by_f[f]]
            tracks = bt.update(dets_f, f)
            for di, b, _s in by_f[f]:
                best, bid = 0.5, None
                for t in tracks:
                    v = iou(t.box_xyxy, b)
                    if v > best:
                        best, bid = v, t.track_id
                if bid is not None:
                    det2track[(f, di)] = bid
            for i, t in enumerate(tracks):            # contact episodes
                for u in tracks[i + 1:]:
                    if iou(t.box_xyxy, u.box_xyxy) >= CONTACT_IOU:
                        until = f + a.burst
                        contact_until[t.track_id] = max(contact_until[t.track_id], until)
                        contact_until[u.track_id] = max(contact_until[u.track_id], until)

        # --- pass 1: sequential decode, trigger machine, batched reads ---
        cap = cv2.VideoCapture(str(REPO / f"data/clips/{a.game}_{ang}_{a.tag}.mp4"))
        votes: dict[int, Counter] = defaultdict(Counter)
        claim: dict[int, int] = {}
        last_read: dict[int, int] = defaultdict(lambda: -10**9)
        burst_until: dict[int, int] = defaultdict(int)
        n_hit = 0
        t0 = time.time()
        pend = []                                       # (f, di, box, tid, crop)

        def flush():
            nonlocal n_hit
            if not pend:
                return
            reads = stack.read_crops([c for *_, c in pend], sub_batch=a.batch)
            for (f, di, b, tid, crop), (num, conf) in zip(pend, reads):
                if num is None:
                    continue
                n_hit += 1
                ev = {"frame": f - offs[ang], "cam": ang,
                      "box": [round(v, 1) for v in b],
                      "number": int(num), "conf": round(conf, 3)}
                kp = pose.get((f, di)) if pose else None
                if kp is not None:
                    k, ks = kp
                    sh = jersey_shade(crop, k - [int(max(0, b[0])), int(max(0, b[1]))], ks)
                    if sh is not None:
                        ev["shade"] = round(sh, 1)
                anchors.append(ev)
                votes[tid][int(num)] += 1
                top, cnt = votes[tid].most_common(1)[0]
                if cnt >= CLAIM_VOTES and cnt / sum(votes[tid].values()) >= CLAIM_SHARE:
                    if tid in claim and claim[tid] != top:
                        burst_until[tid] = f + a.burst   # contradiction -> re-clamp
                    claim[tid] = top
                elif tid in claim and int(num) != claim[tid] and conf >= 0.8:
                    burst_until[tid] = f + a.burst       # confident disagreement
            pend.clear()

        max_f = max(by_f) if by_f else -1
        n_dense = n_sel = 0
        clip_f = -1
        while clip_f < max_f:
            ok, img = cap.read()
            clip_f += 1
            if not ok:
                break
            if clip_f not in by_f:
                continue
            ih, iw = img.shape[:2]
            for di, b, _s in by_f[clip_f]:
                if b[3] - b[1] < MIN_BOX_H:
                    continue
                x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
                x2, y2 = min(iw, int(b[2])), min(ih, int(b[3]))
                if x2 - x1 < 20 or y2 - y1 < MIN_BOX_H:
                    continue
                n_dense += 1
                tid = det2track.get((clip_f, di))
                if tid is None:
                    continue                     # untracked det: no identity to clamp
                unclaimed = tid not in claim
                due = clip_f - last_read[tid] >= a.heartbeat
                hot = clip_f <= max(burst_until[tid], contact_until.get(tid, 0))
                if not (unclaimed or due or hot):
                    continue
                n_sel += 1
                last_read[tid] = clip_f
                pend.append((clip_f, di, b, tid, img[y1:y2, x1:x2].copy()))
            if clip_f % a.flush == 0:
                flush()
        flush()
        cap.release()
        tot_dense += n_dense
        tot_sel += n_sel
        print(f"{ang}: {n_sel}/{n_dense} crops selected ({100*n_sel/max(1,n_dense):.0f}%), "
              f"{n_hit} confident reads, {time.time()-t0:.0f}s", flush=True)

    out = REPO / f"runs/anchors/{key}.jersey_anchors{a.out_suffix}.json"
    out.write_text(json.dumps({"game": a.game, "tag": a.tag,
                               "stride": f"triggered_h{a.heartbeat}_b{a.burst}",
                               "offsets": offs, "anchors": anchors}))
    per_num = Counter(x["number"] for x in anchors)
    print(f"TOTAL: {len(anchors)} anchor events | crops {tot_sel}/{tot_dense} "
          f"({100*tot_sel/max(1,tot_dense):.0f}% of dense) -> {out}")
    print("per number:", dict(per_num.most_common(10)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
