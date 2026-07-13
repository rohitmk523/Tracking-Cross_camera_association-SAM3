#!/usr/bin/env python3
"""E3 stage A (our venv) — build KPR samples for track-absent WHO recovery.

PROTOTYPES: identity-labelled crops from the 134 release clips (track box,
height >=90px, ball NOT overlapping — clean appearance), <=15 per identity,
with RTMPose keypoints as KPR prompts.
QUERIES: for harness shots with NO track candidate under the ball at release
(the E0 track-absent set), a person-sized crop under the ball + keypoints.

Writes runs/e3_kpr/{crops/*.png, samples.json} for stage B (kpr venv).

  .venv/bin/python scripts/e3_kpr_stageA.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import sys as _sys

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANGLES = ("FL", "FR", "NL", "NR")
OFFS = {"FL": 0, "FR": -11, "NL": -1, "NR": -1}
FPS = 29.97
import sys
OUT = REPO / f"runs/e3_kpr_{sys.argv[sys.argv.index('--game')+1][:3] if '--game' in sys.argv else 'e6f'}"
MAX_PROTO = 15


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="e6fba750")
    ap.add_argument("--candidates", action="store_true")
    a = ap.parse_args()
    game = a.game
    led = json.loads((REPO / f"runs/tracking/ledger/shots_{game}_full.json").read_text())
    plays = json.loads((REPO / f"data/plays/{game}_full.json").read_text())["plays"]
    gt = [p for p in plays if ("MAKE" in p["cls"] or "MISS" in p["cls"])]
    roster = json.loads((REPO / f"data/rosters/{game}.json").read_text())
    name_num = {}
    for pr in roster["players"]:
        name_num.setdefault(pr["name"].split()[-1], pr["num"])

    from rtmlib import RTMPose
    pose = RTMPose(
        onnx_model="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
                   "onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
        model_input_size=(192, 256), backend="onnxruntime", device="mps")

    balls, trks = {}, {}
    def chunk_data(tag):
        if tag not in balls:
            b = {a: {} for a in ANGLES}
            for a in ANGLES:
                z = np.load(REPO / f"runs/ball_cache/{game}_{a}_{tag}.ball.npz")
                for bx, s, f, c in zip(z["boxes"], z["scores"], z["frame_idx"], z["classes"]):
                    if int(c) != 0:
                        continue
                    f = int(f)
                    if f not in b[a] or s > b[a][f][1]:
                        b[a][f] = ([float(v) for v in bx], float(s))
            t = defaultdict(dict)
            tdir = REPO / (f"runs/events_fg_{tag}" if game == "e6fba750"
                           else f"runs/events_fg_{game[:3]}_{tag}")
            for p in tdir.glob(f"{game}_{tag}__n*__*.json"):
                parts = p.stem.split("__")
                pl, a2 = "#" + parts[1][1:], parts[2]
                d = json.loads(p.read_text())["frames"]
                t[pl][a2] = {int(f): r["box"] for f, r in d.items() if r.get("present")}
            balls[tag], trks[tag] = b, t
        return balls[tag], trks[tag]

    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    proto_count = defaultdict(int)
    samples = []          # {kind, id/gt, path, kxyc}
    n_query = 0

    for g in gt:
        cand = [o for o in led if abs(o["t"] - g["t"]) <= 1.5 and o.get("rel_f") is not None]
        if not cand:
            continue
        o = min(cand, key=lambda o: abs(o["t"] - g["t"]))
        clip = REPO / f"runs/e1_frames/{game}_{o['chunk']}_{o['rel_f']}_{o['cam']}.mp4"
        if not clip.exists():
            clip = REPO / f"runs/e1_frames/{o['chunk']}_{o['rel_f']}_{o['cam']}.mp4"
        if not clip.exists():
            continue
        ball, tr = chunk_data(o["chunk"])
        ang, rel = o["cam"], o["rel_f"]
        cap = cv2.VideoCapture(str(clip))
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
        cap.release()
        if not frames:
            continue

        def crop_sample(img, box, kind, label):
            ih, iw = img.shape[:2]
            x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
            x2, y2 = min(iw, int(box[2])), min(ih, int(box[3]))
            if x2 - x1 < 20 or y2 - y1 < 50:
                return None
            crop = img[y1:y2, x1:x2]
            kpts, ks = pose(img, bboxes=[[x1, y1, x2, y2]])
            kc = kpts[0] - [x1, y1]
            kxyc = np.concatenate([kc, ks[0][:, None]], axis=1)
            name = f"{kind}_{len(samples):05d}.png"
            cv2.imwrite(str(OUT / "crops" / name), crop)
            samples.append({"kind": kind, "label": label, "path": f"crops/{name}",
                            "kxyc": kxyc.tolist()})
            return True

        # mid frame = release
        mid = frames[min(3, len(frames) - 1)]
        bb = ball[ang].get(rel)
        bxy = None
        if bb:
            bxy = ((bb[0][0] + bb[0][2]) / 2, (bb[0][1] + bb[0][3]) / 2)

        # PROTOTYPES: clean tracked identities in this frame
        for pl in tr:
            if proto_count[pl] >= MAX_PROTO:
                continue
            box = tr[pl].get(ang, {}).get(rel)
            if box is None or (box[3] - box[1]) < 90:
                continue
            if bxy and box[0] - 40 <= bxy[0] <= box[2] + 40 and bxy[1] <= box[3]:
                continue                       # ball on/near him — ambiguous
            if crop_sample(mid, box, "proto", pl):
                proto_count[pl] += 1

        # R1-lite: emit EVERY candidate stream's crop at release, labeled
        # stream|shot-idx — stage B scores each crop against its OWN stream
        # prototypes (identity-switch detection at the decision point)
        if a.candidates and bxy is not None:
            for pl in tr:
                box = tr[pl].get(ang, {}).get(rel)
                if box is None:
                    continue
                bw = max(box[2] - box[0], 1.0)
                if not (box[0] - bw <= bxy[0] <= box[2] + bw) or bxy[1] > box[3]:
                    continue
                crop_sample(mid, box, "cand", f"{pl}|{g['t']}")

        # QUERY: track-absent at release (no candidate under ball)
        num = name_num.get(g["a"].split()[-1])
        has_cand = False
        if bxy:
            for pl in tr:
                box = tr[pl].get(ang, {}).get(rel)
                if box is None:
                    continue
                bw = max(box[2] - box[0], 1.0)
                if box[0] - bw <= bxy[0] <= box[2] + bw and bxy[1] <= box[3]:
                    has_cand = True
                    break
        if bxy and not has_cand and num is not None:
            # W3: query = the DETECTOR box under the ball (real person boxes
            # survive where identity tracks died), not a guessed geometry crop
            qbox = None
            dp = REPO / (f"runs/dets_cache/{game}_{ang}_{o['chunk']}"
                         f"_small_1280_t0.25.dets.npz")
            if dp.exists():
                z = np.load(dp)
                best = None
                for b, s, c, fr in zip(z["boxes"], z["scores"], z["classes"],
                                       z["frame_idx"]):
                    if int(c) != 0 or int(fr) != rel or s < 0.4:
                        continue
                    if bxy[1] > b[3] or bxy[1] > b[1] + 0.6 * (b[3] - b[1]):
                        continue
                    bw = max(b[2] - b[0], 1.0)
                    if not (b[0] - bw <= bxy[0] <= b[2] + bw):
                        continue
                    d = abs(bxy[0] - (b[0] + b[2]) / 2) / bw
                    if best is None or d < best[0]:
                        best = (d, [float(v) for v in b])
                if best is not None:
                    qbox = best[1]
            if qbox is None:
                bw_px = max(bb[0][2] - bb[0][0], 18.0)
                px_h = bw_px * 8.0             # fallback: geometry crop
                qbox = [bxy[0] - px_h * 0.22, bxy[1] - px_h * 0.15,
                        bxy[0] + px_h * 0.22, bxy[1] + px_h * 0.85]
            if crop_sample(mid, qbox, "query", str(num)):
                n_query += 1

    (OUT / "samples.json").write_text(json.dumps(samples))
    n_proto = sum(1 for s in samples if s["kind"] == "proto")
    ids = {s["label"] for s in samples if s["kind"] == "proto"}
    print(f"stage A: {n_proto} prototype crops over {len(ids)} identities, "
          f"{n_query} track-absent queries -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
