"""The trained jersey-number stack (docs/05, audit-approved design): player crop -> number.

    LEGIBILITY gate (ResNet-18, 84.6% held-out)   is a number readable here at all?
      -> number LOCALIZER (YOLO11n, 0.99 mAP50)   tight box around the number
        -> PARSeq STR (95.3% held-out games)      open-vocabulary read ("23", "7", ...)

All three trained on the operator's cross-game annotations with leave-games-out
validation. read_crop() returns (number|None, confidence); a None at any gate is an
honest abstain — feeding fusion a wrong number is worse than feeding it nothing
(the engine's (team,number) authority acts on committed numbers).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
LEGIBILITY_W = REPO / "runs/jersey/legibility_resnet18.pt"
LOCALIZER_W = REPO / "runs/jersey/number_localizer_yolo11n.pt"
PARSEQ_W = REPO / "runs/jersey/parseq_jersey.pt"

LEGIBLE_MIN = 0.60          # legibility gate: P(legible) below this -> abstain
LOC_MIN = 0.40              # localizer box confidence below this -> abstain
READ_MIN = 0.55             # PARSeq min per-char confidence below this -> abstain
PAD = 0.15                  # context padding around the localized number box


class JerseyStack:
    def __init__(self, device: str | None = None):
        import torch
        from torchvision import models
        from torchvision import transforms as T
        from ultralytics import YOLO

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        leg = models.resnet18()
        leg.fc = torch.nn.Linear(leg.fc.in_features, 2)
        leg.load_state_dict(torch.load(LEGIBILITY_W, map_location="cpu", weights_only=True))
        self.legibility = leg.to(self.device).eval()
        self.leg_tf = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
        self.localizer = YOLO(str(LOCALIZER_W))
        self.reader = torch.hub.load("baudm/parseq", "parseq", pretrained=False,
                                     trust_repo=True)
        self.reader.load_state_dict(torch.load(PARSEQ_W, map_location="cpu",
                                               weights_only=True))
        self.reader = self.reader.to(self.device).eval()
        self.read_tf = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])

    def read_crop(self, crop_bgr: np.ndarray) -> tuple[str | None, float]:
        """Player crop (BGR) -> (number string | None, confidence). Abstains honestly."""
        return self.read_crops([crop_bgr])[0]

    def read_crops(self, crops_bgr: list, sub_batch: int = 64,
                   loc_batch: int = 32) -> list[tuple[str | None, float]]:
        """Batched read: same three gates as read_crop, one GPU pass per stage
        instead of one per crop. Returns (number|None, confidence) per input crop,
        order-aligned. This is the Phase-1 throughput path — the models and
        thresholds are identical to the per-crop path."""
        import cv2
        torch = self.torch
        out: list[tuple[str | None, float]] = [(None, 0.0)] * len(crops_bgr)
        valid = [(i, c) for i, c in enumerate(crops_bgr) if c is not None and c.size]
        if not valid:
            return out
        rgbs = {i: cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for i, c in valid}
        # 1. legibility gate (trained at 160x96), batched
        survivors: list[int] = []
        for s in range(0, len(valid), sub_batch):
            chunk = valid[s:s + sub_batch]
            batch = torch.stack([self.leg_tf(cv2.resize(rgbs[i], (96, 160)))
                                 for i, _ in chunk]).to(self.device)
            with torch.no_grad():
                p = torch.softmax(self.legibility(batch), 1)[:, 1].tolist()
            for (i, _), p_leg in zip(chunk, p):
                out[i] = (None, p_leg)
                if p_leg >= LEGIBLE_MIN:
                    survivors.append(i)
        if not survivors:
            return out
        # 2. localize the number, batched
        num_imgs: list = []
        num_idx: list[int] = []
        for s in range(0, len(survivors), loc_batch):
            chunk = survivors[s:s + loc_batch]
            results = self.localizer.predict([crops_bgr[i] for i in chunk], imgsz=640,
                                             conf=LOC_MIN, verbose=False,
                                             device=self.device, batch=loc_batch)
            for i, res in zip(chunk, results):
                if res.boxes is None or len(res.boxes) == 0:
                    continue
                k = int(res.boxes.conf.argmax())
                x1, y1, x2, y2 = (float(v) for v in res.boxes.xyxy[k])
                h, w = crops_bgr[i].shape[:2]
                px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
                rx1, ry1 = max(0, int(x1 - px)), max(0, int(y1 - py))
                rx2, ry2 = min(w, int(x2 + px)), min(h, int(y2 + py))
                num_rgb = rgbs[i][ry1:ry2, rx1:rx2]
                if num_rgb.size == 0:
                    continue
                num_imgs.append(cv2.resize(num_rgb, (128, 32),
                                           interpolation=cv2.INTER_CUBIC))
                num_idx.append(i)
        if not num_imgs:
            return out
        # 3. read them (PARSeq, digit-filtered), batched
        for s in range(0, len(num_imgs), sub_batch):
            batch = torch.stack([self.read_tf(m) for m in
                                 num_imgs[s:s + sub_batch]]).to(self.device)
            with torch.no_grad():
                logits = self.reader(batch)
            texts, confs = self.reader.tokenizer.decode(logits.softmax(-1))
            for i, text, conf in zip(num_idx[s:s + sub_batch], texts, confs):
                num = "".join(ch for ch in text if ch.isdigit())
                c = float(conf.min()) if len(conf) else 0.0
                if num and len(num) <= 2 and c >= READ_MIN:
                    out[i] = (num, c)
                else:
                    out[i] = (None, c)
        return out


MIN_VOTES = 2               # commit a track's number only on >=2 agreeing reads...
MIN_SHARE = 0.6             # ...that are also >=60% of all successful reads for the track


def vote_number(reads: list[str]) -> int | None:
    """Commit a track's jersey number from its per-frame reads — or abstain. Requires
    both absolute support (MIN_VOTES) and majority share (MIN_SHARE): one confident
    wrong read must never name a player."""
    from collections import Counter
    if not reads:
        return None
    num, cnt = Counter(reads).most_common(1)[0]
    if cnt >= MIN_VOTES and cnt / len(reads) >= MIN_SHARE:
        return int(num)
    return None


def read_track_jerseys(video_path, tracks, *, stack: "JerseyStack | None" = None,
                       sample_per_track: int = 12, min_box_h: int = 110) -> dict[int, int]:
    """{track_id -> committed number} for PLAYER tracks in one camera's video.
    Samples up to sample_per_track crops per track (attributes.per_track_crops), reads
    each through the stack, commits via vote_number. Small far-cam boxes are skipped
    (min_box_h) — numbers are a near-camera cue (docs/05)."""
    from .attributes import per_track_crops
    stack = stack or JerseyStack()
    crops_by_id = per_track_crops(video_path, tracks, class_id=0,
                                  sample_per_track=sample_per_track)
    out: dict[int, int] = {}
    for tid, crops in crops_by_id.items():
        reads = []
        for c in crops:
            if c is None or c.shape[0] < min_box_h:
                continue
            num, _ = stack.read_crop(c)
            if num is not None:
                reads.append(num)
        n = vote_number(reads)
        if n is not None:
            out[tid] = n
    return out
