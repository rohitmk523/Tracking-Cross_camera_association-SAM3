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
        import cv2
        torch = self.torch
        if crop_bgr is None or crop_bgr.size == 0:
            return None, 0.0
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        # 1. legibility gate (trained at 160x96)
        leg_in = self.leg_tf(cv2.resize(rgb, (96, 160))).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p_leg = torch.softmax(self.legibility(leg_in), 1)[0, 1].item()
        if p_leg < LEGIBLE_MIN:
            return None, p_leg
        # 2. localize the number
        res = self.localizer.predict(crop_bgr, imgsz=640, conf=LOC_MIN, verbose=False,
                                     device=self.device)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None, p_leg
        k = int(res.boxes.conf.argmax())
        x1, y1, x2, y2 = (float(v) for v in res.boxes.xyxy[k])
        h, w = crop_bgr.shape[:2]
        px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
        rx1, ry1 = max(0, int(x1 - px)), max(0, int(y1 - py))
        rx2, ry2 = min(w, int(x2 + px)), min(h, int(y2 + py))
        num_rgb = rgb[ry1:ry2, rx1:rx2]
        if num_rgb.size == 0:
            return None, p_leg
        # 3. read it (PARSeq, digit-filtered)
        rd_in = self.read_tf(cv2.resize(num_rgb, (128, 32),
                                        interpolation=cv2.INTER_CUBIC)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.reader(rd_in)
        text, conf = self.reader.tokenizer.decode(logits.softmax(-1))
        num = "".join(ch for ch in text[0] if ch.isdigit())
        c = float(conf[0].min()) if len(conf[0]) else 0.0
        if not num or len(num) > 2 or c < READ_MIN:
            return None, c
        return num, c
