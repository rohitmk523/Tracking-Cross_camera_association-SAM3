"""Pipeline demo video v3: detect (5s) -> track (5s) -> FUSE (12s, the star) -> events+narration.

FUSE layout per operator request: LEFT = 2x2 camera grid with BIG global-ID labels,
RIGHT = vertical court (court's right side at top), hollow dots = coasting (briefly unseen).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/Users/rohitkale/Cellstrat/GitHub_Repositories/Tracking-Cross_camera_association-SAM3")
sys.path.insert(0, str(REPO / "src"))
from uball_cc.fusion.court import draw_court  # noqa: E402

ANGLES = ("FL", "FR", "NL", "NR")
CLIP = str(REPO / "data/clips/e6fba750_{ang}_47_12.mp4")
TRACKS = str(REPO / "runs/tracking/e6fba750_{ang}_47_12_teams.json")
WS = REPO / "runs/tracking/e6_worldstate_v4.json"
DEMO = REPO / "runs/tracking/e6_demo_e2e.json"
OUT = REPO / "runs/tracking/pipeline_demo_raw.mp4"

W, H, FPS = 1920, 1080, 30
N_SHORT, N_FUSE = 150, 358
TEAM = {"A": (255, 210, 60), "B": (90, 220, 90), "REF": (0, 230, 230)}
CLS = {0: (235, 235, 235), 1: (0, 230, 230), 2: (60, 160, 255)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def banner(img, stage, title, sub=""):
    cv2.rectangle(img, (0, 0), (W, 74), (12, 12, 12), -1)
    cv2.putText(img, stage, (28, 32), FONT, 0.8, (120, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(img, title, (28, 62), FONT, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    if sub:
        (tw, _), _ = cv2.getTextSize(sub, FONT, 0.6, 1)
        cv2.putText(img, sub, (W - tw - 28, 46), FONT, 0.6, (170, 170, 170), 1, cv2.LINE_AA)


def load_tracks():
    by = {}
    for ang in ANGLES:
        d = json.loads(Path(TRACKS.format(ang=ang)).read_text())
        m = defaultdict(list)
        for t in d["tracks"]:
            m[t["frame"]].append(t)
        by[ang] = m
    return by


def label_box(img, x1, y1, x2, y2, col, lbl, fs, th, bth):
    cv2.rectangle(img, (x1, y1), (x2, y2), col, bth)
    (tw, tth), _ = cv2.getTextSize(lbl, FONT, fs, th)
    yt = max(tth + 10, y1 - 8)
    cv2.rectangle(img, (x1, yt - tth - 8), (x1 + tw + 10, yt + 6), (0, 0, 0), -1)
    cv2.putText(img, lbl, (x1 + 5, yt), FONT, fs, col, th, cv2.LINE_AA)


def cam_tile(frame, tracks, mode):
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t["box_xyxy"])
        if mode == "detect":
            cv2.rectangle(frame, (x1, y1), (x2, y2), CLS.get(t["class_id"], (200, 200, 200)), 4)
        else:
            col = TEAM.get(t.get("team"), (170, 170, 170))
            label_box(frame, x1, y1, x2, y2, col, f"{t.get('team') or '?'}{t['track_id']}",
                      1.5, 3, 4)
    return frame


def grid_positions():
    return [(r, c) for r in (0, 1) for c in (0, 1)]


def grid_segment(vw, caps, tracks_by, mode, n, stage, title, sub):
    for f in range(n):
        canvas = np.zeros((H, W, 3), np.uint8)
        for k, ang in enumerate(ANGLES):
            ok, img = caps[ang].read()
            if not ok:
                img = np.zeros((1080, 1920, 3), np.uint8)
            img = cam_tile(img, tracks_by[ang].get(f, []), mode)
            tile = cv2.resize(img, (960, 503))
            r, c = divmod(k, 2)
            y0 = 74 + r * 503
            canvas[y0:y0 + 503, c * 960:(c + 1) * 960] = tile
            cv2.putText(canvas, ang, (c * 960 + 12, y0 + 32), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        banner(canvas, stage, title, sub)
        vw.write(canvas)


def fuse_segment(vw, caps, tracks_by, ws_frames):
    base, to_px = draw_court(scale=0.40, margin=26)
    hb, wb = base.shape[:2]                      # horizontal court: wb x hb
    vcourt_base = np.rot90(base).copy()          # CCW: court's RIGHT side -> TOP
    vh, vw_ = vcourt_base.shape[:2]              # vertical: vw_ x vh  (= hb x wb)

    def vpx(court_xy):                           # court cm -> vertical-image pixel
        x, y = to_px(court_xy)
        return (y, wb - 1 - x)

    gx0 = 0
    gy0 = 74 + (H - 74 - 720) // 2
    cx0 = 1280 + (640 - vw_) // 2
    cy0 = 74 + (H - 74 - vh) // 2
    for f in range(N_FUSE):
        tracks = ws_frames.get(f, [])
        gmap = {}
        for t in tracks:
            if not t.get("coasting"):
                for cam, lid in (t.get("members") or {}).items():
                    gmap[(cam, lid)] = (t["global_id"], t.get("team"))
        canvas = np.zeros((H, W, 3), np.uint8)
        for k, ang in enumerate(ANGLES):                       # LEFT: 2x2 camera grid
            ok, img = caps[ang].read()
            if not ok:
                img = np.zeros((1080, 1920, 3), np.uint8)
            for t in tracks_by[ang].get(f, []):
                x1, y1, x2, y2 = (int(v) for v in t["box_xyxy"])
                hit = gmap.get((ang, t["track_id"]))
                if hit:
                    gid, team = hit
                    label_box(img, x1, y1, x2, y2, TEAM.get(team, (170, 170, 170)),
                              f"G{gid}", 2.4, 6, 6)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (105, 105, 105), 2)
            tile = cv2.resize(img, (640, 360))
            r, c = divmod(k, 2)
            y0 = gy0 + r * 360
            canvas[y0:y0 + 360, gx0 + c * 640:gx0 + (c + 1) * 640] = tile
            cv2.putText(canvas, ang, (gx0 + c * 640 + 10, y0 + 28), FONT, 0.85,
                        (255, 255, 255), 2, cv2.LINE_AA)
        court = vcourt_base.copy()                             # RIGHT: vertical court
        for t in tracks:
            col = TEAM.get(t.get("team"), (170, 170, 170))
            p = vpx(t["court_xy"])
            if t.get("coasting"):
                cv2.circle(court, p, 13, col, 3)               # hollow = briefly unseen
            else:
                cv2.circle(court, p, 13, col, -1)
                cv2.circle(court, p, 13, (0, 0, 0), 1)
            cv2.putText(court, str(t["global_id"]), (p[0] + 15, p[1] + 7), FONT, 0.85,
                        col, 2, cv2.LINE_AA)
        canvas[cy0:cy0 + vh, cx0:cx0 + vw_] = court
        cv2.putText(canvas, "RIGHT end", (cx0 + 8, cy0 + 24), FONT, 0.6, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(canvas, "LEFT end", (cx0 + 8, cy0 + vh - 10), FONT, 0.6, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(canvas, "same G number = same player in every camera",
                    (gx0 + 14, gy0 + 720 + 34), FONT, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "hollow dot = briefly unseen (0.4s prediction)",
                    (gx0 + 14, gy0 + 720 + 66), FONT, 0.6, (170, 170, 170), 1, cv2.LINE_AA)
        banner(canvas, "STAGE 3 / 5", "FUSE - 4 cameras matched into ONE identity per player",
               "lens-corrected + audio-synced + Kalman")
        vw.write(canvas)


def wrap(text, width=86):
    out, line = [], ""
    for w_ in text.split():
        if len(line) + len(w_) + 1 > width:
            out.append(line)
            line = w_
        else:
            line = f"{line} {w_}".strip()
    if line:
        out.append(line)
    return out


def text_segment(vw, events, narration, seconds=12):
    canvas = np.zeros((H, W, 3), np.uint8)
    banner(canvas, "STAGE 4+5 / 5", "EVENTS (deterministic JSON)  ->  AI PLAY-BY-PLAY (grounded)",
           "the world-state becomes text")
    cv2.rectangle(canvas, (30, 100), (930, 1050), (24, 24, 24), -1)
    cv2.putText(canvas, "events.json", (50, 132), FONT, 0.7, (120, 200, 255), 2, cv2.LINE_AA)
    y = 168
    for e in events[:9]:
        keep = {k: e[k] for k in ("event", "t_sec", "player", "from", "to", "team") if k in e}
        for ln in wrap(json.dumps(keep), 58):
            cv2.putText(canvas, ln, (50, y), FONT, 0.52, (210, 210, 210), 1, cv2.LINE_AA)
            y += 26
        y += 10
        if y > 1010:
            break
    cv2.rectangle(canvas, (960, 100), (1890, 1050), (24, 24, 24), -1)
    cv2.putText(canvas, "AI commentator (Gemini, grounded on the world-state)",
                (980, 132), FONT, 0.62, (120, 255, 170), 2, cv2.LINE_AA)
    y = 172
    for ln in wrap(narration.get("summary", ""), 62):
        cv2.putText(canvas, ln, (980, y), FONT, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28
    y += 14
    for pl in narration.get("play_by_play", [])[:8]:
        for j, ln in enumerate(wrap("- " + pl.get("description", ""), 64)):
            cv2.putText(canvas, ln, (980 + (0 if j == 0 else 18), y), FONT, 0.52,
                        (200, 230, 200), 1, cv2.LINE_AA)
            y += 25
        y += 8
        if y > 1020:
            break
    cv2.putText(canvas, "possession/pass events: pending trained ball detector + ground-truth scoring",
                (30, 1070), FONT, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
    for _ in range(seconds * FPS):
        vw.write(canvas)


def title_card(vw, lines, seconds=2):
    canvas = np.zeros((H, W, 3), np.uint8)
    for i, (txt, scale, col) in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(txt, FONT, scale, 2)
        cv2.putText(canvas, txt, ((W - tw) // 2, 440 + i * 70), FONT, scale, col, 2, cv2.LINE_AA)
    for _ in range(int(seconds * FPS)):
        vw.write(canvas)


def main():
    tracks_by = load_tracks()
    ws = json.loads(WS.read_text())
    ws_frames = {fr["frame"]: fr["tracks"] for fr in ws["frames"]}
    demo = json.loads(DEMO.read_text())
    events = [e for e in demo["events"]["events"] if e["event"] in ("possession", "pass", "turnover")]
    narration = demo.get("narration", {})

    vw = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    title_card(vw, [("4-CAMERA BASKETBALL PIPELINE", 1.4, (255, 255, 255)),
                    ("detect -> track -> fuse -> events -> narration", 0.9, (150, 200, 255)),
                    ("game e6fba750, one 12-second window, all four cameras", 0.7, (160, 160, 160))])

    def fresh_caps():
        return {ang: cv2.VideoCapture(CLIP.format(ang=ang)) for ang in ANGLES}

    caps = fresh_caps()
    grid_segment(vw, caps, tracks_by, "detect", N_SHORT, "STAGE 1 / 5",
                 "DETECT - every player, referee and ball, every frame, every camera",
                 "RF-DETR trained on our own footage")
    caps = fresh_caps()
    grid_segment(vw, caps, tracks_by, "track", N_SHORT, "STAGE 2 / 5",
                 "TRACK - follow each person per camera (each camera numbers on its OWN)",
                 "A7 here and A7 there are NOT the same player yet - fusion solves that")
    caps = fresh_caps()
    fuse_segment(vw, caps, tracks_by, ws_frames)
    text_segment(vw, events, narration)
    vw.release()
    print(f"raw -> {OUT}")


if __name__ == "__main__":
    main()
