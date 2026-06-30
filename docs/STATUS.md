# Cross-Camera Basketball Analysis — Progress

_Last updated: 2026-06-29_

## What the system does
Turns **four fixed-camera videos of a game** into **descriptive, timestamped play-by-play** —
automatically. It figures out *who* is on court (team + jersey), *where* everyone is, and *what
happened* (shots, passes, possessions), and writes it up like a commentator. It runs on the
existing 4-camera rig (far-left, far-right, near-left, near-right).

The pipeline has five stages. Here's where each stands:

| Stage | What it does | Status |
|---|---|---|
| **1. Detect** | Find every player, referee, ball in each camera, every frame | ✅ Working (players strong); ball + ref being strengthened |
| **2. Track** | Follow each person over time + label team + appearance signature | ✅ Working |
| **3. Fuse** | Merge all 4 cameras into one top-down map, one identity per player | ✅ Working (core IP) |
| **4. Events** | Deterministic "what happened" — possession, passes, shots | 🔶 Built; unblocking now (ball detection) |
| **5. Narrate** | AI commentator writes the play-by-play from the verified state | ✅ Working |

## Stage detail

**1 — Detection.** Our own detector (trained only on our footage) reliably finds **players**
(≈88–93% accuracy on unseen games). **Ball** and **referee** detection are the weak spots — the
ball is tiny/fast, refs vary game to game. This is the active improvement area (below).

**2 — Tracking.** Each player is followed across time and tagged with their **team** (by jersey
color) and a unique **appearance signature** for re-identification. Working reliably.

**3 — Cross-camera fusion (the hard, differentiating part).** All four camera views are projected
onto one **top-down court map** and merged so each player gets **one consistent identity** across
all cameras. We cut the over-counting from ~16 to ~13.5 tracked players per frame (real count ≈12)
by fixing how each camera's calibrated region is used. This is the piece most comparable systems
stop short of.

**4 — The "what happened" layer (current focus).** The engine that turns positions into
**possession / pass / turnover** events is **built and tested** — but it depends on reliably
tracking the **ball**, and the ball is too small for the current detector to localize. **This is
the one real blocker**, and it's exactly what we're resolving right now.

**5 — Narration.** A vision-language model (Google Gemini) watches the video, grounded on the
verified identities, and produces **rich, timestamped play-by-play naming real players**. Working
today; it gets more accurate once fed the events layer.

## What we built to unblock Stage 4 (the momentum story)
The ball detector needs more training examples of a **big, clearly-visible ball**. We built a data
pipeline that:
1. Reads the **event database** (every recorded shot, with timestamp and which basket),
2. Automatically pulls the **near-camera frames at each shot** — where the ball is largest and clearest,
3. Pre-labels them and stages them for correction.

This produced **1,152 training frames across all 25 games** — **73% carry a ball box** (vs ~5%
before). ~500 were annotated and we **retrained the detector on AWS GPU**.

## Retrain outcome (held-out games)
- **Ball: AP 0.80**, **referee: 0.90** (a previously-collapsing class, was ~0.04), player **0.91**.
- Near-basket play: the ball is detected well; **possession events are now clean** (sustained
  1–6s holds, sensible passes) on half-court sets.

## Full-court ball tracking — built and working
The honest blocker turned out to be that **mid-court the ball is too small to detect by
appearance — even a human can't reliably spot it in a single frame.** The answer is
**motion-based** detection: the ball is small but moves fast against a near-static court, so its
*motion* reveals it where its appearance can't. We built and validated the full pipeline:
1. **Per-camera motion detection** (3-frame differencing) — finds the moving ball mid-court.
2. **Cross-camera fusion by agreement** — the real ball appears at one court point in ≥2 cameras;
   a per-camera false positive appears in only one. Agreement both locates and confirms the ball.
3. **Audio-sync + Kalman track** — on a real fast break, the 4-camera fusion tracks the ball in a
   **smooth curve up the full court** (single camera managed a fraction of that), and the
   audio-sync validated itself against the rig's known camera offset.

It's **wired into the pipeline**: the fuse stage now writes the ball into the world-state and
derives **possession / pass / turnover** events, which the VLM narrates over.

## Cross-camera fusion — core working, two identity-quality gaps
- **Players:** the 12 on-court players each get **one stable global identity** tracked across all
  4 cameras through the whole clip. The hard part (one-ID-per-player across views) works; a mild
  over-count of transient IDs remains (far-camera calibration residual).
- **Ball:** full-court tracking (above) — fused into the same world-state.
- **Gap 1 — team A/B labels.** Verified against the video: the two teams are **roughly even**
  (cyan vs dark jerseys), but the classifier outputs ~9/3 because **one team is bright cyan (easy
  to cluster) and the other is dark and blends into the shadows** — SigLIP+KMeans mis-clusters the
  dark team. We made the cross-camera labelling architecturally correct (hue-anchored A/B + only
  the near cameras vote — the far cameras provably can't separate teams), but that fixes
  *consistency*, not the dark-team clustering. The real fix is a **better team-separation method**
  (explicit jersey-colour features, or a small trained team classifier) — a dedicated item.
- **Gap 2 — jersey numbers.** Not yet read. Decision: **defer, then train a model — not OCR.**
  General OCR fails on tiny/fisheye/blurred numbers; a number-localizer + a trained recognizer on
  annotated cross-game number crops (applied only on near/large crops, with voting) is the robust
  path — the same data-driven recipe as the ball. Lower leverage than team labels, so it comes after.

## Where we are
- **Working:** detection, tracking, cross-camera fusion (stable player IDs + full-court ball),
  the deterministic event stream (possession/pass/turnover), and narration — end to end.
- **Next, in order:** (1) team A/B via global near-camera classification, (2) trim fusion
  over-count, (3) shot make/miss (rim model exists, to be wired), (4) jersey numbers (trained),
  (5) broaden validation across more games/clips.

**One-line summary:** all five stages work end to end — four-camera video in, identified players +
a full-court ball + a structured "what happened" event stream + descriptive play-by-play out.

**One-line summary:** four of five stages work end-to-end; the fifth (structured "what happened")
works for half-court play now, and the path to full-court ball tracking is validated
(motion-based, prototype proven) — not blocked, just the next build.
