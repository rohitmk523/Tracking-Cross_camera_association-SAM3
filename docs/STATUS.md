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

This produced **1,152 training frames across all 25 games** — **73% already carry a ball box**
(vs ~5% before). It's running through annotation now. Once corrected → we retrain the ball
detector → the possession/shot events **light up automatically** (that machinery is already in place).

## Where we are / what's next
- **Now:** annotating the ball training set (the only step that needs human eyes).
- **Next:** retrain the detector on AWS GPU (pipeline wired, one command) → ball tracking works →
  possession + passes appear.
- **Then:** shot make/miss detection (rim model already exists), and feeding the events into the
  narrator for maximum accuracy.

**One-line summary:** four of five stages are working end-to-end; the fifth (the structured "what
happened" data) is built and gated only on better ball detection — and we've already built and
scaled the data pipeline that fixes it.
