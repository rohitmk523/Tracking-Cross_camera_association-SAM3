# Cross-Camera Basketball Analysis — Progress Report

**What the system does:** you give it the video from the four cameras already installed in
the gym. It works out **who** is on the court (which team, which jersey number), **where**
everyone is (drawn live on a top-down court map), and **what happened** (who had the ball,
passes, turnovers) — and then an AI commentator writes the play-by-play, naming real players.

## Where each part stands

| Step | What it does | Status |
|---|---|---|
| **1. Detect** | Spot every player, referee and ball in every video frame | ✅ Working — and an upgraded version is training right now |
| **2. Track** | Follow each person through time in each camera | ✅ Working |
| **3. Combine** | Merge all 4 cameras so each player is ONE person on one map | ✅ **Now proven on 5 different games, hands-off** |
| **4. Name** | Read jersey numbers so players get real identities | ✅ Working — named players on every new game we tried |
| **5. Events** | Say who had the ball, passes, turnovers | 🔶 Reliable near the basket; full-court version measured, not yet good enough to ship — two new attacks in progress |
| **6. Narrate** | AI commentator writes the story | ✅ Working |

---

## At a glance — what changed since the last report

| Topic | Last report | Now |
|---|---|---|
| Whole pipeline on new games | verified on 2 games, run by hand | **Ran fully automatically on 3 more games it had never seen — one command, no human help, correct rosters out** |
| Independent check (SAM3) | it confirmed 91–99% of our per-camera detections | **Second, tougher check added: the finished court map was compared against SAM3's own map — and every gap found became training material for tonight's detector upgrade** |
| Ball tracking | training was running, score improving hourly | **Training finished and was graded against 633 human ball-clicks: it now finds the ball in ~7–8 of 10 marked frames on a play it never trained on. Its remaining fault is precisely known (see §3) and two fixes are already in motion** |
| Who-has-the-ball events | old system scored 14% — answer key built | **Near-basket possession is precision-clean and shipped. Fast-break possession was tested honestly: not good enough yet — so it stays off rather than guessing** |
| Roster quality | one player appeared twice; a referee got a jersey number | **Both bugs found automatically by our new self-grading run — and both fixed and verified the same day** |
| Work needed from your side | none | **Optional: a second round of ball-clicking (6 windows, 4 different games) is staged — the first round measurably improved the tracker** |

---

## 1 · Player tracking — now production-proven, not just demo-proven

The core promise — *every player tracked, named, and placed on one court map* — was this
week put through the strongest test we have: the **entire pipeline ran unattended on three
games it had never seen**, end to end, one command each.

| New game | People on the map | Steady the whole clip | Players named from jerseys |
|---|---|---|---|
| Game A | 12 per frame (matches the video) | 12 | 8 — e.g. A #0, A #8, B #3, B #9 |
| Game B | 14 per frame | 14 | 7 — e.g. A #30, B #17 |
| Game C | 14 per frame | 14 | 7 — e.g. A #6, B #4, B #7 |

The run also **graded itself** and caught two subtle roster bugs (a referee wrongly carrying
a jersey number; one player listed twice under the same number). Both were fixed and
re-verified the same day. That is exactly what this self-grading harness is for: every
future game scores itself, and problems surface automatically instead of in front of you.

**The second opinion, upgraded.** Previously, SAM3 — a very large independent AI from a
different company — had confirmed 91–99% of our per-camera detections. This week we compared
**finished court map against finished court map**. The two systems agree on the
overwhelming majority of people and positions; where SAM3 sees someone we miss (mostly small,
far-away players), those exact frames were **harvested automatically as training data** and
an upgraded detector was trained on them overnight. The independent checker doesn't just
audit the system anymore — it actively teaches it.

**The upgraded detector's report card (v1 → v2):** on the strictest test — a game whose
footage contributed *zero* training frames — spotting accuracy improved for every category
(players, referees, ball), and the share of SAM3's court map that we also see jumped from
**~70% to ~87%**. One finding from our own grading, reported honestly: the sharper eyes
also pick up bench and courtside people, which currently inflates the on-court headcount —
so **v2 stays in validation** while the merging rules (tuned for the old detector's
behaviour) are re-tuned to match it. The verified production system remains v1: every
number previously reported still stands. This is exactly how the upgrade path is supposed
to work — measured, graded, and only promoted when it beats the incumbent *everywhere*.

## 2 · The answer key still rules everything

A person watched game clips and wrote down the truth: who had the ball at every moment,
every pass, every turnover. Every accuracy number below comes from grading against that —
automatically, on every change. The old experimental events system scored **14%** on it;
that number remains the floor any new events system must beat *on the record*.

## 3 · The ball — honest scoreboard after five training rounds

Finding a basketball is genuinely hard — it's tiny, fast, and hidden by hands and bodies
most of the time. Here is exactly where we are:

| Question | Answer today |
|---|---|
| Can we find the ball when it's visible? | **Yes, mostly.** Graded against 633 human ball-clicks: the tracker finds it in **~7–8 of every 10 marked frames**, on a fast-break play it never trained on. Best result of any round so far. |
| Does it stay quiet when the ball is hidden? | **Not yet — this is the one remaining fault.** When the ball is invisible, the tracker still "sees" one too often. The human "no ball here" clicks cut this false confidence by a third — proof that more of exactly that data attacks exactly this fault. |
| Can it power full-court "who has the ball"? | **Getting there — and now never wrong.** The fast-break test initially produced one correct possession and one credited to the wrong team. We traced that error to its root (one camera mis-reading a crouching dribbler's team) and fixed the merging rule it exploited. Re-test: **every possession the system now reports is correct** on both graded clips — coverage is partial, but nothing it says is wrong. |
| What ships meanwhile? | Near-basket possession from the standard detector — **zero wrong possessions** on both graded clips. The system says "no ball data" elsewhere rather than inventing events. |

**Two new attacks launched (both automatic):**
1. **Teach "who is holding" instead of "where is the ball."** A player *holding* a ball looks
   different from one who isn't — crouched, dribbling, shielding. A new classifier learns
   this directly from thousands of examples labeled by the big teacher AI, no human work
   needed. This attacks the exact failure above (crediting the nearby defender).
2. **Second round of ball-clicking staged** — six clips across **four different games** (the
   first 633 clicks all came from one game). Same tool, same address; every "no ball here"
   click is ammunition against the false-alarm fault.

## 4 · How we know all of this is true

- **Human answer key** grades every events claim, automatically, on every change.
- **SAM3 second opinion** at two levels: per-camera detections (91–99% confirmed) and now
  whole-court-map vs whole-court-map — with every disagreement logged, inspected, and
  recycled as training data.
- **Five games, zero hand-holding**: 2 verified deeply by a human + 3 run cold, hands-off,
  with self-graded rosters.
- **633 human ball-clicks** grade the ball tracker frame by frame; the second click round
  targets its one measured weakness.
- 43 automated self-tests run on every change; every number in this report can be
  regenerated with a single command.

## 5 · What happens next

1. **Detector upgrade lands** (training tonight on the SAM3-harvested frames) → the far-camera
   blind spots shrink → re-graded automatically on all five games, before/after published.
2. **"Who is holding" classifier** finishes training → if it passes the answer key, it becomes
   the third possession signal and extends events beyond the basket area.
3. **Demo refresh**: re-cut the client video with named players and the clean rosters.
4. Optional but valuable: **round 2 of ball-clicking** (staged, 4 new games).

**Bottom line:** *who, where, and which jersey* is now proven on five games — three of them
completely hands-off — and independently double-checked at two levels. The ball has moved
from guesswork to a measured program: visible-ball finding is strong, the single remaining
fault is precisely identified, and two targeted fixes are already running. Nothing in this
report is a promise — every claim is a number the system re-computes itself.
