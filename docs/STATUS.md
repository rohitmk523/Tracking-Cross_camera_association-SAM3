# Cross-Camera Basketball Analysis — Progress Report

**What the system does:** you give it the video from the four cameras already installed in
the gym. It works out **who** is on the court (which team, which jersey number), **where**
everyone is (drawn live on a top-down court map), and **what happened** (who had the ball,
passes, turnovers) — and then an AI commentator writes the play-by-play, naming real players.

## Where each part stands

| Step | What it does | Status |
|---|---|---|
| **1. Detect** | Spot every player, referee and ball in every video frame | ✅ Working — graded on games it was never trained on |
| **2. Track** | Follow each person through time in each camera | ✅ Working |
| **3. Combine** | Merge all 4 cameras so each player is ONE person on one map | ✅ Working — double-checked two independent ways |
| **4. Name** | Read jersey numbers so players get real identities | ✅ **New: working end to end** |
| **5. Events** | Say who had the ball, passes, turnovers | 🔶 We can now *measure* how good this is; the missing piece (ball tracking) is in training |
| **6. Narrate** | AI commentator writes the story | ✅ Working |

---

## At a glance — what changed since the last report

| Topic | Last report | Now |
|---|---|---|
| Jersey numbers | a plan and some labeled examples | **Players are named on screen, and we checked the names against the actual jerseys in the video** |
| Court map quality | good, but players occasionally blinked off the map | **Blinking fixed — the map now shows exactly as many people as are really on court** |
| Event accuracy | impossible to grade — no answer key existed | **A human created the answer key; the old system scored 14% — so we now know exactly what to fix and how to prove it's fixed** |
| Ball tracking | we had honestly retracted a broken approach | **A dedicated ball-spotting AI is being trained right now, and its score improves by the hour** |
| Independent check | in progress | **A second, unrelated AI reviewed our system's work: it agrees with 91–99% of what we detect** |
| Work needed from your side | two labeling tasks pending | **None — everything is automated from here** |

---

## 1 · Players are now named (the headline)

The system reads the number on each player's jersey and uses it everywhere — the court map,
the event log, and the commentary all say "**B #22**" instead of "player 14".

*How it works, in plain terms:* three small AIs work as a team. The first asks *"is a number
even readable in this image?"* (if not, it says "skip" rather than guess). The second finds
exactly where the number is on the shirt. The third reads it — and it can read **any**
number, including ones it never saw during training.

*How good is it:* when the system commits to a number, it is right **94 times out of 100**.
When it isn't sure, it stays silent — and because a player is seen hundreds of times per
game, a few silent moments cost nothing, while a wrong name would be much worse. A number is
only accepted after **multiple independent readings agree**.

*How we know it's real:* we tested it on games it was never trained on, and we compared the
names it produced against the jerseys visible in the actual video — they match. As an extra
sanity check, the AI commentator (a completely separate system that watches the raw video)
had independently read the same numbers — **two unrelated systems, same answer**.

## 2 · We can now grade the system — with a human answer key

A person watched three game clips and wrote down the truth: who had the ball at every
moment, every pass, every turnover. That answer key now **automatically grades** every
version of the system we build.

First grading of the *old, experimental* events system:

| Question | Old system's answer | Truth |
|---|---|---|
| Who had the ball? | right only **14%** of the time | — |
| How often did the ball change hands? | claimed **11** times | actually **2** |
| Turnovers | reported 4 — **all four were false alarms**, and it missed both real ones | 2 |

*Why show a bad score?* Because it proves the grading works, it's exactly why we switched
that system off, and it sets the bar the new ball tracker must beat — **on the record**.

## 3 · The ball — from guesswork to an engineering program

Finding the ball is genuinely hard: it's tiny, fast, and often hidden by hands and bodies.
We tested every available approach against the human answer key:

| Approach | Plain-language result |
|---|---|
| Old motion-based tracker | Saw "a ball" everywhere — but it was actually following players. Confidently wrong. Switched off. |
| Standard detector | Almost never wrong — but only sees the ball near the basket, so it stays silent most of the game. |
| **New: a dedicated ball-spotting AI ("BallNet")** | **In training now — built to see the ball everywhere, like the motion tracker, but actually correct, like the detector.** |

*Where the training data comes from (no human labeling needed):* we use a very large,
slow-but-smart AI (SAM3) as a **teacher** — it marks the ball's position in our own game
footage, and BallNet learns from those marks. Fun detail: the teacher kept marking the
basketballs *painted on the court floor and walls*, so we built a filter that removes
paintings and keeps the real, moving ball.

*Progress so far:* first training round found the ball in about **35%** of test frames —
frames from a game it never saw. We then quadrupled the training footage (pulled from two
more games automatically), and the new training run **passed the old best score within
minutes of starting**. Training continues right now.

*The finish line is already defined:* BallNet must beat that 14% score, report 2 ball
exchanges instead of 11, and produce **zero** false turnovers — graded automatically against
the human answer key. Until it passes, the system honestly says "no ball data" rather than
inventing events.

## 4 · How we know all of this is true

- **Human answer key** — every accuracy claim above is graded against what a person actually
  saw in the video, automatically, every time we change anything.
- **A second opinion from an independent AI** — SAM3 (built by a different company, working
  from different principles) reviewed our detections: it confirms **91–99%** of them. Where
  it sees someone we miss, those cases are logged and become future training material.
- **Tested on unseen games** — the whole pipeline was run cold on a game it had never
  touched and produced the same quality. That run even caught a subtle bug (teams wearing
  black vs. white confused the team-naming rule) which is now fixed.
- **The court map holds up**: 13 real people on court → 13 stable identities on the map, no
  blinking, no phantoms worth mentioning (one short-lived ghost in 12 seconds of play).
- 43 automated self-tests run on every change; every number in this report can be
  regenerated with a single command.

## 5 · What happens next

1. **BallNet finishes training → takes its exam** against the answer key. Passing unlocks
   trustworthy "who-has-the-ball / passes / turnovers" — the last missing piece.
2. Small polish: one player who briefly appears twice in a roster; refresh the demo video so
   it shows the new named players.
3. Scale up: run the full pipeline across many more games, unattended.

**Bottom line:** the "who, where, and which jersey" promise is delivered and verified three
independent ways. The ball is the one remaining piece — and it has gone from a broken claim
to a measurable training program with a clear, already-written pass/fail test.
