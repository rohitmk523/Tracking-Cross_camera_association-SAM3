# Cross-Camera Basketball Analysis — Progress

_Last updated: 2026-07-03_

**What it does:** four fixed cameras in → *who* is on court (identity, team, jersey), *where*
they are (one top-down map), *what happened* (possessions, passes), and an AI commentator's
play-by-play out.

| Stage | Status |
|---|---|
| **1. Detect** (players / referees / ball, every frame) | ✅ Measured: player 0.91 · referee 0.90 · ball 0.80 (held-out games) |
| **2. Track** (follow each person per camera + team) | ✅ Working, hardened |
| **3. Fuse** (4 cameras → one identity per player) | ✅ **Validated on a second, untouched game** |
| **4. Events** (possession / pass JSON) | 🔶 Engine ready — awaiting real ball tracking + ground-truth scoring |
| **5. Narrate** (AI play-by-play) | ✅ Working, grounded on verified identities |

## This week (44 commits) — what changed

**Built**
- Motion-based ball detection + 4-camera fusion machinery (now an explicit *experimental* flag — see honesty note)
- Jersey-number annotation round: tool (tight boxes, resume, undo), dataset builder, training scripts
- Set-of-marks demo video: same player = same number in **every** camera + live court map
- Ground-truth labelling tool (~10 min/clip) — turns event claims into measured accuracy

**Audited — then fixed the same week**
- Full adversarial audit of every subsystem: all claims re-measured from artifacts, methods checked against published research
- Found + fixed: silent camera de-sync (one camera ran 13 frames off — sync now strict + verified)
- Found + fixed: fisheye lens never corrected → recalibrated all 4 cameras (near-cam position error ~35cm → ~15cm)
- Retracted honestly: Wednesday's "full-court ball" followed players, not the ball — replacement plan is literature-backed
- Jersey data corrected **before** spending more annotation: queue now covers all 25 games, leak-proof train/val split

**Measured results** (benchmark window; ~13 real people)

| | Before | After |
|---|---|---|
| Identities created | 30 | **16** |
| Real people tracked ≥ half the clip | 12 | **13 (all)** |
| Phantom identities | 6 | **1** |
| Court-map flicker (dropout gaps) | 234 | **5** |
| Players per frame (median) | 11 | **13 = truth** |
| Frames over-counting | 28% | **~2%** |

**Verified — twice, independently**
- Cold-run on a second, untouched game: same quality profile (15 ids / 12 stable / clean 5-v-5 + 3 refs); also caught + fixed a black-vs-white-kit team-naming bug
- **SAM3** (independent state-of-the-art model, cloud GPU) cross-checked both games: **confirms 91–99% of everything we detect**; the far-camera recall gap it found is already compensated by fusion's zone design, and its disagreement frames are a free hard-example list for the next detector round

**Jersey numbers — plan upgraded, de-risked**
- Modern scene-text recogniser reads **78% of our crops with zero training** → fine-tuning needs fewer labels than planned
- Training is one command once labels land; "unclear" marks now train the legibility gate (nothing wasted)

## Where we are
- **Done + verified:** detect → track → fuse → narrate, with verified sync, corrected lenses, cross-game validation, independent SAM3 confirmation. 38 automated tests pass.
- **In progress (operator):** jersey labels — 99 usable across 8 games (target ~300–400); ground-truth event labels not started.
- **Next:** ① jersey recogniser fine-tune → named players end-to-end ② trained ball detector → trustworthy possession/pass events ③ score events against ground truth. *(Shot make/miss: separate repo.)*

**One line:** identity is solved and independently verified; jersey names and the ball tracker are the two remaining builds, and both have data pipelines, training code, and evaluation ready.
