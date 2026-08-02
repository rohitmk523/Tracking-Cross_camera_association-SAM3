# Brief: Ball-Possession Ground Truth + Annotation Tool

**For:** Akhilesh · **Branch:** `cleanup/possession-core` · **Written:** 2026-08-02

You own two things here:
1. **Build** a small FastAPI annotation tool (in *your* repo/branch, not this one).
2. **Produce** the ground truth with it and send the JSON back.

Everything else in this doc is the context you need for those two jobs to be
correct rather than merely finished.

---

## 0. What changed in this repo, and why (read before you pull)

On 2026-08-01 we did a **possession-core pivot** and deleted ~18 GB and ~110
scripts. Don't be alarmed when files you remember are gone.

**What was removed:** shot detection, make/miss classification, event assembly,
zone labelling, plus every dead experiment (SAM3 era, KPR re-identification,
the twelve refuted WHO approaches, old renderers/solvers), the old fine-tune
caches, and the superseded plan docs.

**Why:** the product is now exactly one thing — *track the ball and the
players, know **who has the ball** at every frame, draw the ring under him.*
Shot/event logic was inherited scaffolding that no longer served that mission
and made every change slower to reason about. It wasn't wrong, it was
out-of-scope, and carrying it meant every refactor had to keep it alive.

**Nothing was destroyed.** Everything is in git history, and the working
copies are in `~/.Trash/uball_cleanup_2026-08-01/` with a `MANIFEST.txt`. If
you need something back, ask — don't re-implement it.

**What survives** is the 8-stage possession chain (see `README.md` and
`docs/GUIDE_POSSESSION_PIPELINE.md`), plus your ring work, which is the
delivery layer of that chain.

Also new since you last looked:
- **Unified 4-class detector** (`player/referee/ball/hoop`) replaces the two
  separate detectors — one batched pass instead of two full passes.
  Weight: `runs/unified_yolo26s_v1/best.pt` (mAP50-95 0.806, recall 0.882).
- **Speed bundle**: unified detection + no pose + jersey OCR at stride 2 took
  a 3-minute window from **37.4 min → 12.4 min** of compute (3.0×).
- That speed came with an **unmeasured accuracy cost** — which is precisely
  why we need your ground truth. See §1.

---

## 1. Why this GT matters (the actual reason, not a chore)

We can currently measure *speed* exactly and *accuracy* not at all. When the
speed bundle changed the holder timeline, the only thing we could say was
"the two runs agree on 57.5% of frames" — with no way to know which one was
right. Every tuning decision in the possession machine is currently made on
inference rather than evidence.

Your annotation converts that into a real scoreboard. With it we can tune the
holder state machine honestly, settle speed-vs-accuracy trades with numbers,
and state a defensible accuracy figure to the client.

**Target scale (agreed with Rohit): ~30 minutes of footage.**

| game | minutes | role |
|---|---|---|
| `c2a354fe` | 10 | tuning — hardest case (5 shared jersey numbers) |
| `e6fba750` | 10 | tuning — cleanest footage |
| `2c490f1a` | 10 | **held out** — nobody tunes against this |

Prefer **5 × 2-minute windows per game** over one contiguous 10 minutes: more
lineups, both halves, both attacking directions. Contiguous 10 min is often
just two lineups and teaches us less.

**Start with Phase 1: the single 3-minute window we already benchmarked**
(`c2a354fe`, 600–780 s). Deliver that, we validate the format end-to-end and
you find out whether your tool is pleasant before committing hours.

---

## 2. Build the tool (FastAPI, same shape as Training_frameworks)

Model it on
`Training_frameworks/Uball 4Cam Detection/src/annotate_server.py` +
`annotate_ui.html` — same pattern: FastAPI backend, single HTML page,
`uvicorn` on `127.0.0.1:8000`, resumable state on disk.

```
your_repo/
  src/possession_annot_server.py   # FastAPI
  src/possession_annot_ui.html     # single page
  data/gt/                         # output JSON lands here
```

### Inputs we give you (per window)
| file | what it is |
|---|---|
| `local_bench_c2a_600_180.mp4` | the rendered 4-angle video (2×2 grid + panel) |
| `holders_c2ademo.json` | **our prediction** — `{"fps":29.97,"segments":[[start_frame,end_frame,"n1W"],...]}` |
| `c2a354fe.json` (roster) | `{"players":[{"num":2,"name":"Trent Wasser","team":1},...]}`, plus `team1_color`/`team2_color` |

**Frame mapping — get this right, everything depends on it:**
```
global_frame = round(window_start_seconds * 29.97) + round(video_time_seconds * 29.97)
# for the Phase-1 window: window_start_seconds = 600  -> base frame 17982
```
Our segments use **global frames**. Emit global frames back. If you emit
video-relative frames instead, say so explicitly in the JSON (`"frame_base"`)
and we'll convert — silent mismatch is the one error that ruins the dataset.

**Stream ids:** `n2` = jersey #2. A trailing `B`/`W` marks the kit for numbers
worn by both teams (`n1B` vs `n1W` — c2a has five such: #1, #2, #3, #4, #8).
Map to names via the roster: number + team (B/W → team per the roster's
`kit_team` if present, else B=team1, W=team2).

### The core interaction — correction, not creation
Do **not** build frame-by-frame labelling. 3 min × 4 cameras × 10 players is
215,000 labels; nobody is doing that. Possession is **piecewise constant**:
you only record *change points*.

Required behaviour:
1. Video plays; the current prediction is shown large ("BALL: #2 Wasser").
2. **When the prediction is wrong, press the correct player's hotkey.** That
   writes a GT change-point at the current frame. Nothing else is needed.
3. `Space` = play/pause, `←/→` = ±1 frame, `,`/`.` = ±1 s, `J`/`L` = speed.
4. **`N` = nobody** (loose ball, in flight, dead ball) — a real and important
   label, don't skip it.
5. **`U` = unsure** — genuinely ambiguous (ball invisible in all 4 angles).
   These get excluded from scoring rather than guessed. Being able to say
   "I can't tell" is a feature; forcing a guess poisons the GT.
6. **Jump to next prediction change** button — most efficient review path is
   to check each moment the prediction switches.
7. Timeline strip: prediction segments on top, your GT below, colour per
   player, so gaps and disagreements are visible at a glance.
8. Autosave to disk every change; resumable — you will not do this in one sitting.

### One methodological rule
Showing our prediction makes you faster but risks **anchoring** — accepting a
wrong prediction because it's already on screen. Mitigate with a `H` hotkey
that hides the prediction; use it for a random ~10% of the window and check
your blind answers against your assisted ones. If they diverge a lot, tell
us — that's information about the GT's reliability, not a failure.

### Output format (exactly this)
```json
{
  "game": "c2a354fe",
  "window": {"start_s": 600, "dur_s": 180, "fps": 29.97, "frame_base": 17982},
  "annotator": "akhilesh",
  "created": "2026-08-02",
  "frames_are": "global",
  "segments": [
    {"start": 17987, "end": 18042, "holder": "n1W", "conf": "sure"},
    {"start": 18043, "end": 18051, "holder": null,  "conf": "sure"},
    {"start": 18052, "end": 18120, "holder": "n3B", "conf": "unsure"}
  ]
}
```
- `holder: null` = nobody (loose/flight/dead).
- `conf: "unsure"` = excluded from scoring.
- Segments must be **contiguous and non-overlapping** across the whole window
  — every frame gets exactly one segment. Add a `/api/validate` endpoint that
  checks this and refuses to export otherwise; a gap silently becomes "no GT"
  on our side.

---

## 3. What you send back, and what we do with it

Send the JSON (per window). Ping Rohit when Phase 1 is ready.

On our side we run **the identical video and window** through the pipeline,
score our holder timeline against your segments, and produce:
- overall holder accuracy (frames, excluding `unsure`),
- accuracy at **change points** (the hard moments — catches and steals),
- an error anatomy: defender-swap vs pass-smear vs gap vs identity error,
- and the settled answer on whether the speed bundle's pose removal actually
  costs accuracy or not.

That last one is currently blocking a production config decision, which is
why Phase 1 matters more than its size suggests.

---

## 4. Ground rules (same as always)

- Never commit `data/`, `runs/`, or credentials.
- Don't run `scripts/aws_*` — they cost money and need Rohit's approval.
- The held-out game (`2c490f1a`) is annotated but **not analysed** by anyone
  until the end. Don't peek at our accuracy on it.
- Annotate what you see, not what you think the system should say. If our
  ring is confidently wrong, mark it wrong — that's the entire value of this.
- Ultralytics is AGPL/commercial: prototype freely, clear before shipping.

## 5. Definition of done (Phase 1)

- [ ] FastAPI tool runs locally, resumable, autosaving.
- [ ] The 3-minute c2a window fully annotated: contiguous segments, no gaps.
- [ ] `unsure` used where genuinely ambiguous, `null` where nobody holds it.
- [ ] Blind spot-check done on ~10% of the window; divergence reported.
- [ ] JSON validated by your own `/api/validate` and sent over.

Questions on frame mapping, stream ids, or anything ambiguous: ask before
annotating an hour in the wrong convention. That failure mode is cheap to
prevent and expensive to discover.
