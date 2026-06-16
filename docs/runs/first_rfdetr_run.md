# First RF-DETR training run — plan (Week 1 D3–4)

The Week-1 D1–2 foundation is done: consolidated dataset, eval harness, AWS
scaffold. This is the confirmed plan for the **first RF-DETR training run** that
D1–2 gates (acceptance: *"a confirmed plan for the first RF-DETR training run"*).

## What we train
- **Model:** RF-DETR-**Small** (Apache-2.0; DINOv2 backbone) — docs/02, docs/04.
- **Resolution:** **1280** (ball ≈13 px at the far endline; 640 is a non-starter).
- **Classes:** `player, referee, ball` (canonical 0/1/2).
- **Data (OURS ONLY):** `data/detect_consolidated` — built by
  `scripts/build_detection_dataset.py` from the 4Cam set.
  - train = **e6fba750** · 1175 imgs (7153 player / 475 ref / **0 ball**)
  - valid = test = **c2a354fe** (held-out per docs/14) · 1200 imgs (9906 / 220 / 0)
- **Epochs:** 60, batch 8, grad-accum 2, early-stop patience 12 (on cross-game val mAP).

## How to launch
```bash
# 0. PREREQUISITE: rotate the flagged AWS keys (docs/11), put them in ~/.aws / .env
python scripts/aws_train.py --dry-run            # verify bundle + plan (no AWS)
python scripts/aws_train.py --i-rotated-creds    # bundle->S3, launch g5.2xlarge, train, self-terminate
aws s3 cp s3://uball-videos-production/_tmp_rfdetr_train/rfdetr_s_1280_ourdata_v1_best.pth.../train.log -   # watch
python scripts/aws_train.py --fetch              # pull best weights -> runs/
```
- Instance: **g5.2xlarge** (A10G 24 GB), DL AMI (verify `ami-012ba162b9cd2729c`),
  120 GB gp3, `InstanceInitiatedShutdownBehavior=terminate` (self-terminates).
- Bundle: ~869 MB (2375 train+valid images). Creds via `~/.aws`/env only —
  **never committed**; launch is **blocked** on the flagged account until rotation
  is confirmed.

## Expected cost / time (rough)
- g5.2xlarge on-demand ≈ \$1.2/hr. RF-DETR-S @1280, ~2.4k imgs, 60 ep ≈ 2–4 hr
  (early-stop likely sooner) → **≈ \$3–6** per run. *GPU turnaround is wall-clock,
  not code-speed (docs/12) — queue A/B variants in parallel.*

## Evaluate after (same harness, held-out game)
```bash
python scripts/aws_train.py --fetch
python scripts/eval_detection.py --detector rfdetr \
  --weights runs/rfdetr_s_1280_ourdata_v1/best.pth --model small --split test
```
Report vs the **baseline** in `runs/eval/` (the existing e6 nano RF-DETR on c2a):
mAP@50 / mAP@[50:95] per class **and the far-endline band player recall** (the
headline). Beat the baseline, with the band recall as the figure of merit (docs/13).

## Known gaps to close on the next iteration (tracked, not blockers)
1. **`ball` = 0 instances** — the consolidated set has no ball labels. The model
   cannot learn ball from this data; **annotate ball** (priority batch, docs/10)
   before relying on ball detection. Train run still valid for player/ref.
2. **val == test** (single held-out game c2a). Early-stop uses the held-out game;
   reported test == val. **Annotate a 3rd game** (from the *fresh* set) to separate
   them (`configs/detection_dataset.yaml` → `split_map`, one-line change).
3. **SAHI** (docs/04, biggest far-endline lever) is scaffolded
   (`configs/train_rfdetr.yaml: sahi`) but OFF for run 1 — measure the plain @1280
   baseline first, then add sliced fine-tune + far-endline-ROI sliced inference.
4. **Single court** — all footage is court-a, so this is cross-**game**, not yet
   cross-**court**. True cross-court eval needs a second venue.

## A/B baseline (internal, never shipped)
Keep a YOLO11-S + P2 internal baseline for an apples-to-apples small-object A/B
(docs/04). AGPL → benchmark only, never in the shippable build.
