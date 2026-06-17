# Rim+ball RF-DETR training runs (retire the YOLO models)

Retraining the far + near **ball+hoop** detectors with **RF-DETR** (Apache) so the
shippable stack has **zero ultralytics/YOLO**. The annotated data already exists in
YOLO *format* (RF-DETR reads it directly) — no re-annotation. Prep:
`scripts/prep_rim_datasets.py`; launch: `scripts/aws_train.py --config <cfg>`.

## Runs (launched 2026-06-17, g5.2xlarge / A10G 24GB, self-terminating)

| run | config | dataset (train/valid) | classes | instance |
|---|---|---|---|---|
| far  | `configs/train_rim_far.yaml`  | rim_far 3678/1044  | Basketball, Basketball Hoop | `i-0f460ac2c174fd69a` |
| near | `configs/train_rim_near.yaml` | rim_near 4386/324  | Basketball, Basketball Hoop | `i-0bfcf99e91bbdf4e9` |

RF-DETR-**Small @1280**, batch 4 × grad-accum 4 (eff. 16), 60 epochs, early-stop
patience 12.

### OOM fix (important)
First far attempt used batch 8 → **CUDA OOM on the A10G** (~20 GB) and self-terminated
(~$0.12). Fix: **batch 8→4, grad-accum 2→4** + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
in the userdata. Batch 4 trains stably (≈10 min/epoch).

### Early signal (far, epoch 0)
`val/mAP_50_95 = 0.689`, EMA mAP `0.724` after one epoch — ball+hoop converges fast.

## Watch / fetch / evaluate
```bash
set -a; source .env; set +a            # AWS creds
# watch a run
aws s3 cp s3://uball-videos-production/_tmp_rfdetr_train/rfdetr-rim-far-v1_best_train.log -
aws s3 cp s3://uball-videos-production/_tmp_rfdetr_train/rfdetr-rim-near-v1_best_train.log -
# instance state (terminated => done; weights uploaded)
aws ec2 describe-instances --region us-east-1 --instance-ids i-0f460ac2c174fd69a i-0bfcf99e91bbdf4e9 \
  --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output text
# pull trained weights when done
python scripts/aws_train.py --config configs/train_rim_far.yaml  --fetch
python scripts/aws_train.py --config configs/train_rim_near.yaml --fetch
```
Weights land at `s3://uball-videos-production/_tmp_rfdetr_train/rfdetr-rim-{far,near}-v1_best.pth`.

## Follow-ups
- Evaluate far/near on their held-out test splits (extend the eval harness to the
  Basketball/Hoop taxonomy, or use RF-DETR's built-in `run_test`).
- Far includes 1250 old-facility `frame_*` frames — ablate (our-data-only) if it hurts.
- ⚠️ **Rotate the exposed admin AWS keys** once both runs finish (they were used
  un-rotated by operator authorization; see `.env` note + docs/11).

> Next: **Model A** (player+referee+ball) — needs the 1500-frame annotation first.
