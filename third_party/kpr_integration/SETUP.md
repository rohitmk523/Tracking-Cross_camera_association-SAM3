# KPR integration — rebuild /tmp/kpr after reboot

1. git clone --depth 1 https://github.com/VlSomers/keypoint_promptable_reidentification /tmp/kpr
2. cd /tmp/kpr && uv venv .venv310 --python 3.10 && source .venv310/bin/activate
   uv pip install torch torchvision numpy Cython h5py Pillow six scipy opencv-python \
     "matplotlib<3.9" future yacs gdown albumentations==1.3.1 pandas tabulate deepdiff \
     wandb monai torchmetrics==1.3.0 timm scikit-image tqdm omegaconf scikit-learn \
     tensorboard segment-anything "numpy<2"
3. Stub training-only dep: write .venv310/.../site-packages/cosine_annealing_warmup/__init__.py
   with a raising CosineAnnealingWarmupRestarts class.
4. Apply repo patches:
   - copy uball_dataset.py -> torchreid/data/datasets/image/uball.py
   - register in torchreid/data/datasets/__init__.py: import UballKPR + "'uball': UballKPR" entry
   - occluded_posetrack21.py TrackingSet.image_gt -> dataclasses.field(default_factory=...) (py3.11+)
   - visualize_query_gallery_rankings.py: guard empty q_matches (skip query, see kpr commit log)
5. Copy configs: kpr_uball_ft.yaml + kpr_uball_test.yaml -> configs/kpr/imagenet/
   train_uball.py -> repo root.
6. Weights: pretrained_models/kpr_occ_pt_IN_82.34_92.33_42323828.pth.tar (gdown 1Np5wu3nQa_Fl_z7Zw2kchJNC8JZVwsh5)
   Fine-tuned: pretrained_models/kpr_uball_ft.pth.tar (runs/kpr_ft/logs/700042914/.../job-700042914_12_model.pth.tar)
7. Dataset for retraining: data/kpr_finetune (scripts/build_kpr_dataset.py) -> /tmp/kpr/reid_data/uball_kpr/
LICENSE: Hippocratic HL3 — commercial check required before shipping.
