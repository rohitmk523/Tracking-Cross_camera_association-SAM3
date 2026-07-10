import glob
import os.path as osp
from collections import defaultdict

import numpy as np

from ..dataset import ImageDataset


class UballKPR(ImageDataset):
    """Basketball jersey-confirmed crops (auto-labeled, no human annotation).

    Layout: <root>/uball_kpr/{images,keypoints,negatives}/{pid}/{pid}_c{camid}_f{frame}_d{det}.jpg|.npy
    Identity = jersey number x kit (roster-driven split). Keypoints are RTMPose 17x3
    in crop coordinates (positive prompt); negatives are other persons inside the crop.
    Split: per identity by frame order — first 80% train, last 20% alternating query/gallery.
    """

    dataset_dir = "uball_kpr"
    train_dir = "images"
    MIN_CROPS = 50

    masks_dirs = {
        "keypoints": (17, False, ".npy", ["p{}".format(x) for x in range(1, 17)],),
        "keypoints_gaussian": (17, False, ".npy", ["p{}".format(x) for x in range(1, 17)],),
        "pose_on_img_crops": (35, False, ".npy", ["p{}".format(x) for x in range(1, 35)]),
    }

    @staticmethod
    def get_masks_config(masks_dir):
        return UballKPR.masks_dirs.get(masks_dir)

    def __init__(self, root="", **kwargs):
        root = osp.abspath(osp.expanduser(root))
        base = osp.join(root, self.dataset_dir)
        img_dir = osp.join(base, "images")

        by_pid = defaultdict(list)
        for path in sorted(glob.glob(osp.join(img_dir, "*", "*.jpg"))):
            pid_name = osp.basename(osp.dirname(path))
            by_pid[pid_name].append(path)
        by_pid = {k: v for k, v in by_pid.items() if len(v) >= self.MIN_CROPS}
        pid2label = {k: i for i, k in enumerate(sorted(by_pid))}

        def build(path, pid_name):
            stem = osp.splitext(osp.basename(path))[0]
            parts = stem.rsplit("_", 3)
            camid = int(parts[1][1:])
            frame = int(parts[2][1:])
            kp = np.load(path.replace("/images/", "/keypoints/").replace(".jpg", ".npy"))
            neg = np.load(path.replace("/images/", "/negatives/").replace(".jpg", ".npy"))
            return {"img_path": path, "pid": pid2label[pid_name], "camid": camid,
                    "keypoints_xyc": kp.astype(np.float32),
                    "negative_kps": neg.astype(np.float32), "frame": frame}

        train, query, gallery = [], [], []
        for pid_name, paths in sorted(by_pid.items()):
            samples = sorted((build(p, pid_name) for p in paths), key=lambda s: (s["frame"], s["camid"]))
            n_tr = int(0.8 * len(samples))
            train.extend(samples[:n_tr])
            held = samples[n_tr:]
            query.extend(held[0::2])
            gallery.extend(held[1::2])
        super().__init__(train, query, gallery, **kwargs)
