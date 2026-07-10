"""Fine-tune KPR on the uball dataset. Wraps main-flow with a restricted unpickler
(the released checkpoint contains only audited data globals; nothing executable)."""
import functools
import pickle
import sys
import types

import numpy as np
import torch

sys.path.insert(0, ".")

AUDITED = {
    ("__builtin__", "set"): set, ("builtins", "set"): set,
    ("_codecs", "encode"): __import__("_codecs").encode,
    ("collections", "OrderedDict"): __import__("collections").OrderedDict,
    ("numpy", "dtype"): np.dtype,
    ("numpy.core.multiarray", "scalar"):
        __import__("numpy.core.multiarray", fromlist=["scalar"]).scalar,
    ("torch", "FloatStorage"): torch.FloatStorage,
    ("torch", "LongStorage"): torch.LongStorage,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("yacs.config", "CfgNode"): __import__("yacs.config", fromlist=["CfgNode"]).CfgNode,
}


class RU(pickle.Unpickler):
    def find_class(self, m, n):
        if (m, n) in AUDITED:
            return AUDITED[(m, n)]
        raise pickle.UnpicklingError(f"blocked global: {m}.{n}")


rp = types.ModuleType("rp")
rp.Unpickler = RU
rp.load = lambda f, **k: RU(f, **{x: v for x, v in k.items()
                                  if x in ("fix_imports", "encoding", "errors")}).load()
_orig_load = torch.load


def safe_load(*a, **k):
    k.pop("weights_only", None)
    k["pickle_module"] = rp
    k["weights_only"] = False
    return _orig_load(*a, **k)


torch.load = safe_load

from torchreid.scripts.builder import build_config, build_torchreid_model_engine
from torchreid.scripts.default_config import engine_run_kwargs

cfg = build_config(config_path="configs/kpr/imagenet/kpr_uball_ft.yaml")
cfg.use_gpu = torch.cuda.is_available()
print(f"use_gpu={cfg.use_gpu} save_dir={cfg.data.save_dir}")
engine, model = build_torchreid_model_engine(cfg)
engine.run(**engine_run_kwargs(cfg))
print("TRAINING DONE, save_dir:", cfg.data.save_dir)
