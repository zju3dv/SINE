from shutil import copyfile, copytree, ignore_patterns
from datetime import datetime
import torch
import random
import numpy as np


def copy_files(src_dir, dst_dir, *ignores):
    copytree(src_dir, dst_dir, ignore=ignore_patterns(*ignores))


def make_source_code_snapshot(log_dir):
    copy_files(
        ".",
        f"{log_dir}",
        "ckpts",
        "results",
        "*.png",
        "*.jpg",
        "*.PNG",
        "*.JPG",
        "saved",
        "__pycache__",
        "data",
        "logs",
        "scans",
        ".vscode",
        "*.so",
        "*.a",
        ".ipynb_checkpoints",
        "build",
        "bin",
        "*.ply",
        "eigen",
        "pybind11",
        "*.npy",
        "*.pth",
        ".git",
        ".git*",
        "docs",
        "torchsearchsorted",
    )


def get_timestamp():
    return datetime.now().strftime(r"%y%m%d_%H%M%S")


def set_rand_seed(seed=1):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # 保证每次返回得的卷积算法是确定的


GLOBAL_WORKER_ID = None


def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_rand_seed(GLOBAL_SEED + worker_id)
