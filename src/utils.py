"""Shared helpers: config loading, seeding, device, paths, JSON logging, timing.

Why: every script must use the SAME seeds and config so results are reproducible
and every fold's output is logged to disk.
"""
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FOLDS_DIR = RESULTS / "folds"
CKPT_DIR = RESULTS / "ckpt"
FIG_DIR = RESULTS / "figures"
TABLE_DIR = RESULTS / "tables"


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Seed python, numpy and torch (CPU + CUDA) so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fold_dir(subject: str) -> Path:
    d = FOLDS_DIR / subject
    d.mkdir(parents=True, exist_ok=True)
    return d


def _to_jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_to_jsonable)


def load_json(path):
    with open(path) as f:
        return json.load(f)


@contextmanager
def timer(name, store: dict | None = None):
    t0 = time.time()
    yield
    dt = time.time() - t0
    print(f"[time] {name}: {dt:.1f}s", flush=True)
    if store is not None:
        store[name] = dt


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
