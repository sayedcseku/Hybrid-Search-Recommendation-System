import os
import random
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
import tensorflow as tf
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Determinism is best-effort; enabling full determinism can slow training.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
