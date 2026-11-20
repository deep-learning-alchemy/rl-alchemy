"""Student-facing configuration and small shared helpers."""
import os
import random
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent

# Student-facing configuration. Most students only edit these three lines.
CONFIG_MODAL_ENVIRONMENT = "YOUR_MODAL_ENVIRONMENT"
CONFIG_WANDB_ENTITY = "YOUR_WANDB_USERNAME_OR_TEAM"
CONFIG_WANDB_PROJECT = "rl-alchemy"

# Instructors usually leave these alone.
CONFIG_MODAL_APP_NAME = "rl_alchemy"
CONFIG_MODAL_VOLUME_NAME = "volume-rl-alchemy"
CONFIG_MODAL_WANDB_SECRET = "dl-alchemy-wandb"


def _present(value):
    return value is not None and value != "" and not str(value).startswith("YOUR_")


def config_str(name, *env_names, default=None):
    for env_name in env_names:
        if _present(os.environ.get(env_name)):
            return os.environ[env_name]
    value = globals().get(name)
    return str(value) if _present(value) else default


def runs_dir() -> Path:
    """Local runs live in ./runs; Modal containers point this at a Volume."""
    return Path(os.environ.get("RL_ALCHEMY_RUNS_DIR", REPO_ROOT / "runs"))


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
