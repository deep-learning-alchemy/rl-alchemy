"""Shared configurations and helpers for the problem starters (Humanoid-v5 throughout)."""
from dataclasses import replace

import numpy as np

from reinforce import PGConfig

# The default agent: Gymnasium's reward, start-state noise 0.01, PPO-style
# updates with GAE, a depth-1 residual MLP policy of width 64, fixed lr 3e-5.
# About an hour per 5M-step run on an 8-CPU Modal container.
DEFAULT = PGConfig()

# Problem 4's gridworld: plain REINFORCE (batch-mean baseline, one step per
# batch) with a small tanh MLP over the one-hot observation. Minutes per run.
COINLOOP = PGConfig(env_id="CoinLoop-v0", total_env_steps=300_000, batch_episodes=8, arch="mlp", hidden=(64, 64),
                    activation="tanh", surrogate="reinforce", gae_lambda=None, baseline="mean",
                    normalize_advantages=False, epochs_per_batch=1, minibatches=1, lr=3e-3, grad_clip=None,
                    obs_norm=False, reward_scale=1.0, env_workers=1, eval_every=5_000, eval_episodes=10)


def fair(cfg, width, depth=1, scale_value=False, target_kl=0.02):
    """Tuning that makes policy sizes comparable: zero-initialized residual
    blocks (every depth starts as the same function), a learning rate adapted
    to a fixed KL per update, a starting rate that shrinks as width^-0.5, and
    optionally a value network as wide as the policy. Widths of 128 and more
    run on a GPU."""
    return replace(cfg, arch="resmlp", activation="gelu", hidden=(width,) * depth, res_init="zero",
                   target_kl=target_kl, lr=2e-5 * (64 / width) ** 0.5,
                   value_hidden=(width, width) if scale_value else cfg.value_hidden,
                   device="cuda" if width >= 128 else "cpu")


def final(run, key="eval_true_return", last=3):
    """Mean of the last few evaluations (default: forward progress, the true return)."""
    _, y = run.series(key)
    return float(np.mean(y[-last:])) if len(y) else float("nan")


def mean_over_training(run, key="eval_true_return"):
    """Mean over all evaluations: a measure of sample efficiency."""
    _, y = run.series(key)
    return float(np.mean(y)) if len(y) else float("nan")


SHORT = dict(lr="lr", batch_episodes="b", weight_decay="wd",
             policy_lag="lag", grad_clip="gc", healthy_reward="healthy", init_log_std="logstd",
             entropy_coef="ent", kl_coef="kl")


def grid(base, prefix, seeds=range(3), **axes):
    """Cartesian product of config overrides, e.g. grid(DEFAULT, "p3-lrb", lr=[...], batch_episodes=[...])."""
    import itertools
    keys = list(axes)
    out = []
    for values in itertools.product(*axes.values()):
        tag = "-".join(f"{SHORT.get(k, k)}{v:g}" if isinstance(v, (int, float)) else f"{SHORT.get(k, k)}{v}"
                       for k, v in zip(keys, values))
        out += [replace(base, run_name=f"{prefix}-{tag}-s{s}", seed=s, **dict(zip(keys, values))) for s in seeds]
    return out
