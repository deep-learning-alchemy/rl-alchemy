"""Problem 2: learning from off-policy data -- no correction, importance sampling,
clipped importance ratios, and PPO.

    uv run python -m experiments.p2_off_policy

The off-policy setup: the policy that collects each batch (the behavior policy
mu) is the policy from `policy_lag` batches ago, and each batch is used for one
pass of 32 minibatch steps. Every variant keeps the default gradient-norm
clipping (0.5); "ppo-full" is the default agent (10 passes per batch) acting
with stale policies.
Every run logs the distance between the current and behavior policies
(behavior_kl_*, ratio_*). Tune the learning rate of every variant (LRS below).
"""
from dataclasses import replace

from analysis import load_runs, plot_metric, savefig
import numpy as np

from experiments.common import DEFAULT, final, grid
from launch import launch

BASE = replace(DEFAULT, tags=("p2",))
OFF = replace(BASE, policy_lag=4, epochs_per_batch=1)  # the off-policy setup (default gradient-norm clipping)
LAGS = [0, 4, 16]
LRS = [1e-5, 3e-5, 1e-4]  # extend the range if the best rate is at an end

# (a)-(e) The variants, from no correction to full PPO, all with the default advantages.
NO_IS = dict(surrogate="reinforce", importance_weights=False)
CORRECTIONS = {
    "none": NO_IS,
    "is": dict(surrogate="reinforce"),             # detached weight times the score
    "is-clip1": dict(surrogate="reinforce", is_clip=1.0),
    "is-clip2": dict(surrogate="reinforce", is_clip=2.0),
    "is-clip5": dict(surrogate="reinforce", is_clip=5.0),
    "ppo": dict(surrogate="ppo"),                  # clipped ratio objective, epsilon 0.2
    "ppo-full": dict(surrogate="ppo", epochs_per_batch=10),  # the default agent, stale actors
}

def slug(name):
    import re
    return re.sub(r"[^a-z0-9+-]+", "-", name.lower()).strip("-")


def setup_runs(seeds=range(3)):
    """(a, b) No correction and importance sampling across lags; also measures how stale the data is."""
    return [c for name in ("none", "is")
            for c in grid(replace(OFF, **CORRECTIONS[name]), f"p2-setup-{name}", seeds, policy_lag=LAGS)]


def correction_runs(seeds=range(3), lrs=LRS, lag=4):
    """(a)-(e) Every variant at one lag, each with its own learning rate."""
    return [c for name, kw in CORRECTIONS.items()
            for c in grid(replace(OFF, policy_lag=lag, **kw), f"p2-corr-{name}-lag{lag}", seeds, lr=lrs)]


def best_lrs(lag=4, names=("none", "is", "is-clip2", "ppo", "ppo-full")):
    """Each variant's learning rate with the best mean final forward progress at this lag."""
    def score(name, lr):
        return np.mean([final(r) for r in load_runs(f"p2-corr-{name}-lag{lag}-lr{lr:g}-s*")])
    return {name: max(LRS, key=lambda lr: score(name, lr)) for name in names}


def lag_runs(best_lr, seeds=range(3), names=("none", "is", "is-clip2", "ppo", "ppo-full")):
    """(c)-(e) The main variants across staleness, at each one's tuned learning rate (best_lr: name -> lr)."""
    return [c for name in names
            for c in grid(replace(OFF, lr=best_lr[name], **CORRECTIONS[name]), f"p2-lag-{name}", seeds,
                          policy_lag=LAGS)]


def plot_runs(pattern, group_by, name, keys=("eval_true_return", "eval_return", "is_weight_max", "kl_update")):
    import matplotlib.pyplot as plt
    runs = load_runs(pattern)
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    for key, ax in zip(keys, axes):
        plot_metric(runs, key, group_by=group_by, ax=ax, logy=key in ("kl_update", "is_weight_max"))
    return savefig(fig, name)


if __name__ == "__main__":
    launch(setup_runs() + correction_runs(), modal=True)
    best = best_lrs()
    print("tuned learning rates at lag 4:", best)
    # Lag 4 is already covered by correction_runs at the same learning rates.
    launch([c for c in lag_runs(best) if c.policy_lag != 4], modal=True)
    plot_runs("p2-setup-*", "policy_lag", "p2_setup.png")
