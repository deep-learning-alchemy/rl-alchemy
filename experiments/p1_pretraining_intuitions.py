"""Problem 1: do the intuitions from pre-training transfer to policy gradients?

    uv run python -m experiments.p1_pretraining_intuitions

(a)-(e) The pre-training experiments. Every comparison needs its own tuning.
For learning-rate-sensitive axes (batch size, weight decay, schedules) sweep
the learning rate per setting. For model size use experiments.common.fair():
zero-initialized residual blocks, a learning rate adapted to a fixed KL per
update with a width-aware starting point, and optionally a value network as
wide as the policy.
"""
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from analysis import group, interpolate, load_runs, savefig
from experiments.common import DEFAULT, fair, final, grid
from launch import launch

BASE = replace(DEFAULT, tags=("p1",))


def lr_batch_grid(seeds=range(3)):
    """(a) Learning rate x episodes per update. Turn on the gradient noise scale."""
    return grid(replace(BASE, noise_scale_every=10), "p1", seeds,
                lr=[1e-5, 3e-5, 1e-4], batch_episodes=[8, 32, 128])


def lr_wd_grid(seeds=range(3)):
    """(b) Is performance a function of lr * weight_decay?"""
    return grid(BASE, "p1", seeds, lr=[1e-5, 3e-5, 1e-4], weight_decay=[0.0, 0.01, 0.03, 0.1, 0.3])


def width_runs(widths=(16, 64, 256, 1024), seeds=range(3), scale_value=False):
    """(c) Policy width with fair tuning."""
    return [replace(fair(BASE, w, 1, scale_value), seed=s,
                    run_name=f"p1-width{w}{'-vscaled' if scale_value else ''}-s{s}")
            for w in widths for s in seeds]


def depth_runs(depths=(1, 2, 4, 8), width=64, seeds=range(3)):
    """(c) Policy depth with fair tuning."""
    return [replace(fair(BASE, width, d), seed=s, run_name=f"p1-w{width}-depth{d}-s{s}")
            for d in depths for s in seeds]


def untuned_size_controls(seeds=range(3)):
    """(c) The same sizes with the default fixed learning rate and initialization."""
    return [replace(BASE, seed=s, hidden=(64,) * d, run_name=f"p1-untuned-depth{d}-s{s}")
            for d in (1, 4) for s in seeds]


SCHEDULES = {
    "constant": dict(lr_schedule="constant"),
    "linear": dict(lr_schedule="linear"),
    "cosine": dict(lr_schedule="cosine"),
    "warmup-cosine": dict(lr_schedule="cosine", warmup_frac=0.05),
    "wsd": dict(lr_schedule="wsd", decay_frac=0.2),
}


def schedules(seeds=range(3)):
    """(d) Each schedule at two peak learning rates."""
    return [c for name, kw in SCHEDULES.items() for c in grid(replace(BASE, **kw), f"p1-sched-{name}", seeds,
                                                               lr=[3e-5, 1e-4])]


def averaging(seeds=range(3), fork_at=(1_000_000, 2_500_000)):
    """(e) EMA of weights, independent seeds, and forks of a shared parent."""
    configs = [replace(BASE, run_name=f"p1-ema-s{s}", seed=s, ema_decay=0.9) for s in seeds]
    configs += [replace(BASE, run_name=f"p1-indep-s{s}", seed=s) for s in seeds]
    configs.append(replace(BASE, run_name="p1-parent", seed=0, save_checkpoints=fork_at))
    configs += [replace(BASE, run_name=f"p1-fork{k}-d{d}", init_from=f"p1-parent/pi_{k}.pt", data_seed=100 + d,
                        total_env_steps=BASE.total_env_steps - k) for k in fork_at for d in range(2)]
    return configs


def size_table(pattern, label_key="hidden"):
    for value, runs in group(load_runs(pattern), label_key).items():
        fwd = np.mean([final(r) for r in runs])
        lr = np.mean([r.rows[-1]["lr"] for r in runs])
        print(f"{str(value):>14s}  forward progress={fwd:6.0f}  settled lr={lr:.1e}  (n={len(runs)})")


def plot_interpolation(pairs, name="p1_interpolation.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, (a, b) in pairs.items():
        rows = interpolate(a, b, episodes=8)
        ax.plot([r["alpha"] for r in rows], [r["eval_true_return"] for r in rows], marker="o", label=label)
    ax.set(xlabel="alpha", ylabel="forward progress of (1-alpha) theta_a + alpha theta_b")
    ax.legend(fontsize=8)
    return savefig(fig, name)


if __name__ == "__main__":
    launch(lr_batch_grid() + lr_wd_grid() + schedules(), modal=True)
    # Depth 1 at width 64 is the same run as width 64, so depth_runs starts at 2.
    launch(width_runs() + width_runs(widths=(1024,), scale_value=True) + depth_runs(depths=(2, 4, 8))
           + untuned_size_controls(), modal=True)
    launch(averaging(), modal=True)  # the forks start after their parent
    size_table("p1-width*")
    size_table("p1-w64-depth*")
