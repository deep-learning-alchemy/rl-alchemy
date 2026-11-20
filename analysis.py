"""Load runs, plot metrics, and average or interpolate saved policies.

    from analysis import load_runs, plot_metric
    runs = load_runs("p2-lrbatch-*")
    plot_metric(runs, "eval_return", group_by="batch_episodes")

A Run holds its config (a dict) and its metric rows. run.series(key) returns
(x, y) for the rows where key was logged, e.g. eval_* keys only appear every
eval_every steps.
"""
import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils import runs_dir


@dataclass
class Run:
    name: str
    config: dict
    rows: list

    def series(self, key, x="env_step"):
        pairs = [(r[x], r[key]) for r in self.rows
                 if r.get(key) is not None and r.get(x) is not None]
        if not pairs:
            return np.array([]), np.array([])
        xs, ys = zip(*pairs)
        return np.asarray(xs, float), np.asarray(ys, float)

    def column(self, key):
        """One value per row (NaN where missing), aligned with self.rows."""
        return np.array([np.nan if r.get(key) is None else r[key] for r in self.rows], float)

    def __getitem__(self, key):
        return self.config[key]


def load_run(name, root=None):
    path = Path(root or runs_dir()) / name
    rows = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines() if line]
    return Run(name, json.loads((path / "config.json").read_text()), rows)


def load_runs(pattern="*", root=None, tag=None):
    root = Path(root or runs_dir())
    runs = []
    for path in sorted(root.iterdir()) if root.exists() else []:
        if fnmatch.fnmatch(path.name, pattern) and (path / "metrics.jsonl").exists():
            run = load_run(path.name, root)
            if tag is None or tag in run.config.get("tags", ()):
                runs.append(run)
    return runs


def group(runs, key):
    """{config value: [runs]} -- e.g. group seeds by replay_ratio."""
    out = {}
    for run in runs:
        value = run.config.get(key)
        out.setdefault(tuple(value) if isinstance(value, list) else value, []).append(run)
    return dict(sorted(out.items(), key=lambda kv: str(kv[0]) if kv[0] is None else kv[0]))


# ----------------------------------------------------------------------------
# Collapse detection and predictor scoring
# ----------------------------------------------------------------------------

def smooth(y, window):
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(np.pad(y, (window // 2, window - 1 - window // 2), mode="edge"), kernel, "valid")


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

def plot_metric(runs, key, x="env_step", group_by=None, ax=None, window=1, logy=False,
                individual=False, label_fmt="{key}={value}"):
    """Mean +- standard error over seeds, one line per value of `group_by`."""
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    groups = group(runs, group_by) if group_by else {None: runs}
    for value, members in groups.items():
        curves = [run.series(key, x) for run in members]
        curves = [(cx, smooth(cy, window)) for cx, cy in curves if len(cx)]
        if not curves:
            continue
        label = label_fmt.format(key=group_by, value=value) if group_by else key
        if individual:
            color = None
            for i, (cx, cy) in enumerate(curves):
                line, = ax.plot(cx, cy, lw=1, alpha=0.8, color=color, label=label if i == 0 else None)
                color = line.get_color()
            continue
        grid = curves[0][0]
        for cx, _ in curves[1:]:
            grid = grid[np.isin(grid, cx)]
        ys = np.stack([np.interp(grid, cx, cy) for cx, cy in curves])
        mean = ys.mean(0)
        line, = ax.plot(grid, mean, label=label)
        if len(curves) > 1:
            err = ys.std(0, ddof=1) / np.sqrt(len(curves))
            ax.fill_between(grid, mean - err, mean + err, color=line.get_color(), alpha=0.2, lw=0)
    ax.set_xlabel(x)
    ax.set_ylabel(key)
    if logy:
        ax.set_yscale("log")
    if group_by or individual:
        ax.legend(fontsize=8)
    return ax


def savefig(fig, name):
    path = Path("plots") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote", path)
    return path


# ----------------------------------------------------------------------------
# Weight averaging and interpolation
# ----------------------------------------------------------------------------

def average_policies(specs, weights=None):
    """A policy whose parameters are the weighted average of saved policies
    ('<run>' or '<run>/<file>.pt'). All must share an architecture. The
    observation-normalization statistics are averaged too."""
    from reinforce import load_policy
    policies = [load_policy(spec)[0] for spec in specs]
    weights = np.full(len(policies), 1 / len(policies)) if weights is None else np.asarray(weights)
    averaged = policies[0]
    state = {k: sum(w * p.state_dict()[k] for w, p in zip(weights, policies))
             for k in averaged.state_dict()}
    averaged.load_state_dict(state)
    return averaged


def evaluate_policy(policy, run, episodes=20, seed=123, env_kwargs=None):
    """Mean stochastic and greedy return of a policy in run's environment
    (with env_kwargs overriding the run's own settings)."""
    import torch
    from envs import make_env
    from reinforce import evaluate, load_config
    cfg = load_config(run.split("/")[0])
    kwargs = {**cfg.env_kwargs, **dict(env_kwargs or {})}
    envs = [make_env(cfg.env_id, 1000 * seed + i, kwargs) for i in range(episodes)]
    return evaluate(policy, envs, torch.Generator().manual_seed(seed))


def interpolate(spec_a, spec_b, alphas=np.linspace(0, 1, 11), episodes=20, seed=123, env_kwargs=None):
    """Returns along the straight line (1 - alpha) * theta_a + alpha * theta_b.
    A barrier (a dip below both endpoints) means the two solutions are not in
    one linearly connected basin."""
    rows = []
    for alpha in alphas:
        policy = average_policies([spec_a, spec_b], [1 - alpha, alpha])
        rows.append(dict(alpha=float(alpha), **evaluate_policy(policy, spec_a, episodes, seed, env_kwargs)))
    return rows
