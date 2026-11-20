"""Problem 4: reward hacking on CoinLoop, a gridworld small enough to solve exactly.

    uv run python -m experiments.p4_reward_hacking

The agent trains on the proxy reward (the true reward plus +0.2 each time it
enters the charged coin cell); eval_true_return is the fraction of episodes
that reach the goal. eval_length = 50 means the agent ran out the clock.
Runs take minutes on two CPU cores.
"""
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from analysis import load_runs, plot_metric, savefig
from envs import coinloop_solve
from experiments.common import COINLOOP
from launch import launch

GAMMAS = (0.8, 0.9, 0.95, 0.99)
COIN = replace(COINLOOP, tags=("p4",))


def coinloop_predictions(gammas=GAMMAS, **env_kwargs):
    """(a) The exactly optimal proxy behavior (value iteration) when episodes never
    end, and in 50-step episodes with a policy that can see the time."""
    rows = []
    for g in gammas:
        for aware in (False, True):
            sol = coinloop_solve(g, horizon_aware=aware, **env_kwargs)
            rows.append(dict(gamma=g, horizon_aware=aware, **sol))
            print(f"gamma={g:<5g} horizon_aware={aware!s:<5}  coins={sol['coins']:2d}  "
                  f"reaches_goal={sol['reaches_goal']!s:<5}  proxy={sol['proxy_return']:.2f}  "
                  f"true={sol['true_return']:.2f}")
    return rows


def coinloop(gammas=GAMMAS, time_obs=(False, True), seeds=range(3)):
    """(a) When does REINFORCE farm the coin instead of finishing?"""
    return [replace(COIN, run_name=f"p4-coin-g{g:g}-time{int(t)}-s{s}", seed=s, gamma=g,
                    env_kwargs=dict(time_obs=t))
            for g in gammas for t in time_obs for s in seeds]


# (b) KL-regularized optimization of the proxy, as in RLHF: maximize the proxy
# minus c_KL * KL(pi || pi_ref). The reference is trained only briefly on the
# true reward (coin_reward=0 makes the proxy equal the true reward), so it
# reaches the goal in only about 85% of episodes.
REFERENCE = replace(COIN, run_name="p4-ref", gamma=0.99, env_kwargs=dict(coin_reward=0.0), batch_episodes=4,
                    total_env_steps=600)
KL_COEFS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)


def kl_sweep(kl_coefs=KL_COEFS, seeds=range(3), steps=200_000):
    finetune = [replace(COIN, run_name=f"p4-kl{c:g}-s{s}", seed=s, gamma=0.99, init_from=REFERENCE.name(),
                        kl_ref=REFERENCE.name(), kl_coef=c, total_env_steps=steps, eval_episodes=50)
                for c in kl_coefs for s in seeds]
    return [REFERENCE] + finetune


def plot_proxy_vs_true(pattern="p4-coin-*", group_by="gamma", name="p4_coinloop.png"):
    runs = load_runs(pattern)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_metric(runs, "eval_return", group_by=group_by, ax=axes[0])
    plot_metric(runs, "eval_true_return", group_by=group_by, ax=axes[1])
    axes[0].set_title("proxy return (what the agent optimizes)")
    axes[1].set_title("true return (fraction reaching the goal)")
    return savefig(fig, name)


def plot_kl_frontier(pattern="p4-kl*"):
    """Gao et al.-style frontier: true and proxy return against KL to the reference."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for run in load_runs(pattern):
        kl = run.column("kl_ref")
        proxy, true = run.column("eval_return"), run.column("eval_true_return")
        keep = np.isfinite(proxy) & np.isfinite(kl)
        ax.scatter(kl[keep], true[keep], s=8, c="C0", label="true" if run.name.endswith("s0") else None)
        ax.scatter(kl[keep], proxy[keep], s=8, c="C1", label="proxy" if run.name.endswith("s0") else None)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set(xlabel="KL(pi || pi_ref) on visited states", ylabel="eval return")
    ax.legend(fontsize=8)
    return savefig(fig, "p4_kl_frontier.png")


if __name__ == "__main__":
    coinloop_predictions()
    launch(coinloop() + kl_sweep(), modal=True)
    plot_proxy_vs_true()
    plot_kl_frontier()
