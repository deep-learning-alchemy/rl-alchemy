"""Problem 3: exploration and the optimization landscape (Humanoid).

    uv run python -m experiments.p3_exploration

Policy-gradient training is stochastic gradient ascent on J(theta). Every run
trains on Gymnasium's proxy reward; CRAWL removes the rule that a fall ends the
episode. (a)-(c) study the optimum it finds and what lets the optimizer escape it; (d) the
start-state distribution. Tune the learning rate for each new setting.
"""
from dataclasses import replace

import torch

from analysis import evaluate_policy, interpolate, load_runs
from experiments.common import DEFAULT, final
from launch import launch
from reinforce import load_policy

BASE = replace(DEFAULT, tags=("p3",))


# Every run trains on Gymnasium's proxy reward. CRAWL removes the rule that a
# fall ends the episode (the survival bonus is then paid only while upright);
# WALK is the default agent, with the rule. CRAWL runs are also evaluated
# under the default rule ("upright").
TASK_EVAL = (("upright", dict(terminate_when_unhealthy=True)),)
CRAWL = replace(BASE, env_kwargs=dict(terminate_when_unhealthy=False), extra_evals=TASK_EVAL)
WALK = BASE
CHECKPOINTS = (500_000, 1_000_000, 2_000_000, 3_000_000)


def local_optimum_runs(seeds=range(3)):
    """(a) The proxy reward without and with the fall rule, with checkpoints."""
    return ([replace(CRAWL, run_name=f"p3-crawl-s{s}", seed=s, save_checkpoints=CHECKPOINTS) for s in seeds]
            + [replace(WALK, run_name=f"p3-walk-s{s}", seed=s, save_checkpoints=CHECKPOINTS) for s in seeds])


def escape_runs(seeds=range(3)):
    """(b) Ways out of the crawl: action noise (entropy bonus, initial noise), gradient noise, and a
    larger survival bonus in the proxy reward."""
    variants = {"ent0.01": dict(entropy_coef=0.01),
                "anneal-ent0.01": dict(entropy_coef=0.01, entropy_coef_end=0.0, entropy_anneal_frac=0.3),
                "logstd0.5": dict(init_log_std=0.5),
                "b128": dict(batch_episodes=128),
                "lr1e-4": dict(lr=1e-4),
                # Changing the proxy reward itself: a larger survival bonus (default 5).
                "healthy10": dict(env_kwargs=dict(terminate_when_unhealthy=False, healthy_reward=10.0)),
                "healthy20": dict(env_kwargs=dict(terminate_when_unhealthy=False, healthy_reward=20.0))}
    return [replace(CRAWL, run_name=f"p3-crawl-{k}-s{s}", seed=s, **kw) for k, kw in variants.items() for s in seeds]


def continuation_runs(seeds=range(3)):
    """(c) Remove the fall rule from a walker: at the end of training, and after 2M steps."""
    return ([replace(CRAWL, run_name=f"p3-walker-nofall-s{s}", seed=s, init_from=f"p3-walk-s{s}")
             for s in seeds]
            + [replace(CRAWL, run_name=f"p3-curriculum-s{s}", seed=s, init_from=f"p3-walk-s{s}/pi_2000000.pt",
                       total_env_steps=3_000_000) for s in seeds])


def random_directions(spec, scales=(0.0, 0.03, 0.1, 0.3, 1.0), num_dirs=4, episodes=8, seed=0, env_kwargs=None):
    """Returns of theta + scale * |theta_layer| * d for random unit directions d."""
    policy, _ = load_policy(spec)
    base = {k: v.clone() for k, v in policy.state_dict().items()}
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for d in range(num_dirs):
        direction = {}
        for k, v in base.items():
            if v.dtype.is_floating_point and "norm" not in k and v.dim() > 0:
                r = torch.randn(v.shape, generator=gen)
                direction[k] = r / (r.norm() + 1e-12) * v.norm()
        for scale in scales:
            policy.load_state_dict({k: v + scale * direction[k] if k in direction else v for k, v in base.items()})
            rows.append(dict(direction=d, scale=scale, **evaluate_policy(policy, spec, episodes, env_kwargs=env_kwargs)))
    return rows


NOISE_LEVELS = (0.01, 0.05, 0.1, 0.2)  # evaluated
TRAIN_NOISE_LEVELS = (0.01, 0.05, 0.1)  # trained; 0.2 is held out (evaluated only)


def coverage_runs(seeds=range(3)):
    """(d) Train at different start-state noise levels; evaluate at all of them."""
    evals = tuple((f"noise{n:g}", dict(reset_noise_scale=n)) for n in NOISE_LEVELS)
    return [replace(BASE, run_name=f"p3-train-noise{n:g}-s{s}", seed=s, env_kwargs=dict(reset_noise_scale=n),
                    extra_evals=evals) for n in TRAIN_NOISE_LEVELS for s in seeds]


def coverage_table(pattern="p3-train-noise*"):
    for run in load_runs(pattern):
        cols = "  ".join(f"@{n:g}={final(run, f'eval_noise{n:g}_true_return'):6.0f}" for n in NOISE_LEVELS)
        print(f"{run.name:28s} {cols}")


if __name__ == "__main__":
    launch(local_optimum_runs() + escape_runs() + coverage_runs(), modal=True)
    launch(continuation_runs(), modal=True)
    coverage_table()
