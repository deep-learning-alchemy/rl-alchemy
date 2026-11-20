# CS312 Assignment 7: Exotic — Policy Gradients

This repository is the starter code for the reinforcement-learning assignment.
It is separate from the shared language-modeling repository. Everything here
trains policies with policy gradients (REINFORCE and PPO-style updates) in
PyTorch, on two environments: Gymnasium's MuJoCo
[Humanoid](https://gymnasium.farama.org/environments/mujoco/humanoid/)
(Problems 1–3) and CoinLoop, a small gridworld in `envs.py` (Problem 4).

The handout is distributed separately. This README covers setup,
the code, and the logged metrics.

## 1. Setup

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone <this repo> rl-alchemy
cd rl-alchemy
uv python install 3.10
uv sync
uv run pytest -q              # ~10 s of CPU checks
```

Open [utils.py](utils.py) and set your Modal environment (and, optionally, W&B):

```python
CONFIG_MODAL_ENVIRONMENT = "YOUR_MODAL_ENVIRONMENT"
CONFIG_WANDB_ENTITY = "YOUR_WANDB_USERNAME_OR_TEAM"
CONFIG_WANDB_PROJECT = "rl-alchemy"
```

## 2. Compute

Humanoid physics runs on the CPU. A default run (5M environment steps, the
batch's 32 episodes stepped by 8 worker processes) takes about an hour on an
8-CPU Modal container. Small policies train on CPU; policies of width 128 or
more should set `device="cuda"` and run on an H100, where a run of 5M steps
takes 20–90 minutes depending on size. Quiz questions assume this Modal
hardware.

You can run short debugging runs locally (`total_env_steps=200_000`), but run
real experiments on Modal.

## 3. Running experiments

A run is one `PGConfig` ([reinforce.py](reinforce.py)). Experiments build lists
of configs and hand them to `launch`:

```python
from dataclasses import replace
from experiments.common import DEFAULT
from launch import launch

configs = [replace(DEFAULT, run_name=f"lr{lr:g}-s{s}", lr=lr, seed=s)
           for lr in (1e-5, 3e-5, 1e-4) for s in range(3)]
launch(configs, modal=True)                # waits, copying results into runs/
launch(configs, modal=True, detach=True)   # returns at once; runs keep going on Modal
launch(configs, dry_run=True)              # print names only
```

Every run saves its final policy as `pi_final.pt`; `save_checkpoints=(500_000, 2_000_000)`
also saves `pi_500000.pt` and `pi_2000000.pt`. Detached runs write to the Modal Volume `volume-rl-alchemy`; copy them into
`runs/` with `modal_launch.fetch_modal(names)`. Runs that read other runs
(`init_from`, `kl_ref`)
start after them automatically when not detached. A run whose
`runs/<name>/DONE` exists is skipped unless `overwrite=True`. Runs are
deterministic given their seeds; the launcher notes configs that differ only
in their names.

### Watching the robot

```bash
uv run python -m render RUN                          # runs/RUN/pi_final.pt -> renders/RUN.gif
uv run python -m render RUN --file pi_500000.pt      # a checkpoint from save_checkpoints
uv run python -m render RUN --fetch                  # copy the policy from the Modal Volume first
uv run python -m render RUN --env '{"terminate_when_unhealthy": true}'   # override env_kwargs
```

Rendering happens on your own machine, not on Modal, and takes a few seconds
per GIF. On a laptop (macOS, Windows, or Linux with a desktop) it works as is.
On a Linux machine with no display, such as a server you reach over SSH,
MuJoCo cannot open a window to draw into; run with `MUJOCO_GL=egl` (uses the
GPU) or `MUJOCO_GL=osmesa` (CPU only, slower), e.g.
`MUJOCO_GL=egl uv run python -m render RUN`.

## 4. Code

| File | Contents |
|---|---|
| [reinforce.py](reinforce.py) | `PGConfig`, policies, batched rollouts, the update (REINFORCE and PPO-style), evaluation |
| [envs.py](envs.py) | `make_env(env_id, seed, env_kwargs)`, which builds Humanoid or CoinLoop and logs each step's true reward next to the training (proxy) reward; the CoinLoop gridworld (Problem 4); and `coinloop_solve`, which computes the exactly optimal proxy-reward policy for CoinLoop by value iteration |
| [parallel_envs.py](parallel_envs.py) | Stepping a batch's environments in worker processes |
| [trajectories.py](trajectories.py) | Episodes and returns-to-go |
| [diagnostics.py](diagnostics.py) | Entropy, KL, effective rank, gradient noise scale |
| [analysis.py](analysis.py) | `load_runs`, plotting, weight averaging and interpolation |
| [render.py](render.py) | GIFs of saved policies |
| [launch.py](launch.py), [modal_launch.py](modal_launch.py) | Launchers |
| [experiments/](experiments) | One starter per problem; shared configs and the fair-scaling helper in `common.py` |

### The update

Each update runs `batch_episodes` complete episodes side by side with the
current stochastic policy (a Gaussian over the 17 joint torques with a learned
state-independent standard deviation), computes advantages, and optimizes

```
L(θ) = -(1/N) Σ_t surrogate_t  −  c_H · entropy  [+ c_KL · KL(π_θ ‖ π_ref)]
```

| Option | Default | Meaning |
|---|---|---|
| `baseline` | `"value"` | learned value network (`value_hidden`, an MLP), batch mean, or none |
| `gae_lambda` | `0.95` | GAE advantages, bootstrapping truncated episodes; `None` uses Monte Carlo returns-to-go |
| `normalize_advantages` | `True` | standardize advantages within each batch |
| `surrogate` | `"ppo"` | `"reinforce"`: `sg(π/μ) · log π · A`; `"ppo"`: clipped ratio (`ppo_clip=0.2`) |
| `epochs_per_batch`, `minibatches` | `10`, `32` | passes over each batch and optimizer steps per pass |
| `importance_weights`, `is_clip` | `True`, `None` | correct reused or stale data with π/μ |
| `policy_lag` | `0` | act with the policy from this many updates ago |
| `grad_clip` | `0.5` | gradient-norm clipping |
| `kl_ref`, `kl_coef` | — | penalty `kl_coef` · KL(π_θ ‖ π_ref) to a saved policy, exact over actions at the batch's states |
| `entropy_coef` (+ `_end`, `anneal_frac`) | `0` | entropy bonus and its linear anneal |
| `init_log_std` | `-0.5` | initial log standard deviation of the action noise |

Plain REINFORCE is `surrogate="reinforce", gae_lambda=None, epochs_per_batch=1,
minibatches=1, grad_clip=None`. It barely learns Humanoid.

### Optimization and model options

| Option | Default | Meaning |
|---|---|---|
| `lr`, `optimizer` | `3e-5`, AdamW | learning rate; `weight_decay`, `lr_schedule` (`constant`, `linear`, `cosine`, `wsd`), `warmup_frac` |
| `target_kl` | `None` | adapt the learning rate after every batch to keep KL(π_old ‖ π_new) near this value |
| `arch`, `hidden`, `activation` | `"resmlp"`, `(64,)`, `"gelu"` | policy network; `resmlp` stacks pre-norm residual blocks `x + W2 act(W1 LayerNorm(x))` (4× expansion), `hidden=(width,) * depth` |
| `res_init` | `"default"` | `"zero"` zero-initializes each block's output layer, so every depth starts as the same function |
| `value_hidden` | `(256, 256)` | value network (MLP) |
| `obs_norm`, `reward_scale` | `True`, `0.1` | running observation normalization; rewards scaled before computing returns |
| `ema_decay` | `None` | also evaluate an EMA of the weights |
| `device`, `env_workers` | `"cpu"`, `8` | where the networks run; processes stepping the environments |

`experiments.common.fair(cfg, width, depth, scale_value)` applies the tuning
needed to compare model sizes fairly: zero-initialized residual blocks, the
adaptive learning rate (`target_kl=0.02`) starting from `2e-5 · (64/width)^0.5`,
and optionally a value network as wide as the policy.

### Rewards and evaluation

The proxy reward is Humanoid's own reward: `forward_reward_weight` × forward
velocity (default 1.25) + `healthy_reward` per step upright (default 5) − small
control and contact costs. The true reward, logged as `eval_true_return`, is
the forward velocity alone; its sum over an episode is the forward progress.
Set reward weights, the start-state noise `reset_noise_scale` (default 0.01)
and whether a fall ends the episode (`terminate_when_unhealthy`, default true)
through `env_kwargs`. `train_reward` picks what the agent trains on: `"proxy"`
(the environment's reward) or `"true"` (forward velocity). `extra_evals=(("name", env_kwargs), ...)` also evaluates
under other settings, e.g. held-out start-state noise.

## 5. Logged metrics

Each line of `runs/<name>/metrics.jsonl` is one batch.

| Key | Meaning |
|---|---|
| `surrogate_loss`, `loss` | the policy-gradient term alone, and what the optimizer minimizes: `surrogate_loss − entropy_coef · entropy + kl_coef · kl_ref` (equal when both coefficients are 0); first minibatch of the batch |
| `surrogate_loss_no_baseline`, `neg_logp`, `surrogate_loss_after` | other loss-like quantities; none of these, nor the two above, is the objective or tracks performance |
| `predicted_improvement` | what the surrogate says the update gained, on its own batch |
| `entropy`, `log_std`, `kl_update`, `clip_frac` | policy entropy and mean log standard deviation; KL(π_old ‖ π_new) caused by the update; fraction of steps whose ratio moved more than 20% |
| `lr`, `lr_scale` | current learning rate and, with `target_kl`, its adaptive multiplier |
| `grad_norm`, `param_norm`, `update_norm` | gradient norm before clipping; parameter norm; change since the previous batch |
| `noise_scale` | gradient noise scale in episodes, every `noise_scale_every` batches |
| `adv_mean`, `adv_std`, `value_loss`, `value_explained_var` | advantages and value network |
| `is_weight_mean`, `is_weight_max`, `kl_ref` | importance weights used by the update (1 without correction); KL to the reference policy |
| `behavior_kl_start`/`_end`, `ratio_max_*`, `ratio_outside_clip_*` | distance between the current and behavior policies on the batch, before and after the batch's updates, whether or not the update corrects for it |
| `train_return`, `train_true_return`, `train_length` | the batch's episodes |
| `probe_entropy`, `policy_churn`, `srank` | on a fixed set of 512 states: entropy, change in the mean action since the previous batch, effective rank of the last hidden layer |
| `eval_return`, `eval_true_return`, `eval_length` | stochastic policy: proxy return, forward progress, episode length |
| `eval_greedy_*` | mean action, no noise |
| `eval_<name>_*` | the same under each `extra_evals` setting |

## 6. Tests

```bash
uv run pytest -q
```

They check returns-to-go, GAE, that the REINFORCE estimator matches the exact
policy gradient on a bandit, the stop-gradient identity,
schedules and the adaptive learning rate, diagnostics, parallel environment
stepping, the launcher, weight averaging, and short runs of every training mode.
