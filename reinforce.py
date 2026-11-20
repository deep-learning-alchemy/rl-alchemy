"""On-policy REINFORCE with the knobs studied in Assignment 7.

    uv run python -m reinforce         # one default Humanoid run, locally

A run is fully described by a PGConfig. train(cfg) writes
runs/<run_name>/{config.json, metrics.jsonl, pi_final.pt, ...}. Each
metrics.jsonl row is one policy update; rows at evaluation times also carry
eval_* fields.

One update: run `batch_episodes` complete episodes side by side with the
current policy (one batched forward pass per time step), compute discounted
returns-to-go G_t, subtract a baseline, and descend

    L(theta) = -(1/N) sum_t w_t log pi_theta(a_t | s_t) A_t  -  c_H * entropy

where N is the number of steps in the batch, A_t = G_t - b_t, and w_t = 1 for
on-policy data. w_t is a (detached, optionally clipped) importance weight
pi/mu when a batch is reused for several epochs or collected by a stale policy mu.

Discrete action spaces use a categorical policy; continuous ones (MuJoCo) use
a diagonal Gaussian with a learned, state-independent log standard deviation.

Clocks: `env_step` counts environment interactions; `update` counts optimizer
steps. Learning-rate and entropy schedules are functions of the fraction of
the env-step budget used.
"""
import copy
import dataclasses
import hashlib
import json
import math
import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Independent, Normal, kl_divergence
from torch.nn import functional as F

import diagnostics as D
from envs import make_env
from parallel_envs import ParallelEnvs
from trajectories import Episode, batch_tensors
from utils import config_str, runs_dir, seed_everything


@dataclass(frozen=True)
class PGConfig:
    run_name: str | None = None
    tags: tuple = ()
    seed: int = 0
    init_seed: int | None = None         # network initialization (default: seed)
    data_seed: int | None = None         # environment and action sampling (default: seed)

    # Environment and reward.
    env_id: str = "Humanoid-v5"
    env_kwargs: dict = field(default_factory=dict)
    reward_scale: float = 0.1            # multiplies every training reward
    train_reward: str = "proxy"          # "proxy" (the environment's reward) or "true" (info["true_reward"])

    # Data.
    total_env_steps: int = 5_000_000
    batch_episodes: int = 32
    gamma: float = 0.99

    # Gradient estimator.
    baseline: str = "value"              # "none", "mean" (batch mean of G), or "value" (learned V(s))
    normalize_advantages: bool = True
    entropy_coef: float = 0.0
    entropy_coef_end: float | None = None  # linear anneal to this value ...
    entropy_anneal_frac: float = 1.0       # ... over this fraction of the env-step budget
    surrogate: str = "ppo"               # "reinforce": w.detach() * log pi * A; "ppo": clipped ratio objective
    ppo_clip: float = 0.2
    gae_lambda: float | None = 0.95      # with baseline="value": GAE(lambda) advantages, bootstrapping
                                         # truncated episodes; None uses Monte Carlo returns-to-go
    epochs_per_batch: int = 10           # passes over each collected batch
    minibatches: int = 32                # optimizer steps per pass (random split of the batch's steps)
    policy_lag: int = 0                  # act with the policy from this many updates ago (stale actors)
    importance_weights: bool = True      # correct reused or stale data with pi/mu
    is_clip: float | None = None         # clip importance weights at this value

    # Optimizer.
    optimizer: str = "adam"              # "adam" (AdamW) or "sgd"
    lr: float = 3e-5
    weight_decay: float = 0.0            # decoupled (AdamW-style)
    betas: tuple = (0.9, 0.999)
    momentum: float = 0.0                # sgd only
    lr_schedule: str = "constant"        # "constant", "linear", "cosine", "wsd"
    warmup_frac: float = 0.0
    lr_end_frac: float = 0.0             # final lr / peak lr for decaying schedules
    decay_frac: float = 0.2              # wsd: decay linearly over the last fraction of the budget
    grad_clip: float | None = 0.5
    ema_decay: float | None = None       # also evaluate an EMA of the weights
    target_kl: float | None = None       # adapt the learning rate so KL(pi_old || pi_new) per batch
                                         # stays near this value (x lr_adapt_factor when far off)
    lr_adapt_factor: float = 1.5

    # Networks.
    arch: str = "resmlp"                 # "mlp" or "resmlp" (pre-norm residual MLP blocks)
    hidden: tuple = (64,)                # layer widths; for resmlp, (width,) * depth
    activation: str = "gelu"             # "tanh", "relu" or "gelu"
    obs_norm: bool = True                # running mean/std observation normalization (frozen with init_from)
    init_log_std: float = -0.5           # continuous actions only
    res_init: str = "default"            # resmlp: "zero" initializes each block's output layer to zero,
                                         # so every depth starts as the same function
    value_hidden: tuple = (256, 256)     # baseline="value" only (MLP)
    value_lr: float = 1e-3
    value_steps: int = 20
    device: str = "cpu"                  # "cuda" helps only for large networks
    env_workers: int = 8                 # >1: step the batch's environments in this many processes

    # Initialization and regularization toward a reference policy.
    init_from: str | None = None         # "<run>" or "<run>/<file>.pt"
    kl_ref: str | None = None            # reference policy for a KL(pi || pi_ref) penalty (exact KL over
    kl_coef: float = 0.0                 # actions at the batch's states, added to the loss)

    # Logging, evaluation and artifacts.
    eval_every: int = 250_000            # env steps
    eval_episodes: int = 8
    extra_evals: tuple = ()              # ((name, env_kwargs), ...): also evaluate under these settings,
                                         # logged as eval_<name>_return, eval_<name>_true_return, ...
    noise_scale_every: int = 0           # updates; 0 disables
    probe_size: int = 512
    save_checkpoints: tuple = ()         # env steps
    save_final: bool = True
    wandb: bool = False

    def name(self):
        if self.run_name:
            return self.run_name
        blob = json.dumps(dataclasses.asdict(self), sort_keys=True, default=str)
        return f"{self.env_id.split('-')[0].lower()}-{hashlib.sha1(blob.encode()).hexdigest()[:8]}"


# ----------------------------------------------------------------------------
# Networks
# ----------------------------------------------------------------------------

ACTIVATIONS = dict(tanh=nn.Tanh, relu=nn.ReLU, gelu=nn.GELU)


class RunningNorm(nn.Module):
    """Observation normalization with running statistics (saved with the model)."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x):
        batch_mean, batch_var, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, total = batch_mean - self.mean, self.count + n
        self.mean += delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta ** 2 * self.count * n / total) / total
        self.count = total

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-10, 10)


class ResBlock(nn.Module):
    """x + W2 act(W1 LayerNorm(x)): the transformer's pre-norm MLP sublayer."""

    def __init__(self, width, activation, expansion=4, zero_init=False):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, expansion * width)
        self.act = ACTIVATIONS[activation]()
        self.down = nn.Linear(expansion * width, width)
        if zero_init:
            nn.init.zeros_(self.down.weight)
            nn.init.zeros_(self.down.bias)

    def forward(self, x):
        return x + self.down(self.act(self.up(self.norm(x))))


def make_body(in_dim, hidden, activation, arch, zero_init=False):
    if arch == "mlp":
        layers, width = [], in_dim
        for h in hidden:
            layers += [nn.Linear(width, h), ACTIVATIONS[activation]()]
            width = h
        return nn.Sequential(*layers), width
    if arch == "resmlp":
        width = hidden[0]
        assert all(h == width for h in hidden), "resmlp uses hidden=(width,) * depth"
        blocks = [nn.Linear(in_dim, width)] + [ResBlock(width, activation, zero_init=zero_init) for _ in hidden]
        return nn.Sequential(*blocks, nn.LayerNorm(width)), width
    raise ValueError(f"unknown arch {arch!r}")


class Policy(nn.Module):
    def __init__(self, obs_dim, action_space, hidden=(64, 64), activation="tanh", arch="mlp",
                 obs_norm=True, init_log_std=-0.5, res_init="default"):
        super().__init__()
        self.continuous = isinstance(action_space, gym.spaces.Box)
        self.act_dim = action_space.shape[0] if self.continuous else action_space.n
        if self.continuous:  # sampled actions are clipped to these before stepping the environment
            self.action_low = np.asarray(action_space.low, np.float32)
            self.action_high = np.asarray(action_space.high, np.float32)
        self.norm = RunningNorm(obs_dim) if obs_norm else nn.Identity()
        self.body, width = make_body(obs_dim, tuple(hidden), activation, arch, zero_init=res_init == "zero")
        self.head = nn.Linear(width, self.act_dim)
        if self.continuous:
            self.log_std = nn.Parameter(torch.full((self.act_dim,), float(init_log_std)))
            nn.init.zeros_(self.head.bias)
            self.head.weight.data.mul_(0.01)

    def features(self, obs):
        return self.body(self.norm(obs))

    def forward(self, obs):
        """Logits (discrete) or action means (continuous)."""
        return self.head(self.features(obs))

    def dist(self, obs):
        out = self(obs)
        if self.continuous:
            return Independent(Normal(out, self.log_std.exp()), 1)
        return Categorical(logits=out)

    @torch.no_grad()
    def act(self, obs, generator, greedy=False):
        """Sample actions; returns (actions, log mu(a|s)) for the behavior policy."""
        out = self(obs)
        if self.continuous:
            std = self.log_std.exp()
            a = out if greedy else out + std * torch.randn(out.shape, generator=generator, device=out.device)
            return a, Normal(out, std).log_prob(a).sum(-1)
        probs = F.softmax(out, -1)
        a = probs.argmax(-1) if greedy else torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        return a, probs.gather(-1, a[:, None]).squeeze(-1).clamp_min(1e-12).log()


class Value(nn.Module):
    def __init__(self, obs_dim, hidden=(64, 64)):
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.body, width = make_body(obs_dim, tuple(hidden), "tanh", "mlp")
        self.head = nn.Linear(width, 1)

    def forward(self, obs):
        return self.head(self.body(self.norm(obs))).squeeze(-1)


def build_policy(cfg, env):
    return Policy(env.observation_space.shape[0], env.action_space, cfg.hidden, cfg.activation,
                  cfg.arch, cfg.obs_norm, cfg.init_log_std, cfg.res_init)


def _resolve(spec, default_file):
    """'<run>' or '<run>/<file>.pt' -> path under runs/."""
    path = runs_dir() / spec
    return path if spec.endswith(".pt") else path / default_file


def load_config(run):
    raw = json.loads((runs_dir() / run / "config.json").read_text())
    known = {f.name for f in dataclasses.fields(PGConfig)}
    return PGConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in known})


def load_policy(spec, file="pi_final.pt"):
    """Rebuild a saved policy on the CPU. spec is '<run>' or '<run>/<file>.pt'."""
    cfg = load_config(spec.split("/")[0])
    policy = build_policy(cfg, make_env(cfg.env_id, env_kwargs=cfg.env_kwargs))
    policy.load_state_dict(torch.load(_resolve(spec, file), weights_only=True, map_location="cpu"))
    return policy.eval(), cfg


# ----------------------------------------------------------------------------
# Rollouts and evaluation
# ----------------------------------------------------------------------------

def rollouts(policy, envs, generator, greedy=False):
    """One complete episode in each env, stepped in lockstep with batched actions."""
    device = next(policy.parameters()).device
    parallel = isinstance(envs, ParallelEnvs)
    obs = envs.reset_all() if parallel else [env.reset()[0] for env in envs]
    steps = [dict(o=[], a=[], r=[], tr=[], lp=[]) for _ in range(len(envs))]
    terminated = [False] * len(envs)
    active = list(range(len(envs)))
    while active:
        x = torch.as_tensor(np.stack([obs[i] for i in active]), dtype=torch.float32, device=device)
        actions, logp = policy.act(x, generator, greedy)
        actions, logp = actions.cpu().numpy(), logp.cpu().numpy()
        env_actions = [np.clip(actions[j], policy.action_low, policy.action_high) if policy.continuous
                       else int(actions[j])
                       for j in range(len(active))]
        pairs = list(zip(active, env_actions))
        results = envs.step(pairs) if parallel else [envs[i].step(a) for i, a in pairs]
        still = []
        for j, i in enumerate(active):
            a = actions[j]
            next_obs, reward, term, trunc, info = results[j]
            s = steps[i]
            s["o"].append(obs[i])
            s["a"].append(a)
            s["r"].append(reward)
            s["tr"].append(info["true_reward"])
            s["lp"].append(logp[j])
            obs[i] = next_obs
            s["final"] = next_obs
            if term or trunc:
                terminated[i] = bool(term)
            else:
                still.append(i)
        active = still
    action_dtype = np.float32 if policy.continuous else np.int64
    return [Episode(np.asarray(s["o"], np.float32), np.asarray(s["a"], action_dtype),
                    np.asarray(s["r"], np.float32), np.asarray(s["tr"], np.float32),
                    np.asarray(s["lp"], np.float32), terminated[i],
                    final_obs=np.asarray(s["final"], np.float32))
            for i, s in enumerate(steps)]


def evaluate(policy, envs, generator, prefix="eval"):
    """Stochastic (the policy REINFORCE optimizes) and greedy rollouts."""
    out = {}
    for mode, greedy in (("", False), ("_greedy", True)):
        eps = rollouts(policy, envs, generator, greedy=greedy)
        out[f"{prefix}{mode}_return"] = float(np.mean([e.rewards.sum() for e in eps]))
        out[f"{prefix}{mode}_true_return"] = float(np.mean([e.true_rewards.sum() for e in eps]))
        if not greedy:
            out[f"{prefix}_return_std"] = float(np.std([e.rewards.sum() for e in eps]))
            out[f"{prefix}_length"] = float(np.mean([len(e) for e in eps]))
            out[f"{prefix}_terminated_frac"] = float(np.mean([e.terminated for e in eps]))
    return out


def lr_factor(cfg, frac):
    if cfg.warmup_frac > 0 and frac < cfg.warmup_frac:
        return frac / cfg.warmup_frac
    if cfg.lr_schedule == "constant":
        return 1.0
    progress = min(max((frac - cfg.warmup_frac) / max(1 - cfg.warmup_frac, 1e-12), 0.0), 1.0)
    end = cfg.lr_end_frac
    if cfg.lr_schedule == "linear":
        return 1 - (1 - end) * progress
    if cfg.lr_schedule == "cosine":
        return end + (1 - end) * 0.5 * (1 + math.cos(math.pi * progress))
    if cfg.lr_schedule == "wsd":
        decay = min(max((frac - (1 - cfg.decay_frac)) / max(cfg.decay_frac, 1e-12), 0.0), 1.0)
        return 1 - (1 - end) * decay
    raise ValueError(f"unknown lr_schedule {cfg.lr_schedule!r}")


def entropy_coef(cfg, frac):
    if cfg.entropy_coef_end is None:
        return cfg.entropy_coef
    progress = min(frac / max(cfg.entropy_anneal_frac, 1e-12), 1.0)
    return cfg.entropy_coef + progress * (cfg.entropy_coef_end - cfg.entropy_coef)


class Logger:
    def __init__(self, cfg, run_dir):
        self.path = run_dir / "metrics.jsonl"
        self.path.write_text("")
        self.wandb = None
        if cfg.wandb:
            import wandb
            self.wandb = wandb.init(
                entity=config_str("CONFIG_WANDB_ENTITY", "WANDB_ENTITY"),
                project=config_str("CONFIG_WANDB_PROJECT", "WANDB_PROJECT", default="rl-alchemy"),
                name=cfg.name(), tags=list(cfg.tags), config=dataclasses.asdict(cfg))

    def log(self, row):
        row = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in row.items()}
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if self.wandb:
            self.wandb.log(row, step=row["update"])

    def close(self):
        if self.wandb:
            self.wandb.finish()


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: PGConfig, verbose=True):
        torch.set_num_threads(1)
        self.cfg, self.verbose = cfg, verbose
        self.device = torch.device(cfg.device)
        init_seed = cfg.seed if cfg.init_seed is None else cfg.init_seed
        data_seed = cfg.seed if cfg.data_seed is None else cfg.data_seed
        self.run_dir = runs_dir() / cfg.name()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2, default=str))

        seed_everything(data_seed)
        seeds = [1000 * data_seed + i for i in range(cfg.batch_episodes)]
        eval_seeds = [1000 * data_seed + 500 + i for i in range(cfg.eval_episodes)]
        def env_group(group_seeds, kwargs):
            if cfg.env_workers > 1:
                return ParallelEnvs(cfg.env_id, group_seeds, kwargs, cfg.env_workers)
            return [make_env(cfg.env_id, s, kwargs) for s in group_seeds]
        self.envs = env_group(seeds, cfg.env_kwargs)
        self.eval_envs = env_group(eval_seeds, cfg.env_kwargs)
        self.extra_eval_envs = {name: env_group(eval_seeds, {**cfg.env_kwargs, **dict(kwargs)})
                                for name, kwargs in cfg.extra_evals}
        env = make_env(cfg.env_id, seeds[0], cfg.env_kwargs)  # for spaces
        self.gen = torch.Generator(self.device).manual_seed(data_seed)
        self.eval_gen = torch.Generator(self.device).manual_seed(data_seed + 10_000)
        self.rng = np.random.default_rng(data_seed)
        self.minibatch_gen = torch.Generator().manual_seed(data_seed + 20_000)
        self.lr_scale = 1.0  # adapted when target_kl is set

        torch.manual_seed(init_seed)
        self.policy = build_policy(cfg, env)
        if cfg.init_from:
            self.policy.load_state_dict(torch.load(_resolve(cfg.init_from, "pi_final.pt"),
                                                   weights_only=True, map_location="cpu"))
        self.policy.to(self.device)
        self.value = None
        if cfg.baseline == "value":
            self.value = Value(env.observation_space.shape[0], cfg.value_hidden).to(self.device)
            self.value_opt = torch.optim.Adam(self.value.parameters(), lr=cfg.value_lr)
        self.ref = None
        if cfg.kl_ref:
            self.ref = load_policy(cfg.kl_ref)[0].requires_grad_(False).to(self.device)
        self.ema = copy.deepcopy(self.policy).requires_grad_(False) if cfg.ema_decay else None
        self.actor = copy.deepcopy(self.policy).requires_grad_(False) if cfg.policy_lag else None
        self.history = []  # recent policy states, for policy_lag

        params = self.policy.parameters()
        if cfg.optimizer == "adam":
            self.opt = torch.optim.AdamW(params, lr=cfg.lr, betas=tuple(cfg.betas),
                                         weight_decay=cfg.weight_decay, foreach=True)
        elif cfg.optimizer == "sgd":
            self.opt = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                                       weight_decay=cfg.weight_decay)
        else:
            raise ValueError(f"unknown optimizer {cfg.optimizer!r}")

        self.logger = Logger(cfg, self.run_dir)
        self.updates, self.env_step = 0, 0
        self.probe = self.probe_actions = None
        self.last_params = [p.detach().clone() for p in self.policy.parameters()]
        self.start_time = time.time()

    # -- one update -------------------------------------------------------
    def frac(self):
        return self.env_step / self.cfg.total_env_steps

    def batch(self, episodes):
        cfg = self.cfg
        if cfg.train_reward == "true":
            episodes = [dataclasses.replace(e, rewards=e.true_rewards) for e in episodes]
        batch = batch_tensors(episodes, cfg.gamma, cfg.reward_scale)
        return {k: v.to(self.device) for k, v in batch.items()}

    def advantages(self, batch):
        cfg, rtg = self.cfg, batch["rtg"]
        stats = {}
        if cfg.baseline == "none":
            b = torch.zeros_like(rtg)
        elif cfg.baseline == "mean":
            b = rtg.mean().expand_as(rtg)
        elif cfg.baseline == "value":
            self.value.norm.update(batch["obs"])
            with torch.no_grad():
                b = self.value(batch["obs"])
            target = rtg
            if cfg.gae_lambda is not None:
                gae = self.gae(batch, b)
                target = gae + b  # lambda-returns
                rtg = b + gae     # so that rtg - b below is the GAE advantage
            stats["value_explained_var"] = D.explained_variance(b, target)
            for _ in range(cfg.value_steps):
                value_loss = F.mse_loss(self.value(batch["obs"]), target)
                self.value_opt.zero_grad(set_to_none=True)
                value_loss.backward()
                self.value_opt.step()
                stats["value_loss"] = value_loss.item()
        else:
            raise ValueError(f"unknown baseline {cfg.baseline!r}")
        adv = rtg - b
        stats["adv_mean"], stats["adv_std"] = adv.mean().item(), adv.std().item()
        if cfg.normalize_advantages:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, stats

    @torch.no_grad()
    def gae(self, batch, values):
        """Generalized advantage estimation (Schulman et al., 2016) within each
        episode; truncated episodes bootstrap from V(final observation)."""
        cfg = self.cfg
        final_v = self.value(batch["final_obs"]) * (~batch["terminated"]).float()
        rewards, ids = batch["rewards"].cpu().numpy(), batch["episode_id"].cpu().numpy()
        v, fv = values.cpu().numpy(), final_v.cpu().numpy()
        adv = np.zeros_like(rewards)
        bounds = np.flatnonzero(np.diff(ids)) + 1
        for lo, hi in zip(np.concatenate([[0], bounds]), np.concatenate([bounds, [len(ids)]])):
            next_v, running = fv[ids[lo]], 0.0
            for t in range(hi - 1, lo - 1, -1):
                delta = rewards[t] + cfg.gamma * next_v - v[t]
                running = delta + cfg.gamma * cfg.gae_lambda * running
                adv[t], next_v = running, v[t]
        return torch.as_tensor(adv, device=values.device)

    @torch.no_grad()
    def mismatch(self, batch, when):
        """How far the current policy is from the behavior policy on the batch's
        own samples: KL(mu || pi) (k3 estimator) and the raw ratio pi/mu, whether
        or not the update uses importance weights."""
        log_r = self.policy.dist(batch["obs"]).log_prob(batch["actions"]) - batch["behavior_logp"]
        r = log_r.exp()
        return {f"behavior_kl_{when}": (r - 1 - log_r).mean().item(),
                f"ratio_max_{when}": r.max().item(), f"ratio_outside_clip_{when}": (
                    (r - 1).abs() > self.cfg.ppo_clip).float().mean().item()}

    def weights(self, logp, batch):
        if not self.cfg.importance_weights:
            return torch.ones_like(logp)
        w = (logp.detach() - batch["behavior_logp"]).exp()
        return w.clamp(max=self.cfg.is_clip) if self.cfg.is_clip else w

    def loss_terms(self, batch, adv, ent_coef):
        cfg = self.cfg
        dist = self.policy.dist(batch["obs"])
        logp = dist.log_prob(batch["actions"])
        w = self.weights(logp, batch)
        if cfg.surrogate == "reinforce":
            per_step = -(w * logp * adv)
        elif cfg.surrogate == "ppo":
            ratio = (logp - batch["behavior_logp"]).exp()
            clipped = ratio.clamp(1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * adv
            per_step = -torch.min(ratio * adv, clipped)
        else:
            raise ValueError(f"unknown surrogate {cfg.surrogate!r}")
        surrogate = per_step.mean()
        ent = dist.entropy().mean()
        loss = surrogate - ent_coef * ent
        kl_ref = None
        if self.ref is not None:
            kl_ref = kl_divergence(dist, self.ref.dist(batch["obs"])).mean()
            loss = loss + cfg.kl_coef * kl_ref
        return loss, surrogate, ent, kl_ref, dist, logp, w

    def noise_scale(self, batch, adv):
        """Per-episode gradients of the surrogate -> gradient noise scale."""
        params = list(self.policy.parameters())
        grads = []
        for i in range(int(batch["episode_id"].max()) + 1):
            mask = batch["episode_id"] == i
            logp = self.policy.dist(batch["obs"][mask]).log_prob(batch["actions"][mask])
            g = torch.autograd.grad(-(logp * adv[mask]).sum(), params, allow_unused=True)
            grads.append(torch.cat([(torch.zeros_like(p) if x is None else x).flatten()
                                    for x, p in zip(g, params)]))
        b, tr_sigma, g_sq = D.gradient_noise_scale(torch.stack(grads))
        return dict(noise_scale=b, grad_var_trace=tr_sigma, grad_sq_norm=g_sq)

    def update(self, episodes):
        cfg = self.cfg
        frac = self.frac()
        lr = cfg.lr * lr_factor(cfg, frac) * self.lr_scale
        for group in self.opt.param_groups:
            group["lr"] = lr
        ent_coef = entropy_coef(cfg, frac)
        batch = self.batch(episodes)
        if cfg.obs_norm and not cfg.init_from:
            # A run started from a saved policy keeps that policy's normalization,
            # so it starts out as exactly that policy.
            self.policy.norm.update(batch["obs"])
        adv, row = self.advantages(batch)
        row.update(lr=lr, entropy_coef=ent_coef, batch_steps=len(batch["actions"]))

        # Quantities that look like a loss. None of them is the objective.
        with torch.no_grad():
            dist0 = self.policy.dist(batch["obs"])
            logp0 = dist0.log_prob(batch["actions"])
            row["surrogate_loss_no_baseline"] = -(logp0 * batch["rtg"]).mean().item()
            row["neg_logp"] = -logp0.mean().item()
        if cfg.noise_scale_every and self.updates % cfg.noise_scale_every == 0 and len(episodes) > 1:
            row.update(self.noise_scale(batch, adv))

        row.update(self.mismatch(batch, "start"))
        n = len(batch["actions"])
        steps = [(epoch, idx) for epoch in range(cfg.epochs_per_batch)
                 for idx in torch.randperm(n, generator=self.minibatch_gen).chunk(cfg.minibatches)]
        for k, (epoch, idx) in enumerate(steps):
            mb = batch if cfg.minibatches == 1 else {
                key: (v[idx] if key not in ("final_obs", "terminated") else v) for key, v in batch.items()}
            loss, surrogate, ent, kl_ref, _, _, w = self.loss_terms(mb, adv if cfg.minibatches == 1 else adv[idx], ent_coef)
            self.opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                raise Diverged(f"non-finite loss at update {self.updates}, env step {self.env_step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), cfg.grad_clip if cfg.grad_clip else float("inf"))
            if not torch.isfinite(grad_norm):
                raise Diverged(f"non-finite gradient at update {self.updates}, env step {self.env_step}")
            if k == 0:
                row.update(surrogate_loss=surrogate.item(), loss=loss.item(), entropy=ent.item(),
                           grad_norm=grad_norm.item(), is_weight_mean=w.mean().item(),
                           is_weight_max=w.max().item())
                if kl_ref is not None:
                    row["kl_ref"] = kl_ref.item()
            self.opt.step()
            self.updates += 1
            if self.ema is not None:
                with torch.no_grad():
                    for p, e in zip(self.policy.parameters(), self.ema.parameters()):
                        e.lerp_(p, 1 - cfg.ema_decay)
                    for b, e in zip(self.policy.buffers(), self.ema.buffers()):
                        e.copy_(b)

        # What the surrogate predicts about this step, on the same batch.
        with torch.no_grad():
            dist1 = self.policy.dist(batch["obs"])
            logp1 = dist1.log_prob(batch["actions"])
            ratio = (logp1 - logp0).exp()
            row["predicted_improvement"] = ((ratio - 1) * (batch["rtg"] - batch["rtg"].mean())).mean().item()
            row["surrogate_loss_after"] = -(self.weights(logp1, batch) * logp1 * adv).mean().item()
            row["kl_update"] = kl_divergence(dist0, dist1).mean().item()
        row.update(self.mismatch(batch, "end"))
        if cfg.target_kl:
            if row["kl_update"] > 2 * cfg.target_kl:
                self.lr_scale /= cfg.lr_adapt_factor
            elif row["kl_update"] < cfg.target_kl / 2:
                self.lr_scale *= cfg.lr_adapt_factor
            self.lr_scale = min(max(self.lr_scale, 1e-3), 1e2)
            row["lr_scale"] = self.lr_scale
        with torch.no_grad():
            row["clip_frac"] = ((ratio - 1).abs() > 0.2).float().mean().item()
        row["epochs"] = cfg.epochs_per_batch
        row["optimizer_steps"] = len(steps)
        return row

    # -- logging ----------------------------------------------------------
    def log(self, row, episodes, do_eval):
        cfg = self.cfg
        row = dict(update=self.updates, env_step=self.env_step,
                   wall_time=time.time() - self.start_time, **row)
        if episodes:
            row.update(train_return=float(np.mean([e.rewards.sum() for e in episodes])),
                       train_true_return=float(np.mean([e.true_rewards.sum() for e in episodes])),
                       train_length=float(np.mean([len(e) for e in episodes])),
                       train_terminated_frac=float(np.mean([e.terminated for e in episodes])))
        params = list(self.policy.parameters())
        row["param_norm"] = D.flat_norm(params)
        row["update_norm"] = D.flat_norm([p - q for p, q in zip(params, self.last_params)])
        self.last_params = [p.detach().clone() for p in params]
        with torch.no_grad():
            out = self.policy(self.probe)
            row["probe_entropy"] = self.policy.dist(self.probe).entropy().mean().item()
            row["srank"] = D.srank(self.policy.features(self.probe).float().cpu())
            if self.policy.continuous:
                row["log_std"] = self.policy.log_std.mean().item()
                actions = out  # mean action
                if self.probe_actions is not None:
                    row["policy_churn"] = (actions - self.probe_actions).abs().mean().item()
            else:
                row["probe_logit_scale"] = out.std(-1).mean().item()
                actions = out.argmax(-1)
                if self.probe_actions is not None:
                    row["policy_churn"] = (actions != self.probe_actions).float().mean().item()
            self.probe_actions = actions
        if do_eval:
            row.update(evaluate(self.policy, self.eval_envs, self.eval_gen))
            for name, envs in self.extra_eval_envs.items():
                row.update(evaluate(self.policy, envs, self.eval_gen, prefix=f"eval_{name}"))
            if self.ema is not None:
                row.update(evaluate(self.ema, self.eval_envs, self.eval_gen, "ema"))
            if self.verbose:
                print(f"[{cfg.name()}] env_step={self.env_step} update={self.updates} "
                      f"entropy={row['entropy']:.3f} eval_return={row['eval_return']:.1f} "
                      f"true={row['eval_true_return']:.2f} t={row['wall_time']:.0f}s", flush=True)
        self.logger.log(row)

    def checkpoint(self, name):
        torch.save(self.policy.state_dict(), self.run_dir / name)
        if self.ema is not None:
            torch.save(self.ema.state_dict(), self.run_dir / name.replace("pi_", "ema_"))

    def set_probe(self, episodes):
        obs = np.concatenate([e.obs for e in episodes])
        idx = self.rng.choice(len(obs), min(self.cfg.probe_size, len(obs)), replace=False)
        self.probe = torch.as_tensor(obs[idx], device=self.device)

    # -- loops ------------------------------------------------------------
    def run_online(self):
        cfg = self.cfg
        next_eval, checkpoints = 0, sorted(cfg.save_checkpoints)
        while self.env_step < cfg.total_env_steps:
            actor = self.policy
            if cfg.policy_lag:
                self.history = (self.history + [copy.deepcopy(self.policy.state_dict())])[-(cfg.policy_lag + 1):]
                self.actor.load_state_dict(self.history[0])
                actor = self.actor
            episodes = rollouts(actor, self.envs, self.gen)
            if self.probe is None:
                self.set_probe(episodes)
            do_eval = self.env_step >= next_eval
            row = self.update(episodes)
            self.env_step += sum(len(e) for e in episodes)
            if do_eval:
                next_eval += cfg.eval_every
            while checkpoints and self.env_step >= checkpoints[0]:
                self.checkpoint(f"pi_{checkpoints.pop(0)}.pt")
            self.log(row, episodes, do_eval or self.env_step >= cfg.total_env_steps)

    def run(self):
        diverged = None
        try:
            self.run_online()
        except (Diverged, ValueError) as error:  # a finished run that blew up, not a crash
            if isinstance(error, ValueError) and "invalid values" not in str(error):
                raise  # NaN parameters surface as invalid distribution arguments
            diverged = str(error).splitlines()[0][:200]
            self.logger.log(dict(env_step=self.env_step, update=self.updates, diverged=diverged))
            if self.verbose:
                print(f"[{self.cfg.name()}] diverged: {diverged}", flush=True)
        finally:
            self.logger.close()
            for envs in (self.envs, self.eval_envs, *self.extra_eval_envs.values()):
                if isinstance(envs, ParallelEnvs):
                    envs.close()
        if self.cfg.save_final:
            self.checkpoint("pi_final.pt")
        (self.run_dir / "DONE").write_text(json.dumps(dict(diverged=diverged)) if diverged else "")
        return self.run_dir


class Diverged(RuntimeError):
    """The loss became non-finite; the run stops and is marked diverged in metrics.jsonl and DONE."""


def train(cfg: PGConfig, verbose=True):
    return Trainer(cfg, verbose).run()


def run(cfg, verbose=True):
    """Entry point for the launchers."""
    return train(cfg, verbose)


if __name__ == "__main__":
    train(PGConfig(run_name="humanoid-default"))
