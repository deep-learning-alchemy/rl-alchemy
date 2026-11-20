"""Fast CPU checks: environments, returns, the gradient estimator, schedules,
diagnostics, launching, and short runs of every training mode.

    uv run pytest -q
"""
import dataclasses
import json
import math

import numpy as np
import pytest
import torch

import analysis
import diagnostics as D
from envs import make_env
from launch import stages
from reinforce import PGConfig, Trainer, entropy_coef, load_policy, lr_factor, train
from trajectories import Episode, batch_tensors, returns_to_go


@pytest.fixture(autouse=True)
def tmp_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_ALCHEMY_RUNS_DIR", str(tmp_path))
    return tmp_path


# Fast smoke-test settings: CartPole with a small plain MLP and one REINFORCE step per batch.
TINY = dict(env_id="CartPole-v1", total_env_steps=1_500, batch_episodes=4, eval_every=500,
            eval_episodes=2, noise_scale_every=2, probe_size=32, hidden=(16, 16), value_hidden=(16,),
            arch="mlp", activation="tanh", surrogate="reinforce", gae_lambda=None, epochs_per_batch=1,
            minibatches=1, lr=3e-3, grad_clip=None, value_steps=5, reward_scale=1.0, env_workers=1)


def test_returns_to_go():
    np.testing.assert_allclose(returns_to_go(np.array([1.0, 1.0, 1.0]), 0.5), [1.75, 1.5, 1.0])


def test_surrogate_gradient_matches_exact_policy_gradient():
    """On a one-step bandit, the batch-averaged REINFORCE gradient converges to
    the exact gradient of expected reward, for any constant baseline."""
    torch.manual_seed(0)
    logits = torch.zeros(3, requires_grad=True)
    rewards = torch.tensor([1.0, 0.0, 2.0])
    exact = torch.autograd.grad((torch.softmax(logits, 0) * rewards).sum(), logits)[0]
    for b in (0.0, 5.0):
        probs = torch.softmax(logits, 0).detach()
        a = torch.multinomial(probs, 200_000, replacement=True)
        logp = torch.log_softmax(logits, 0)[a]
        surrogate = -(logp * (rewards[a] - b)).mean()
        estimate = -torch.autograd.grad(surrogate, logits)[0]
        torch.testing.assert_close(estimate, exact, atol=0.02 + 0.01 * b, rtol=0)


def test_humanoid_true_reward_is_forward_velocity():
    import numpy as np
    env = make_env("Humanoid-v5", seed=0, env_kwargs=dict(forward_reward_weight=12.5, healthy_reward=0.0))
    _, r, _, _, info = env.step(np.zeros(17, np.float32))
    assert info["true_reward"] == pytest.approx(info["x_velocity"])
    assert info["reward_forward"] == pytest.approx(12.5 * info["x_velocity"])


def test_schedules():
    assert lr_factor(PGConfig(), 0.7) == 1.0
    assert lr_factor(PGConfig(lr_schedule="linear"), 0.5) == pytest.approx(0.5)
    assert lr_factor(PGConfig(lr_schedule="cosine"), 1.0) == pytest.approx(0.0)
    assert lr_factor(PGConfig(lr_schedule="cosine", warmup_frac=0.1), 0.05) == pytest.approx(0.5)
    wsd = PGConfig(lr_schedule="wsd", decay_frac=0.2)
    assert lr_factor(wsd, 0.7) == 1.0 and lr_factor(wsd, 0.9) == pytest.approx(0.5)
    cfg = PGConfig(entropy_coef=0.1, entropy_coef_end=0.0, entropy_anneal_frac=0.5)
    assert entropy_coef(cfg, 0.25) == pytest.approx(0.05) and entropy_coef(cfg, 0.9) == 0.0


def test_diagnostics_on_known_inputs():
    assert D.srank(torch.eye(8)) == 8 and D.srank(torch.ones(8, 8)) == 1
    logits = torch.zeros(5, 2)
    assert D.entropy(logits)[0].item() == pytest.approx(math.log(2))
    assert D.kl(logits, logits).abs().max().item() == pytest.approx(0.0)
    # Identical per-example gradients: no noise. Pure noise: infinite/undefined scale.
    same = torch.ones(10, 3)
    assert D.gradient_noise_scale(same)[0] == pytest.approx(0.0)
    g = torch.randn(100_000, 4) + torch.tensor([1.0, 0, 0, 0])
    assert D.gradient_noise_scale(g)[0] == pytest.approx(4.0, rel=0.05)


def test_batch_tensors():
    ep = Episode(np.zeros((3, 2), np.float32), np.array([0, 1, 0]), np.ones(3, np.float32),
                 np.ones(3, np.float32), np.full(3, -0.7, np.float32), True)
    batch = batch_tensors([ep, ep], 1.0, reward_scale=2.0)
    np.testing.assert_allclose(batch["rtg"][:3].numpy(), [6, 4, 2])
    assert batch["episode_id"].tolist() == [0, 0, 0, 1, 1, 1]


@pytest.mark.parametrize("overrides", [
    dict(),
    dict(baseline="value", normalize_advantages=True, entropy_coef=0.01, entropy_coef_end=0.0),
    dict(baseline="none", epochs_per_batch=3, is_clip=2.0, lr_schedule="cosine", warmup_frac=0.1,
         weight_decay=0.1, ema_decay=0.9, grad_clip=1.0),
    dict(optimizer="sgd", momentum=0.9, lr=0.01, activation="relu", lr_schedule="wsd"),
    dict(env_id="Humanoid-v5", total_env_steps=700, batch_episodes=2, eval_episodes=1,
         arch="resmlp", hidden=(16,), activation="gelu", surrogate="ppo", gae_lambda=0.95,
         epochs_per_batch=2, minibatches=2, env_kwargs=dict(reset_noise_scale=0.1)),
])
def test_short_training_runs_log_finite_metrics(overrides):
    cfg = PGConfig(run_name="t", save_checkpoints=(500,), **{**TINY, **overrides})
    run_dir = train(cfg, verbose=False)
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["env_step"] >= cfg.total_env_steps
    assert rows[-1]["update"] == len(rows) * cfg.epochs_per_batch * cfg.minibatches
    for key in ("surrogate_loss", "entropy", "kl_update", "predicted_improvement", "param_norm",
                "eval_return", "eval_true_return", "eval_greedy_return"):
        assert math.isfinite(rows[-1][key]), key
    assert any("noise_scale" in r for r in rows)
    for name in ("DONE", "pi_final.pt", "pi_500.pt"):
        assert (run_dir / name).exists(), name
    if cfg.ema_decay:
        assert "ema_return" in rows[-1] and (run_dir / "ema_final.pt").exists()
    assert load_policy("t")[0] is not None


def test_init_from_and_kl(tmp_runs):
    train(PGConfig(run_name="expert", **TINY), verbose=False)
    # Fine-tuning starts from the reference: KL is zero at the first update.
    ft = train(PGConfig(run_name="ft", init_from="expert", kl_ref="expert", kl_coef=0.1, **TINY),
               verbose=False)
    first = json.loads((ft / "metrics.jsonl").read_text().splitlines()[0])
    assert first["kl_ref"] == pytest.approx(0.0, abs=1e-7)


def test_rerun_clears_stale_done(tmp_runs):
    run_dir = train(PGConfig(run_name="r", **TINY), verbose=False)
    assert (run_dir / "DONE").exists()
    Trainer(PGConfig(run_name="r", **TINY), verbose=False)  # a rerun has started but not finished
    assert not (run_dir / "DONE").exists()


def test_stages_order_dependencies():
    a = PGConfig(run_name="a")
    b = PGConfig(run_name="b", kl_ref="a")
    c = PGConfig(run_name="c", init_from="b")
    d = PGConfig(run_name="d", init_from="a/pi_100.pt")
    names = [sorted(x.name() for x in s) for s in stages([c, d, b, a])]
    assert names == [["a"], ["b", "d"], ["c"]]


def test_interpolation_endpoints(tmp_runs):
    train(PGConfig(run_name="x", **TINY), verbose=False)
    train(PGConfig(run_name="y", seed=1, **TINY), verbose=False)
    mid = analysis.average_policies(["x", "y"])
    px, py = load_policy("x")[0], load_policy("y")[0]
    for k, v in mid.state_dict().items():
        torch.testing.assert_close(v, 0.5 * (px.state_dict()[k] + py.state_dict()[k]))
    rows = analysis.interpolate("x", "y", alphas=[0.0, 1.0], episodes=2)
    assert [r["alpha"] for r in rows] == [0.0, 1.0]


def test_ratio_and_detached_weight_give_identical_gradients():
    """grad[sg(pi/mu) log pi] == grad[pi/mu]: the stop-gradient identity."""
    logits = torch.randn(5, 3, requires_grad=True)
    actions = torch.tensor([0, 2, 1, 1, 0])
    mu_logp = torch.log_softmax(torch.randn(5, 3), -1).gather(1, actions[:, None]).squeeze(1)
    adv = torch.randn(5)
    logp = torch.log_softmax(logits, -1).gather(1, actions[:, None]).squeeze(1)
    ratio = (logp - mu_logp).exp()
    g1 = torch.autograd.grad((ratio.detach() * logp * adv).mean(), logits, retain_graph=True)[0]
    g2 = torch.autograd.grad((ratio * adv).mean(), logits)[0]
    torch.testing.assert_close(g1, g2)


@pytest.mark.parametrize("overrides", [
    dict(policy_lag=2, surrogate="ppo"),
    dict(policy_lag=1, importance_weights=False),
    dict(surrogate="ppo", epochs_per_batch=2),
])
def test_estimator_variants_train(overrides):
    run_dir = train(PGConfig(run_name="v", **{**TINY, **overrides}), verbose=False)
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert math.isfinite(rows[-1]["surrogate_loss"]) and math.isfinite(rows[-1]["eval_return"])


def test_gae_matches_returns_at_lambda_one_and_td_at_zero():
    """GAE(1) advantages equal G_t - V(s_t) (bootstrapping truncated episodes);
    GAE(0) equals the one-step TD error."""
    t = Trainer(PGConfig(**{**TINY, "run_name": "gae", "gae_lambda": 1.0}), verbose=False)
    from reinforce import rollouts
    batch = t.batch(rollouts(t.policy, t.envs, t.gen))
    with torch.no_grad():
        v = t.value(batch["obs"])
        fv = t.value(batch["final_obs"]) * (~batch["terminated"]).float()
    adv1 = t.gae(batch, v)
    t.cfg = dataclasses.replace(t.cfg, gae_lambda=0.0)
    adv0 = t.gae(batch, v)
    ids, r, g = batch["episode_id"], batch["rewards"], t.cfg.gamma
    for i in range(int(ids.max()) + 1):
        idx = torch.nonzero(ids == i).squeeze(1)
        nxt = torch.cat([v[idx][1:], fv[i:i + 1]])
        torch.testing.assert_close(adv0[idx], r[idx] + g * nxt - v[idx], rtol=1e-5, atol=1e-5)
        ret, out = fv[i].item(), []
        for k in reversed(idx.tolist()):
            ret = r[k].item() + g * ret
            out.append(ret)
        torch.testing.assert_close(adv1[idx], torch.tensor(out[::-1]) - v[idx], rtol=1e-4, atol=1e-4)


def test_ppo_with_gae_and_minibatches_trains():
    cfg = PGConfig(**{**TINY, "run_name": "ppo", "surrogate": "ppo", "gae_lambda": 0.95, "epochs_per_batch": 3,
                      "minibatches": 4, "grad_clip": 0.5})
    run_dir = train(cfg, verbose=False)
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["optimizer_steps"] == 12 and rows[-1]["update"] == 12 * len(rows)
    assert math.isfinite(rows[-1]["eval_return"])


def test_parallel_envs_give_identical_runs(tmp_runs):
    seq = train(PGConfig(run_name="seq", **TINY), verbose=False)
    par = train(PGConfig(**{**TINY, "run_name": "par", "env_workers": 2}), verbose=False)
    rows = [[json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines()] for d in (seq, par)]
    for a, b in zip(*rows):
        assert a["env_step"] == b["env_step"] and a["train_return"] == pytest.approx(b["train_return"])


def test_zero_init_residual_blocks_start_as_identity():
    from reinforce import ResBlock
    block = ResBlock(8, "gelu", zero_init=True)
    x = torch.randn(5, 8)
    torch.testing.assert_close(block(x), x)


def test_target_kl_lowers_a_too_high_learning_rate_and_extra_evals_log():
    cfg = PGConfig(**{**TINY, "run_name": "kl", "lr": 0.5, "target_kl": 1e-4,
                      "extra_evals": (("short", dict(max_episode_steps=50)),)})
    run_dir = train(cfg, verbose=False)
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["lr_scale"] < 1.0 and rows[-1]["lr"] < cfg.lr
    assert "eval_short_return" in rows[-1] and "eval_short_length" in rows[-1]
    assert rows[-1]["eval_short_length"] <= 50


def test_divergence_ends_run_cleanly(tmp_path, monkeypatch):
    """A run whose parameters become NaN stops, and is marked diverged rather than crashing."""
    import json
    import torch
    from reinforce import Trainer
    monkeypatch.setenv("RL_ALCHEMY_RUNS_DIR", str(tmp_path))
    t = Trainer(PGConfig(**{**TINY, "run_name": "div", "total_env_steps": 3000, "eval_every": 1000}), verbose=False)
    original, calls = t.update, [0]

    def poisoned(episodes):
        calls[0] += 1
        if calls[0] == 2:
            with torch.no_grad():
                for p in t.policy.parameters():
                    p.fill_(float("nan"))
        return original(episodes)

    t.update = poisoned
    run_dir = t.run()
    assert json.loads((run_dir / "DONE").read_text())["diverged"]
    assert "diverged" in json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[-1])


def test_coinloop_rewards_and_solver():
    from envs import coinloop_solve, make_env
    env = make_env("CoinLoop-v0", 0)
    env.reset(seed=0)
    _, proxy, _, _, info = env.step(1)  # right: towards the coin, not on it yet
    assert proxy == 0.0 and info["true_reward"] == 0.0
    _, proxy, _, _, info = env.step(1)  # onto the charged coin
    assert proxy == 0.2 and info["true_reward"] == 0.0
    assert coinloop_solve(0.8)["reaches_goal"]
    farm = coinloop_solve(0.99)
    assert not farm["reaches_goal"] and farm["coins"] > 1
    assert coinloop_solve(0.99, horizon_aware=True)["reaches_goal"]


def test_coinloop_trains(tmp_path, monkeypatch):
    from dataclasses import replace
    from experiments.common import COINLOOP
    monkeypatch.setenv("RL_ALCHEMY_RUNS_DIR", str(tmp_path))
    from reinforce import run
    run(replace(COINLOOP, run_name="coin", total_env_steps=2000, eval_every=1000), verbose=False)
    assert (tmp_path / "coin" / "DONE").exists()


def test_continuous_actions_clipped_to_action_space():
    """Humanoid's controls live in [-0.4, 0.4]; sampled actions must be clipped there, not to [-1, 1]."""
    import numpy as np
    import torch
    from envs import make_env
    from reinforce import build_policy, rollouts
    env = make_env("Humanoid-v5", 0)
    policy = build_policy(PGConfig(init_log_std=1.0), env)  # very noisy actions
    assert np.allclose(policy.action_low, -0.4) and np.allclose(policy.action_high, 0.4)
    seen = []
    original = env.step

    def step(action):
        seen.append(np.asarray(action))
        return original(action)

    env.step = step
    rollouts(policy, [env], torch.Generator().manual_seed(0))
    acts = np.concatenate(seen)
    assert acts.min() >= -0.4 - 1e-6 and acts.max() <= 0.4 + 1e-6
