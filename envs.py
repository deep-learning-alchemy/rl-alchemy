"""The Humanoid environment, with its proxy and true rewards.

make_env() returns Gymnasium's Humanoid-v5 (17 joint torques, 348-dimensional
observations, episodes end when the torso falls or after 1,000 steps). The
agent trains on the environment's own reward, the proxy reward:

    forward_reward_weight * forward velocity      (default 1.25)
    + healthy_reward for every step it stays up   (default 5)
    - ctrl_cost_weight * |action|^2 - contact cost

All of these weights, and the start-state noise (reset_noise_scale, default
0.01), are set through env_kwargs. The true reward, reported in
info["true_reward"] and never trained on by default, is the forward velocity
alone; its sum over an episode is the forward progress.

make_env("CoinLoop-v0") returns a small gridworld with a respawning coin and
an exact solver (Problem 4; see CoinLoop below). make_env() also accepts other
Gymnasium environment ids (the tests use CartPole for speed); their true
reward is their own reward.
"""
import gymnasium as gym
import numpy as np


class TrueReward(gym.Wrapper):
    """Adds info['true_reward']: forward velocity for Humanoid, else the reward."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info.setdefault("true_reward", float(info["x_velocity"]) if "x_velocity" in info else float(reward))
        return obs, reward, terminated, truncated, info


def make_env(env_id="Humanoid-v5", seed=0, env_kwargs=None):
    """env_kwargs go to gym.make (or CoinLoop)."""
    kwargs = dict(env_kwargs or {})
    base = CoinLoop(**kwargs) if env_id == "CoinLoop-v0" else gym.make(env_id, **kwargs)
    env = TrueReward(base)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


MOVES = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])  # up, right, down, left


class CoinLoop(gym.Env):
    """A size x size grid. Start top-left, goal bottom-right.

    True reward: goal_reward on reaching the goal (episode terminates), minus
    step_cost per step. Proxy reward additionally pays coin_reward each time
    the agent *enters* the coin cell while the coin is charged; the coin then
    recharges for `recharge` steps. Episodes are truncated at max_steps.

    Observation: one-hot position, coin-timer fraction, and (if time_obs) the
    fraction of the episode elapsed.
    """

    metadata = {"render_modes": []}

    def __init__(self, size=5, coin=(0, 2), goal_reward=1.0, coin_reward=0.2,
                 recharge=4, step_cost=0.0, max_steps=50, time_obs=False):
        self.size, self.coin = size, tuple(coin)
        self.start, self.goal = (0, 0), (size - 1, size - 1)
        self.goal_reward, self.coin_reward = goal_reward, coin_reward
        self.recharge, self.step_cost = recharge, step_cost
        self.max_steps, self.time_obs = max_steps, time_obs
        dim = size * size + 1 + int(time_obs)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (dim,), np.float32)
        self.action_space = gym.spaces.Discrete(4)

    def _obs(self):
        obs = np.zeros(self.observation_space.shape, np.float32)
        obs[self.pos[0] * self.size + self.pos[1]] = 1.0
        obs[self.size * self.size] = self.timer / self.recharge
        if self.time_obs:
            obs[-1] = self.t / self.max_steps
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos, self.timer, self.t = self.start, 0, 0
        return self._obs(), {}

    def transition(self, pos, timer, action):
        """Deterministic dynamics, shared with coinloop_solve()."""
        r, c = np.clip(np.array(pos) + MOVES[action], 0, self.size - 1)
        new = (int(r), int(c))
        timer = max(timer - 1, 0)
        coin = new == self.coin and new != pos and timer == 0
        if coin:
            timer = self.recharge
        done = new == self.goal
        true = (self.goal_reward if done else 0.0) - self.step_cost
        proxy = true + (self.coin_reward if coin else 0.0)
        return new, timer, proxy, true, done, coin

    def step(self, action):
        self.pos, self.timer, proxy, true, done, coin = self.transition(
            self.pos, self.timer, int(action))
        self.t += 1
        truncated = self.t >= self.max_steps and not done
        return self._obs(), proxy, done, truncated, dict(true_reward=true, coin=coin)


def coinloop_solve(gamma, horizon_aware=False, **env_kwargs):
    """Exact optimal policy for the proxy reward, by value iteration.

    horizon_aware=False solves the infinite-horizon discounted problem, as if
    episodes never ended at the time limit. horizon_aware=True solves the
    finite-horizon problem with time-dependent policies (what an agent with
    time_obs=True could represent). A REINFORCE agent without time_obs
    optimizes a third objective: the finite-horizon return of a stationary
    policy. Returns the greedy policy's proxy and true returns over one episode.
    """
    env = CoinLoop(**env_kwargs)
    cells = [(r, c) for r in range(env.size) for c in range(env.size)]
    timers = range(env.recharge + 1)
    T = env.max_steps

    def backup(V):
        Q = {}
        for pos in cells:
            for timer in timers:
                for a in range(4):
                    new, nt, proxy, _, done, _ = env.transition(pos, timer, a)
                    Q[pos, timer, a] = proxy + (0.0 if done else gamma * V[new, nt])
        return Q

    def greedy(Q):
        return {(p, k): max(range(4), key=lambda a: Q[p, k, a]) for p in cells for k in timers}

    V = {(p, k): 0.0 for p in cells for k in timers}
    if horizon_aware:
        policies = []
        for _ in range(T):  # policies[i] is used with T - i steps remaining
            Q = backup(V)
            V = {(p, k): max(Q[p, k, a] for a in range(4)) for p in cells for k in timers}
            policies.append(greedy(Q))
        policies = policies[::-1]
    else:
        for _ in range(10_000):
            Q = backup(V)
            new_V = {(p, k): max(Q[p, k, a] for a in range(4)) for p in cells for k in timers}
            delta = max(abs(new_V[s] - V[s]) for s in V)
            V = new_V
            if delta < 1e-10:
                break
        policies = [greedy(Q)] * T

    pos, timer = env.start, 0
    proxy_return = true_return = 0.0
    coins, reached = 0, False
    for t in range(T):
        pos, timer, proxy, true, done, coin = env.transition(pos, timer, policies[t][pos, timer])
        proxy_return += proxy
        true_return += true
        coins += coin
        if done:
            reached = True
            break
    return dict(proxy_return=proxy_return, true_return=true_return, coins=coins,
                reaches_goal=reached, steps=t + 1, start_value=V[env.start, 0])
