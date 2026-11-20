"""Episodes and discounted returns-to-go."""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Episode:
    obs: np.ndarray          # (T, obs_dim)
    actions: np.ndarray      # (T,) discrete or (T, act_dim) continuous
    rewards: np.ndarray      # (T,) proxy reward the agent trains on
    true_rewards: np.ndarray # (T,)
    behavior_logp: np.ndarray  # (T,) log mu(a_t | s_t) of the policy that acted
    terminated: bool
    final_obs: np.ndarray | None = None  # observation after the last step (for bootstrapping)

    def __len__(self):
        return len(self.actions)


def returns_to_go(rewards, gamma):
    """G_t = sum_{k>=t} gamma^(k-t) r_k within one episode (no bootstrapping)."""
    out = np.zeros(len(rewards), np.float32)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def batch_tensors(episodes, gamma, reward_scale=1.0):
    """Concatenate episodes into flat tensors for one policy-gradient update.
    Training rewards are reward_scale * r."""
    obs = torch.as_tensor(np.concatenate([e.obs for e in episodes]))
    actions = torch.as_tensor(np.concatenate([e.actions for e in episodes]))
    behavior_logp = torch.as_tensor(np.concatenate([e.behavior_logp for e in episodes]))
    rtg = torch.as_tensor(np.concatenate([returns_to_go(reward_scale * e.rewards, gamma)
                                          for e in episodes]))
    episode_id = torch.as_tensor(np.concatenate([np.full(len(e), i) for i, e in enumerate(episodes)]))
    rewards = torch.as_tensor(np.concatenate([reward_scale * e.rewards for e in episodes]))
    final_obs = torch.as_tensor(np.stack([e.obs[-1] if e.final_obs is None else e.final_obs for e in episodes]))
    terminated = torch.as_tensor([e.terminated for e in episodes])
    return dict(obs=obs, actions=actions, behavior_logp=behavior_logp, rtg=rtg, episode_id=episode_id,
                rewards=rewards.float(), final_obs=final_obs, terminated=terminated)
