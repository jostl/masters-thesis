import numpy as np


def rtgs(rewards, terminals, normalize=False):
    # Calculates rewards-to-go
    # Code based on https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/2_rtg_pg.py
    n = len(rewards)
    rewards_to_go = np.zeros_like(rewards)
    for i in reversed(range(n)):
        # Logic: 'i' is an index for a state.
        # Only add rewards-to-go if 'i + 1' exist and 'i' is not a terminal state
        rewards_to_go[i] = rewards[i] + (rewards_to_go[i + 1] if i + 1 < n and not terminals[i] else 0)
    if normalize:
        rewards_to_go = (rewards_to_go - np.mean(rewards_to_go)) / (np.std(rewards_to_go) + 1e-7)
    return rewards_to_go


def gae(rewards, terminals, values, gamma, lmbda, normalize=False):
    # Calculates generalized advantage estimation (GAE).
    # Code based on https://nn.labml.ai/rl/ppo/gae.html
    n_advantages = len(rewards)
    advantages = np.zeros(n_advantages)
    last_advantage = 0
    last_value = values[-1]
    for t in reversed(range(n_advantages)):
        mask = 1 - terminals[t]
        last_value = last_value * mask
        last_advantage = last_advantage * mask
        delta = rewards[t] + gamma * last_value - values[t]
        last_advantage = delta + gamma * lmbda * last_advantage
        advantages[t] = last_advantage
        last_value = values[t]
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
    return advantages
