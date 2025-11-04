"""
Minimal RL training entrypoint for RIR estimation.

Uses DQNAgent if PyTorch is installed; otherwise falls back to a RandomAgent
so you can validate the environment quickly.
"""

import os
import sys
from typing import Dict, List

import numpy as np

# Make src importable
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rl_framework import RIREstimationEnv, DQNAgent  # type: ignore


class RandomAgent:
    """Simple random policy for quick environment validation."""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state, training=True):  # noqa: D401
        return self.action_space.sample()

    def remember(self, *args, **kwargs):
        pass

    def replay(self, *args, **kwargs):
        pass

    def update_target_network(self, *args, **kwargs):
        pass


def _dqn_action_codebook() -> np.ndarray:
    """Small codebook mapping discrete actions to 6D continuous actions."""
    np.random.seed(42)
    codebook = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [-1.0, -0.5, 0.0, 0.5, 1.0, 0.0],
        [1.0, -1.0, 1.0, -1.0, 0.0, 0.0],
        [-0.25, 0.25, -0.25, 0.25, 0.0, 0.0],
        [0.75, -0.75, 0.25, -0.25, 0.5, -0.5],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ], dtype=np.float32)
    return codebook


def train(episodes: int = 10, max_steps: int = 50) -> List[Dict]:
    env = RIREstimationEnv(max_iterations=max_steps, action_space_type='continuous')

    # Choose agent based on torch availability
    try:
        import torch  # noqa: F401

        use_torch = True
    except Exception:
        use_torch = False

    if use_torch:
        state_dim = env.observation_space.shape[0]
        # For continuous Box, discretize to a small set for DQN demo
        action_dim = 8

        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
        codebook = _dqn_action_codebook()
    else:
        agent = RandomAgent(env.action_space)

    stats: List[Dict] = []
    for ep in range(episodes):
        state, info = env.reset()
        total_reward = 0.0
        for step in range(max_steps):
            action_raw = agent.act(state, training=True)
            if isinstance(agent, DQNAgent):
                # Map discrete action to continuous vector via codebook
                action = codebook[int(action_raw)]
            else:
                action = action_raw
            next_state, reward, terminated, truncated, info = env.step(action)
            if isinstance(agent, DQNAgent):
                agent.remember(state, action, reward, next_state, terminated or truncated)
            state = next_state
            total_reward += reward
            if terminated or truncated:
                break
        if isinstance(agent, DQNAgent):
            agent.replay()
            if ep % 5 == 0:
                agent.update_target_network()
        ep_stats = {
            'episode': ep,
            'total_reward': float(total_reward),
            'steps': step + 1,
            'final_metrics': info.get('metrics', {}),
        }
        stats.append(ep_stats)
        print(f"Episode {ep}: reward={total_reward:.3f}, steps={step+1}, metrics={ep_stats['final_metrics']}")
    return stats


if __name__ == "__main__":
    train(episodes=5, max_steps=20)
