import os
import sys
import numpy as np

# Add src to path
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rl_framework import StreamingRIREstimationEnv, DQNAgent  # type: ignore


def synth_segment(sr=16000, dur=1.0, f0=220.0, rir_len=512):
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)
    clean = 0.2 * (np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * (2*f0) * t))
    rir = np.zeros(rir_len)
    rir[0] = 1.0
    if rir_len > 120:
        rir[120] = 0.25
    if rir_len > 240:
        rir[240] = 0.12
    reverb = np.convolve(clean, rir, mode='same')
    return reverb, clean, rir


def main():
    env = StreamingRIREstimationEnv(max_iterations=10)

    # Create agent (discretized actions)
    try:
        import torch  # noqa: F401
        use_torch = True
    except Exception:
        use_torch = False

    if use_torch:
        state_dim = env.observation_space.shape[0]
        action_dim = 8
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
        # Build small discrete-to-continuous action codebook (6D vectors)
        rng = np.random.RandomState(0)
        codebook = rng.uniform(-1.0, 1.0, size=(action_dim, 6)).astype(np.float32)
        codebook[0] = 0.0  # anchor at zeros
    else:
        class RandomAgent:
            def __init__(self, action_space):
                self.action_space = action_space
            def act(self, state, training=True):
                return self.action_space.sample()
            def remember(self, *args, **kwargs):
                pass
            def replay(self, *args, **kwargs):
                pass
            def update_target_network(self, *args, **kwargs):
                pass
        agent = RandomAgent(env.action_space)

    # Push 3 incoming segments with slightly different pitches
    for f0 in [200.0, 240.0, 280.0]:
        seg = synth_segment(f0=f0)
        env.push_segment(*seg)

    # Process each segment iteratively, retaining the global RIR estimate
    all_stats = []
    seg_idx = 0
    while True:
        try:
            state, info = env.start_next_segment()
        except RuntimeError:
            break  # no more segments
        total_reward = 0.0
        for step in range(env.max_iterations):
            a = agent.act(state, training=True)
            # Map discrete DQN action to continuous vector if needed
            if use_torch and isinstance(a, (int, np.integer)):
                action = codebook[int(a)]
            else:
                action = a
            next_state, reward, terminated, truncated, info = env.step(action)
            if hasattr(agent, 'remember'):
                agent.remember(state, action, reward, next_state, terminated or truncated)
            state = next_state
            total_reward += reward
            if terminated or truncated:
                break
        if hasattr(agent, 'replay'):
            agent.replay()
        print(f"Segment {seg_idx}: reward={total_reward:.3f}, metrics={info.get('metrics', {})}")
        seg_idx += 1
        all_stats.append({'segment': seg_idx, 'reward': total_reward, 'metrics': info.get('metrics', {})})

    # Show final global RIR estimate length
    if env.global_rir_estimate is not None:
        print("Final global RIR estimate length:", len(env.global_rir_estimate))


if __name__ == "__main__":
    main()
