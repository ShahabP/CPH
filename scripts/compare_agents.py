"""
Comparison framework for DQN vs Neural RIR agents.

Runs both approaches on the same synthetic data and compares:
- Learning curves (rewards over episodes)
- RIR estimation quality (correlation, DRR)
- Computational efficiency
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add src to path
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from neural_rir_agent import NeuralRIRAgent, NeuralRIREnvironment, compute_rir_drr_metric
from rl_framework import RIREstimationEnv, DQNAgent
from audio_processing import AudioProcessor


def create_test_data(sample_rate: int = 16000, duration: float = 2.0, 
                     num_scenarios: int = 5):
    """Create multiple test scenarios with different RIRs and speech."""
    scenarios = []
    
    for i in range(num_scenarios):
        # Varied clean signals
        t = np.linspace(0, duration, int(duration * sample_rate))
        if i == 0:
            # Sine wave with decay
            clean = 0.3 * np.sin(2 * np.pi * 440 * t) * np.exp(-t * 0.5)
        elif i == 1:
            # Chirp signal
            clean = 0.2 * np.sin(2 * np.pi * (200 + 300 * t) * t)
        elif i == 2:
            # Speech-like formants
            f1, f2 = 800, 1200
            clean = 0.2 * (np.sin(2 * np.pi * f1 * t) + 0.7 * np.sin(2 * np.pi * f2 * t))
            clean *= (1 + 0.3 * np.sin(2 * np.pi * 150 * t))  # F0 modulation
        elif i == 3:
            # White noise burst
            clean = 0.15 * np.random.randn(len(t)) * np.exp(-t * 2)
        else:
            # Mixed harmonics
            clean = 0.2 * sum(np.sin(2 * np.pi * (220 * (h+1)) * t) / (h+1) 
                             for h in range(5))
        
        # Varied RIRs
        rir_length = 512
        true_rir = np.zeros(rir_length)
        true_rir[0] = 1.0
        
        if i < 3:
            # Exponential decay with early reflections
            rt60 = 0.3 + i * 0.2
            for j in range(1, rir_length):
                decay = np.exp(-6.91 * j / (rt60 * sample_rate))
                true_rir[j] = decay * np.random.randn() * 0.1
            # Add strong early reflections
            for delay in [50, 120, 200]:
                if delay < rir_length:
                    true_rir[delay] += 0.3 * np.exp(-delay / 100)
        else:
            # More complex RIR with multiple reflections
            for j in range(1, min(300, rir_length)):
                if j in [30, 60, 100, 150, 200]:
                    # Strong reflections
                    true_rir[j] = 0.4 * np.exp(-j / 80) * (1 + 0.1 * np.random.randn())
                else:
                    # Background reverberation
                    true_rir[j] = 0.05 * np.exp(-j / 120) * np.random.randn()
        
        # Create reverberant audio
        reverb = np.convolve(clean, true_rir, mode='same')
        
        scenarios.append({
            'clean': clean,
            'reverb': reverb,
            'true_rir': true_rir,
            'name': f'scenario_{i}'
        })
    
    return scenarios


def run_dqn_agent(scenario: dict, episodes: int = 10, max_steps: int = 20):
    """Run DQN agent on scenario."""
    print(f"Running DQN on {scenario['name']}...")
    
    # Create environment and agent
    env = RIREstimationEnv(max_iterations=max_steps, rir_length=len(scenario['true_rir']))
    state_dim = env.observation_space.shape[0]
    agent = DQNAgent(state_dim=state_dim, action_dim=16, learning_rate=1e-3)
    
    # Discrete to continuous action mapping
    rng = np.random.RandomState(42)
    codebook = rng.uniform(-1, 1, size=(16, 6)).astype(np.float32)
    
    episode_rewards = []
    episode_drrs = []
    episode_correlations = []
    
    start_time = time.time()
    
    for episode in range(episodes):
        state, info = env.reset(options={'reverb_audio': scenario['reverb']})
        total_reward = 0
        
        for step in range(max_steps):
            action_idx = agent.act(state, training=True)
            action = codebook[action_idx % len(codebook)]
            
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.remember(state, action, reward, next_state, terminated or truncated)
            
            state = next_state
            total_reward += reward
            
            if terminated or truncated:
                break
        
        agent.replay()
        
        # Evaluate final RIR
        if hasattr(env, 'current_rir_estimate') and env.current_rir_estimate is not None:
            final_rir = env.current_rir_estimate
            drr = compute_rir_drr_metric(final_rir)
            min_len = min(len(final_rir), len(scenario['true_rir']))
            correlation = np.corrcoef(final_rir[:min_len], scenario['true_rir'][:min_len])[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            drr, correlation = 0.0, 0.0
        
        episode_rewards.append(total_reward)
        episode_drrs.append(drr)
        episode_correlations.append(correlation)
    
    elapsed_time = time.time() - start_time
    
    return {
        'rewards': episode_rewards,
        'drrs': episode_drrs,
        'correlations': episode_correlations,
        'time': elapsed_time,
        'final_rir': final_rir if 'final_rir' in locals() else None
    }


def run_neural_agent(scenario: dict, episodes: int = 10, max_steps: int = 20):
    """Run Neural RIR agent on scenario."""
    print(f"Running Neural Agent on {scenario['name']}...")
    
    # Create environment and agent
    env = NeuralRIREnvironment(max_iterations=max_steps, rir_length=len(scenario['true_rir']))
    agent = NeuralRIRAgent(rir_length=len(scenario['true_rir']), learning_rate=1e-3)
    
    episode_rewards = []
    episode_drrs = []
    episode_correlations = []
    
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset(scenario['reverb'], scenario['clean'])
        total_reward = 0
        
        for step in range(max_steps):
            next_state, reward, terminated, info = env.step(agent)
            total_reward += reward
            
            if terminated:
                break
        
        # Learn from episode
        agent.end_episode()
        
        # Evaluate final RIR
        final_rir = env.current_rir
        drr = compute_rir_drr_metric(final_rir)
        min_len = min(len(final_rir), len(scenario['true_rir']))
        correlation = np.corrcoef(final_rir[:min_len], scenario['true_rir'][:min_len])[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        episode_rewards.append(total_reward)
        episode_drrs.append(drr)
        episode_correlations.append(correlation)
    
    elapsed_time = time.time() - start_time
    
    return {
        'rewards': episode_rewards,
        'drrs': episode_drrs,
        'correlations': episode_correlations,
        'time': elapsed_time,
        'final_rir': final_rir
    }


def plot_comparison(dqn_results: dict, neural_results: dict, scenario_name: str, 
                   output_dir: Path):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    episodes = range(len(dqn_results['rewards']))
    
    # Rewards
    axes[0, 0].plot(episodes, dqn_results['rewards'], 'b-', label='DQN', alpha=0.7)
    axes[0, 0].plot(episodes, neural_results['rewards'], 'r-', label='Neural Policy', alpha=0.7)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Learning Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # DRR
    axes[0, 1].plot(episodes, dqn_results['drrs'], 'b-', label='DQN', alpha=0.7)
    axes[0, 1].plot(episodes, neural_results['drrs'], 'r-', label='Neural Policy', alpha=0.7)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('DRR (dB)')
    axes[0, 1].set_title('Direct-to-Reverberant Ratio')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Correlations
    axes[1, 0].plot(episodes, dqn_results['correlations'], 'b-', label='DQN', alpha=0.7)
    axes[1, 0].plot(episodes, neural_results['correlations'], 'r-', label='Neural Policy', alpha=0.7)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('RIR Correlation')
    axes[1, 0].set_title('RIR Estimation Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final RIRs comparison
    if dqn_results['final_rir'] is not None and neural_results['final_rir'] is not None:
        x = np.arange(min(len(dqn_results['final_rir']), len(neural_results['final_rir'])))
        axes[1, 1].plot(x, dqn_results['final_rir'][:len(x)], 'b-', label='DQN Final RIR', alpha=0.7)
        axes[1, 1].plot(x, neural_results['final_rir'][:len(x)], 'r-', label='Neural Final RIR', alpha=0.7)
        axes[1, 1].set_xlabel('Tap Index')
        axes[1, 1].set_ylabel('Amplitude')
        axes[1, 1].set_title('Final RIR Estimates')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'DQN vs Neural Policy Comparison - {scenario_name}')
    plt.tight_layout()
    
    output_path = output_dir / f'comparison_{scenario_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot: {output_path}")


def main():
    """Run full comparison."""
    print("=== DQN vs Neural RIR Agent Comparison ===\n")
    
    # Create test scenarios
    scenarios = create_test_data(num_scenarios=3)  # Start with 3 for speed
    
    # Output directory
    output_dir = Path('experiments/comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    all_results = {'dqn': [], 'neural': []}
    
    # Run comparison on each scenario
    for scenario in scenarios:
        print(f"\n=== {scenario['name']} ===")
        print(f"True RIR structural metric: {compute_rir_drr_metric(scenario['true_rir']):.2f} dB")
        
        # Run both agents
        dqn_results = run_dqn_agent(scenario, episodes=5, max_steps=15)
        neural_results = run_neural_agent(scenario, episodes=5, max_steps=15)
        
        # Store results
        all_results['dqn'].append(dqn_results)
        all_results['neural'].append(neural_results)
        
        # Print summary
        print(f"\nResults for {scenario['name']}:")
        print(f"DQN     - Final reward: {dqn_results['rewards'][-1]:.3f}, "
              f"DRR: {dqn_results['drrs'][-1]:.2f} dB, "
              f"Corr: {dqn_results['correlations'][-1]:.3f}, "
              f"Time: {dqn_results['time']:.2f}s")
        print(f"Neural  - Final reward: {neural_results['rewards'][-1]:.3f}, "
              f"DRR: {neural_results['drrs'][-1]:.2f} dB, "
              f"Corr: {neural_results['correlations'][-1]:.3f}, "
              f"Time: {neural_results['time']:.2f}s")
        
        # Create comparison plot
        plot_comparison(dqn_results, neural_results, scenario['name'], output_dir)
    
    # Overall summary
    print(f"\n=== Overall Comparison ===")
    
    dqn_avg_reward = np.mean([r['rewards'][-1] for r in all_results['dqn']])
    neural_avg_reward = np.mean([r['rewards'][-1] for r in all_results['neural']])
    dqn_avg_drr = np.mean([r['drrs'][-1] for r in all_results['dqn']])
    neural_avg_drr = np.mean([r['drrs'][-1] for r in all_results['neural']])
    dqn_avg_corr = np.mean([r['correlations'][-1] for r in all_results['dqn']])
    neural_avg_corr = np.mean([r['correlations'][-1] for r in all_results['neural']])
    dqn_avg_time = np.mean([r['time'] for r in all_results['dqn']])
    neural_avg_time = np.mean([r['time'] for r in all_results['neural']])
    
    print(f"Average Final Reward    - DQN: {dqn_avg_reward:.3f}, Neural: {neural_avg_reward:.3f}")
    print(f"Average Final DRR       - DQN: {dqn_avg_drr:.2f} dB, Neural: {neural_avg_drr:.2f} dB")
    print(f"Average Final Correlation - DQN: {dqn_avg_corr:.3f}, Neural: {neural_avg_corr:.3f}")
    print(f"Average Training Time   - DQN: {dqn_avg_time:.2f}s, Neural: {neural_avg_time:.2f}s")
    
    # Winner analysis
    better_reward = "Neural" if neural_avg_reward > dqn_avg_reward else "DQN"
    better_drr = "Neural" if neural_avg_drr > dqn_avg_drr else "DQN"
    better_corr = "Neural" if neural_avg_corr > dqn_avg_corr else "DQN"
    faster = "Neural" if neural_avg_time < dqn_avg_time else "DQN"
    
    print(f"\nWinner by metric:")
    print(f"  Reward: {better_reward}")
    print(f"  DRR: {better_drr}")
    print(f"  Correlation: {better_corr}")
    print(f"  Speed: {faster}")


if __name__ == "__main__":
    main()