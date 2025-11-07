#!/usr/bin/env python3
"""
RIR Initialization Method Comparison
====================================

This script compares random vs exponential decay RIR initialization methods
in the RL-based RIR estimation system.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rl_framework import RIREstimationEnv
from neural_rir_agent import NeuralRIRAgent

def compare_initialization_methods(episodes: int = 50, max_steps: int = 15):
    """Compare random vs exponential decay RIR initialization methods."""

    print("🎯 RIR Initialization Method Comparison")
    print("=" * 50)
    print()

    # Results storage
    results = {
        'random': {'rewards': [], 'correlations': [], 'final_rirs': []},
        'exponential_decay': {'rewards': [], 'correlations': [], 'final_rirs': []}
    }

    init_methods = ['random', 'exponential_decay']

    for init_method in init_methods:
        print(f"🔬 Testing {init_method.replace('_', ' ').title()} Initialization")
        print("-" * 40)

        # Create environment with specific initialization method
        env = RIREstimationEnv(
            max_iterations=max_steps,
            rir_length=1024,
            rir_init_method=init_method
        )

        # Create neural agent
        agent = NeuralRIRAgent(
            rir_length=1024,
            hidden_dim=768,
            learning_rate=0.001
        )

        method_rewards = []
        method_correlations = []
        method_final_rirs = []

        for episode in range(episodes):
            # Reset environment
            state, info = env.reset()
            episode_reward = 0
            done = False
            step = 0

            while not done and step < max_steps:
                # Agent selects action
                action = agent.act(state)

                # Environment step
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Store experience
                agent.remember(state, action, reward, next_state, done)

                # Update
                state = next_state
                episode_reward += reward
                step += 1

            # Train agent
            agent.replay()

            # Store results
            final_metrics = info.get('metrics', {})
            correlation = final_metrics.get('correlation', 0.0)
            final_rir = env.current_state.current_rir_estimate.copy()

            method_rewards.append(episode_reward)
            method_correlations.append(correlation)
            method_final_rirs.append(final_rir)

            if episode % 10 == 0:
                print(f"Episode {episode:2d}: Reward={episode_reward:6.2f}, "
                      f"Correlation={correlation:.3f}")

        # Store method results
        results[init_method]['rewards'] = method_rewards
        results[init_method]['correlations'] = method_correlations
        results[init_method]['final_rirs'] = method_final_rirs

        print(f"✓ Completed {init_method} testing")
        print(f"  Average Reward: {np.mean(method_rewards):.2f}")
        print(f"  Average Correlation: {np.mean(method_correlations):.3f}")
        print()

    # Analysis and visualization
    create_comparison_plots(results)

    # Statistical comparison
    print("📊 STATISTICAL COMPARISON:")
    print("-" * 30)

    random_corr = np.array(results['random']['correlations'])
    exp_corr = np.array(results['exponential_decay']['correlations'])

    print("Final Correlation with Ground Truth:")
    print(f"  Random: {np.mean(random_corr):.3f} ± {np.std(random_corr):.3f}")
    print(f"  Exp Decay: {np.mean(exp_corr):.3f} ± {np.std(exp_corr):.3f}")
    print(f"  Improvement: {(np.mean(exp_corr) - np.mean(random_corr)):.3f}")
    print()

    random_rewards = np.array(results['random']['rewards'])
    exp_rewards = np.array(results['exponential_decay']['rewards'])

    print("Average Episode Rewards:")
    print(f"  Random: {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")
    print(f"  Exp Decay: {np.mean(exp_rewards):.2f} ± {np.std(exp_rewards):.2f}")
    print(f"  Improvement: {(np.mean(exp_rewards) - np.mean(random_rewards)):.2f}")
    print()

    # Convergence analysis
    print("🎯 CONVERGENCE ANALYSIS:")
    print("-" * 25)

    # Moving average of correlations
    window = 5
    random_ma = np.convolve(random_corr, np.ones(window)/window, mode='valid')
    exp_ma = np.convolve(exp_corr, np.ones(window)/window, mode='valid')

    print("Moving Average Correlation (last 10 episodes):")
    print(f"  Random: {random_ma[-1]:.3f}")
    print(f"  Exp Decay: {exp_ma[-1]:.3f}")
    print()

    return results

def create_comparison_plots(results):
    """Create comparison plots for the two initialization methods."""

    # Create output directory
    viz_dir = Path("experiments/initialization_comparison")
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('RIR Initialization Method Comparison', fontsize=16, fontweight='bold')

    episodes = len(results['random']['rewards'])

    # 1. Reward comparison
    ax1.plot(results['random']['rewards'], 'b-', label='Random Init', alpha=0.7)
    ax1.plot(results['exponential_decay']['rewards'], 'r-', label='Exp Decay Init', alpha=0.7)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Correlation comparison
    ax2.plot(results['random']['correlations'], 'b-', label='Random Init', alpha=0.7)
    ax2.plot(results['exponential_decay']['correlations'], 'r-', label='Exp Decay Init', alpha=0.7)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Correlation')
    ax2.set_title('RIR-Ground Truth Correlation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Final RIR comparison (average of last 5 episodes)
    random_final_avg = np.mean(results['random']['final_rirs'][-5:], axis=0)
    exp_final_avg = np.mean(results['exponential_decay']['final_rirs'][-5:], axis=0)

    time_ms = np.arange(len(random_final_avg)) / 16  # 16kHz to ms

    ax3.plot(time_ms, random_final_avg, 'b-', label='Random Init', linewidth=2)
    ax3.plot(time_ms, exp_final_avg, 'r-', label='Exp Decay Init', linewidth=2)
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Average Final RIR Estimates')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 64)  # Show first 64ms

    # 4. RIR evolution comparison (first and last episode)
    # Random init evolution
    random_first = results['random']['final_rirs'][0]
    random_last = results['random']['final_rirs'][-1]

    # Exp decay init evolution
    exp_first = results['exponential_decay']['final_rirs'][0]
    exp_last = results['exponential_decay']['final_rirs'][-1]

    ax4.plot(time_ms, random_first, 'b--', label='Random Init (Start)', alpha=0.7)
    ax4.plot(time_ms, random_last, 'b-', label='Random Init (End)', linewidth=2)
    ax4.plot(time_ms, exp_first, 'r--', label='Exp Decay Init (Start)', alpha=0.7)
    ax4.plot(time_ms, exp_last, 'r-', label='Exp Decay Init (End)', linewidth=2)
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Amplitude')
    ax4.set_title('RIR Evolution (First vs Last Episode)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 64)

    plt.tight_layout()

    # Save plot
    save_path = viz_dir / "initialization_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 Comparison plot saved to: {save_path}")

    # Create RIR initialization visualization
    create_initialization_visualization()

def create_initialization_visualization():
    """Create visualization of the two initialization methods."""

    viz_dir = Path("experiments/initialization_comparison")
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Create sample initializations
    env = RIREstimationEnv(rir_length=1024)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('RIR Initialization Methods', fontsize=14, fontweight='bold')

    # Generate multiple samples for each method
    num_samples = 5
    time_ms = np.arange(1024) / 16  # 16kHz to ms

    # Random initialization
    random_inits = []
    for _ in range(num_samples):
        rir = env._get_initial_rir_estimate(None, None, "random")
        random_inits.append(rir)

    # Exponential decay initialization
    exp_inits = []
    for _ in range(num_samples):
        rir = env._get_exponential_decay_rir()
        exp_inits.append(rir)

    # Plot random initializations
    for i, rir in enumerate(random_inits):
        ax1.plot(time_ms, rir, alpha=0.6, label=f'Sample {i+1}' if i < 3 else "")
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Random Initialization\n(Gaussian noise, σ=0.1)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 64)
    ax1.legend()

    # Plot exponential decay initializations
    for i, rir in enumerate(exp_inits):
        ax2.plot(time_ms, rir, alpha=0.6, label=f'Sample {i+1}' if i < 3 else "")
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Exponential Decay Initialization\n(Physically plausible RIR)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 64)
    ax2.legend()

    plt.tight_layout()

    save_path = viz_dir / "initialization_methods.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🎨 Initialization visualization saved to: {save_path}")

if __name__ == "__main__":
    # Run comparison
    results = compare_initialization_methods(episodes=50, max_steps=15)

    print("✅ Comparison complete!")
    print("📁 Results saved in: experiments/initialization_comparison/")