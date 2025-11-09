"""
Train all combinations of agents and initialization methods.

This script trains:
1. Q-Learning (QN) with random initialization
2. Q-Learning (QN) with exponential decay initialization
3. Deep Q-Network (DQN) with random initialization
4. Deep Q-Network (DQN) with exponential decay initialization
5. Neural RIR Agent with random initialization
6. Neural RIR Agent with exponential decay initialization
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Import agents and environment
from src.rl_framework import RIREstimationEnv
from src.rl_framework.qn_agent import QNAgent, train_qn_agent
from src.rl_framework.dqn_agent import DQNAgent, train_dqn_agent
from src.neural_rir_agent import NeuralRIRAgent, NeuralRIREnvironment


def create_results_dir() -> Path:
    """Create directory for results."""
    results_dir = Path("experiments/all_combinations")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def train_qn_combination(init_method: str, episodes: int = 500) -> Dict:
    """
    Train Q-Learning agent with specified initialization.
    
    Args:
        init_method: 'random' or 'exponential_decay'
        episodes: Number of training episodes
        
    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*60}")
    print(f"Training Q-Learning with {init_method} initialization")
    print(f"{'='*60}")
    
    # Create environment
    env = RIREstimationEnv(
        max_iterations=15,
        rir_length=1024,
        feature_dim=513,
        action_space_type='continuous',
        rir_init_method=init_method
    )
    
    # Create agent
    state_bins = {
        'reverb_energy': 5,
        'dereverb_energy': 5,
        'rir_peak': 5,
        'rir_energy': 5,
        'rir_sparsity': 5,
        'correlation': 5,
        'iteration': 3
    }
    
    agent = QNAgent(
        state_bins=state_bins,
        action_space_size=8,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Train
    training_stats = train_qn_agent(env, agent, episodes=episodes, max_steps=15, verbose=True)
    
    # Extract metrics
    results = {
        'agent_type': 'QN',
        'init_method': init_method,
        'episodes': episodes,
        'training_stats': training_stats,
        'final_stats': agent.get_stats(),
        'final_metrics': {
            'avg_reward': np.mean([s['total_reward'] for s in training_stats[-50:]]),
            'avg_correlation': np.mean([s['final_metrics'].get('correlation', 0.0) 
                                       for s in training_stats[-50:]]),
            'avg_steps': np.mean([s['steps'] for s in training_stats[-50:]])
        }
    }
    
    return results


def train_dqn_combination(init_method: str, episodes: int = 500) -> Dict:
    """
    Train DQN agent with specified initialization.
    
    Args:
        init_method: 'random' or 'exponential_decay'
        episodes: Number of training episodes
        
    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*60}")
    print(f"Training DQN with {init_method} initialization")
    print(f"{'='*60}")
    
    # Create environment
    env = RIREstimationEnv(
        max_iterations=15,
        rir_length=1024,
        feature_dim=513,
        action_space_type='continuous',
        rir_init_method=init_method
    )
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=8,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        memory_size=10000,
        batch_size=32,
        target_update_freq=10
    )
    
    # Train
    training_stats = train_dqn_agent(env, agent, episodes=episodes, max_steps=15, verbose=True)
    
    # Extract metrics
    results = {
        'agent_type': 'DQN',
        'init_method': init_method,
        'episodes': episodes,
        'training_stats': training_stats,
        'final_stats': agent.get_stats(),
        'final_metrics': {
            'avg_reward': np.mean([s['total_reward'] for s in training_stats[-50:]]),
            'avg_correlation': np.mean([s['final_metrics'].get('correlation', 0.0) 
                                       for s in training_stats[-50:]]),
            'avg_steps': np.mean([s['steps'] for s in training_stats[-50:]]),
            'avg_loss': np.mean([s['avg_loss'] for s in training_stats[-50:]])
        }
    }
    
    return results


def train_neural_combination(init_method: str, episodes: int = 500) -> Dict:
    """
    Train Neural RIR agent with specified initialization.
    
    Args:
        init_method: 'random' or 'exponential_decay'
        episodes: Number of training episodes
        
    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*60}")
    print(f"Training Neural RIR Agent with {init_method} initialization")
    print(f"{'='*60}")
    
    # Create agent
    agent = NeuralRIRAgent(
        rir_length=1024,
        learning_rate=3e-4,
        gamma=0.95,
        update_scale=0.05
    )
    
    # Training loop
    training_stats = []
    sample_rate = 16000
    
    for episode in range(episodes):
        # Generate synthetic reverberant speech
        duration = 2.5
        t = np.linspace(0, duration, int(duration * sample_rate))
        clean = 0.2 * np.sum([
            np.sin(2 * np.pi * f * t) * np.exp(-t * 0.5)
            for f in [440, 880, 1320]
        ], axis=0)
        
        # Generate true RIR based on initialization method
        true_rir = np.zeros(1024)
        if init_method == 'exponential_decay':
            # Exponential decay initialization
            true_rir[0] = 1.0
            for i in range(1, min(200, len(true_rir))):
                if np.random.random() < 0.1:  # Sparse early reflections
                    true_rir[i] = 0.3 * np.exp(-i / 100) * (0.5 + np.random.random())
            for i in range(200, len(true_rir)):
                true_rir[i] = 0.1 * np.exp(-i / 50) * (0.8 + 0.4 * np.random.random())
        else:
            # Random-like but still realistic for ground truth
            true_rir[0] = 1.0
            for i in range(1, 200):
                true_rir[i] = np.random.uniform(0, 0.3) * np.exp(-i / 80)
        
        reverb = np.convolve(clean, true_rir, mode='same')
        
        # Initialize RIR based on method
        if init_method == 'exponential_decay':
            initial_rir = agent._get_exponential_decay_rir(1024)
        else:
            initial_rir = np.random.normal(0, 0.1, 1024)
        
        # Create environment and reset
        env = NeuralRIREnvironment(max_iterations=15, sample_rate=sample_rate)
        env.reset(reverb, clean, initial_rir=initial_rir)
        
        # Run episode
        rewards = []
        for step in range(env.max_iterations):
            rir_state, reward, terminated, info = env.step(agent)
            rewards.append(reward)
            if terminated:
                break
        
        # Train agent at end of episode
        train_info = agent.end_episode()
        
        # Compute correlation with true RIR
        final_rir = rir_state
        min_len = min(len(final_rir), len(true_rir))
        correlation = np.corrcoef(final_rir[:min_len], true_rir[:min_len])[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # Record statistics
        episode_stats = {
            'episode': episode,
            'total_reward': sum(rewards),
            'steps': len(rewards),
            'final_metrics': {
                'correlation': correlation,
                'rir_energy': float(np.sum(final_rir ** 2)),
                'rir_peak': float(np.max(np.abs(final_rir)))
            },
            'train_info': train_info if train_info else {}
        }
        training_stats.append(episode_stats)
        
        if episode % 50 == 0:
            print(f"Episode {episode}/{episodes}, "
                  f"Reward: {episode_stats['total_reward']:.2f}, "
                  f"Correlation: {correlation:.3f}, "
                  f"Steps: {episode_stats['steps']}")
    
    # Extract metrics
    results = {
        'agent_type': 'Neural',
        'init_method': init_method,
        'episodes': episodes,
        'training_stats': training_stats,
        'final_metrics': {
            'avg_reward': np.mean([s['total_reward'] for s in training_stats[-50:]]),
            'avg_correlation': np.mean([s['final_metrics'].get('correlation', 0.0) 
                                       for s in training_stats[-50:]]),
            'avg_steps': np.mean([s['steps'] for s in training_stats[-50:]])
        }
    }
    
    return results


def plot_comparison(all_results: List[Dict], results_dir: Path):
    """
    Create comprehensive comparison plots.
    
    Args:
        all_results: List of results dictionaries
        results_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Comparison of All Agent-Initialization Combinations', fontsize=16, fontweight='bold')
    
    colors = {
        'QN-random': '#1f77b4',
        'QN-exponential_decay': '#aec7e8',
        'DQN-random': '#ff7f0e',
        'DQN-exponential_decay': '#ffbb78',
        'Neural-random': '#2ca02c',
        'Neural-exponential_decay': '#98df8a'
    }
    
    # Plot 1: Rewards over episodes
    ax = axes[0, 0]
    for result in all_results:
        label = f"{result['agent_type']}-{result['init_method']}"
        stats = result['training_stats']
        episodes = [s['episode'] for s in stats]
        rewards = [s['total_reward'] for s in stats]
        
        # Smooth rewards
        window = 20
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(episodes[:len(smoothed)], smoothed, label=label, color=colors.get(label, 'gray'), linewidth=2)
        else:
            ax.plot(episodes, rewards, label=label, color=colors.get(label, 'gray'), linewidth=2)
    
    ax.set_xlabel('Episode', fontweight='bold')
    ax.set_ylabel('Total Reward', fontweight='bold')
    ax.set_title('Learning Curves (Smoothed)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Correlation over episodes
    ax = axes[0, 1]
    for result in all_results:
        label = f"{result['agent_type']}-{result['init_method']}"
        stats = result['training_stats']
        episodes = [s['episode'] for s in stats]
        correlations = [s['final_metrics'].get('correlation', 0.0) for s in stats]
        
        # Smooth correlations
        window = 20
        if len(correlations) >= window:
            smoothed = np.convolve(correlations, np.ones(window)/window, mode='valid')
            ax.plot(episodes[:len(smoothed)], smoothed, label=label, color=colors.get(label, 'gray'), linewidth=2)
        else:
            ax.plot(episodes, correlations, label=label, color=colors.get(label, 'gray'), linewidth=2)
    
    ax.set_xlabel('Episode', fontweight='bold')
    ax.set_ylabel('RIR Correlation', fontweight='bold')
    ax.set_title('RIR Estimation Quality')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Steps per episode
    ax = axes[0, 2]
    for result in all_results:
        label = f"{result['agent_type']}-{result['init_method']}"
        stats = result['training_stats']
        episodes = [s['episode'] for s in stats]
        steps = [s['steps'] for s in stats]
        
        # Smooth steps
        window = 20
        if len(steps) >= window:
            smoothed = np.convolve(steps, np.ones(window)/window, mode='valid')
            ax.plot(episodes[:len(smoothed)], smoothed, label=label, color=colors.get(label, 'gray'), linewidth=2)
        else:
            ax.plot(episodes, steps, label=label, color=colors.get(label, 'gray'), linewidth=2)
    
    ax.set_xlabel('Episode', fontweight='bold')
    ax.set_ylabel('Steps to Convergence', fontweight='bold')
    ax.set_title('Convergence Speed')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Final performance comparison (bar chart)
    ax = axes[1, 0]
    labels = [f"{r['agent_type']}\n{r['init_method'][:3]}" for r in all_results]
    rewards = [r['final_metrics']['avg_reward'] for r in all_results]
    colors_list = [colors.get(f"{r['agent_type']}-{r['init_method']}", 'gray') for r in all_results]
    
    bars = ax.bar(range(len(labels)), rewards, color=colors_list, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Average Reward (Final 50 eps)', fontweight='bold')
    ax.set_title('Final Performance: Reward')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 5: Final correlation comparison (bar chart)
    ax = axes[1, 1]
    correlations = [r['final_metrics']['avg_correlation'] for r in all_results]
    
    bars = ax.bar(range(len(labels)), correlations, color=colors_list, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Average Correlation (Final 50 eps)', fontweight='bold')
    ax.set_title('Final Performance: RIR Quality')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 6: Final steps comparison (bar chart)
    ax = axes[1, 2]
    final_steps = [r['final_metrics']['avg_steps'] for r in all_results]
    
    bars = ax.bar(range(len(labels)), final_steps, color=colors_list, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Average Steps (Final 50 eps)', fontweight='bold')
    ax.set_title('Final Performance: Convergence')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'all_combinations_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot: {results_dir / 'all_combinations_comparison.png'}")
    plt.close()


def save_results(all_results: List[Dict], results_dir: Path):
    """Save results to files."""
    # Save full results as pickle
    with open(results_dir / 'all_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    # Save summary as JSON (without full training stats)
    summary = []
    for result in all_results:
        summary.append({
            'agent_type': result['agent_type'],
            'init_method': result['init_method'],
            'episodes': result['episodes'],
            'final_metrics': result['final_metrics']
        })
    
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved results to {results_dir}")


def print_summary_table(all_results: List[Dict]):
    """Print a summary table of all results."""
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Agent':<15} {'Init Method':<20} {'Avg Reward':<15} {'Avg Correlation':<18} {'Avg Steps':<12}")
    print("-"*80)
    
    for result in all_results:
        agent = result['agent_type']
        init = result['init_method']
        reward = result['final_metrics']['avg_reward']
        corr = result['final_metrics']['avg_correlation']
        steps = result['final_metrics']['avg_steps']
        
        print(f"{agent:<15} {init:<20} {reward:>14.2f} {corr:>17.3f} {steps:>11.1f}")
    
    print("="*80)


def main():
    """Main training function."""
    print("="*80)
    print("TRAINING ALL AGENT-INITIALIZATION COMBINATIONS")
    print("="*80)
    print("This will train 6 combinations:")
    print("  1. Q-Learning (QN) - Random Init")
    print("  2. Q-Learning (QN) - Exponential Decay Init")
    print("  3. Deep Q-Network (DQN) - Random Init")
    print("  4. Deep Q-Network (DQN) - Exponential Decay Init")
    print("  5. Neural RIR Agent - Random Init")
    print("  6. Neural RIR Agent - Exponential Decay Init")
    print("="*80)
    
    results_dir = create_results_dir()
    all_results = []
    
    # Number of episodes for each agent type
    qn_episodes = 300
    dqn_episodes = 300
    neural_episodes = 200
    
    # Train all combinations
    combinations = [
        ('qn', 'random', qn_episodes),
        ('qn', 'exponential_decay', qn_episodes),
        ('dqn', 'random', dqn_episodes),
        ('dqn', 'exponential_decay', dqn_episodes),
        ('neural', 'random', neural_episodes),
        ('neural', 'exponential_decay', neural_episodes)
    ]
    
    for agent_type, init_method, episodes in combinations:
        if agent_type == 'qn':
            result = train_qn_combination(init_method, episodes)
        elif agent_type == 'dqn':
            result = train_dqn_combination(init_method, episodes)
        else:  # neural
            result = train_neural_combination(init_method, episodes)
        
        all_results.append(result)
    
    # Save results
    save_results(all_results, results_dir)
    
    # Create comparison plots
    plot_comparison(all_results, results_dir)
    
    # Print summary
    print_summary_table(all_results)
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
