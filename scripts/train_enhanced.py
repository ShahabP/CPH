#!/usr/bin/env python3
"""
Enhanced training script with all performance improvements.

Key changes from train_all_combinations.py:
1. QN: Increased bins (5 → 8-10 for critical features)
2. DQN: Larger network, prioritized replay (implementation in updated dqn_agent.py)
3. Neural: Longer training, better hyperparameters
4. All: More episodes (600 → 1200-1500)
5. Better learning rate schedules
"""

import sys
import numpy as np
import pickle
import json
import torch
from pathlib import Path
from typing import Dict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl_framework import RIREstimationEnv
from src.rl_framework.qn_agent import QNAgent, train_qn_agent
from src.rl_framework.dqn_agent import DQNAgent, train_dqn_agent
from src.neural_rir_agent import NeuralRIRAgent, NeuralRIREnvironment, compute_rir_drr_metric


def train_qn_enhanced(init_method: str, episodes: int = 1200,
                      rir_length: int = 4096, rt60_ms: float = 300.0) -> Dict:
    """
    Train Q-Learning with ENHANCED parameters for better performance.
    
    Improvements:
    - Increased state bins (8-10 instead of 5)
    - More episodes (1200 vs 600)
    - Better action space size (12 vs 8)
    - Adaptive learning rate
    """
    print(f"\n{'='*60}")
    print(f"Training ENHANCED Q-Learning with {init_method}")
    print(f"{'='*60}")
    
    env = RIREstimationEnv(
        max_iterations=15,
        rir_length=rir_length,
        feature_dim=513,
        action_space_type='continuous',
        rir_init_method=init_method,
        rt60_ms=rt60_ms
    )
    
    # ENHANCED state bins - more granular for critical features
    state_bins = {
        'reverb_energy': 8,        # 5 → 8 (more granular energy tracking)
        'dereverb_energy': 8,      # 5 → 8 
        'rir_peak': 10,            # 5 → 10 (peak is critical for DRR)
        'rir_energy': 8,           # 5 → 8
        'rir_sparsity': 7,         # 5 → 7 (sparsity matters for dereverberation)
        'correlation': 10,         # 5 → 10 (correlation is key metric)
        'iteration': 5             # 3 → 5 (better temporal awareness)
    }
    
    agent = QNAgent(
        state_bins=state_bins,
        action_space_size=12,       # 8 → 12 (more action diversity)
        learning_rate=0.15,         # 0.1 → 0.15 (faster learning initially)
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9975,       # 0.995 → 0.9975 (slower decay for longer training)
        epsilon_min=0.01
    )
    
    # Train with more episodes
    training_stats = train_qn_agent(env, agent, episodes=episodes, max_steps=15, verbose=True)

    sample_rate = 16000
    try:
        final_rir_est = env.current_state.current_rir_estimate
        structural_drr = compute_rir_drr_metric(final_rir_est, sample_rate=sample_rate)
    except Exception:
        structural_drr = None
    
    results = {
        'agent_type': 'QN_Enhanced',
        'init_method': init_method,
        'episodes': episodes,
        'training_stats': training_stats,
        'final_stats': agent.get_stats(),
        'final_rir': final_rir_est,
        'rir_length': rir_length,
        'rt60_ms': rt60_ms,
        'final_metrics': {
            'avg_reward': np.mean([s['total_reward'] for s in training_stats[-50:]]),
            'avg_correlation': np.mean([s['final_metrics'].get('correlation', 0.0) 
                                       for s in training_stats[-50:]]),
            'avg_steps': np.mean([s['steps'] for s in training_stats[-50:]]),
            'structural_drr': structural_drr
        }
    }
    
    print(f"\nFinal metrics:")
    print(f"  Avg reward (last 50): {results['final_metrics']['avg_reward']:.2f}")
    print(f"  Avg correlation: {results['final_metrics']['avg_correlation']:.3f}")
    print(f"  Structural DRR: {structural_drr:.2f} dB" if structural_drr else "  Structural DRR: N/A")
    
    return results


def train_dqn_enhanced(init_method: str, episodes: int = 1200,
                       rir_length: int = 4096, rt60_ms: float = 300.0) -> Dict:
    """
    Train DQN with ENHANCED parameters.
    
    Improvements:
    - Larger network (hidden dims)
    - Bigger replay buffer (50k vs 10k)
    - More episodes
    - Better batch size (64 vs 32)
    """
    print(f"\n{'='*60}")
    print(f"Training ENHANCED DQN with {init_method}")
    print(f"{'='*60}")
    
    env = RIREstimationEnv(
        max_iterations=15,
        rir_length=rir_length,
        feature_dim=513,
        action_space_type='continuous',
        rir_init_method=init_method,
        rt60_ms=rt60_ms
    )
    
    state_dim = env.observation_space.shape[0]
    
    # ENHANCED DQN with larger capacity
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=12,              # 8 → 12 (more actions)
        learning_rate=5e-4,         # 1e-3 → 5e-4 (more stable)
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9975,       # Slower decay for longer training
        epsilon_min=0.01,
        memory_size=50000,          # 10000 → 50000 (larger replay buffer)
        batch_size=64,              # 32 → 64 (better gradients)
        target_update_freq=10,
        hidden_dims=[768, 512, 256, 128]  # Deeper network
    )
    
    # Train
    training_stats = train_dqn_agent(env, agent, episodes=episodes, max_steps=15, verbose=True)

    sample_rate = 16000
    try:
        final_rir_est = env.current_state.current_rir_estimate
        structural_drr = compute_rir_drr_metric(final_rir_est, sample_rate=sample_rate)
    except Exception:
        structural_drr = None
    
    results = {
        'agent_type': 'DQN_Enhanced',
        'init_method': init_method,
        'episodes': episodes,
        'training_stats': training_stats,
        'final_stats': agent.get_stats(),
        'final_rir': final_rir_est,
        'rir_length': rir_length,
        'rt60_ms': rt60_ms,
        'final_metrics': {
            'avg_reward': np.mean([s['total_reward'] for s in training_stats[-50:]]),
            'avg_correlation': np.mean([s['final_metrics'].get('avg_correlation', 0.0) 
                                       for s in training_stats[-50:]]),
            'avg_steps': np.mean([s['steps'] for s in training_stats[-50:]]),
            'avg_loss': np.mean([s.get('avg_loss', 0.0) for s in training_stats[-50:]]),
            'structural_drr': structural_drr
        }
    }
    
    print(f"\nFinal metrics:")
    print(f"  Avg reward (last 50): {results['final_metrics']['avg_reward']:.2f}")
    print(f"  Avg correlation: {results['final_metrics']['avg_correlation']:.3f}")
    print(f"  Avg loss: {results['final_metrics']['avg_loss']:.4f}")
    print(f"  Structural DRR: {structural_drr:.2f} dB" if structural_drr else "  Structural DRR: N/A")
    
    return results


def train_neural_enhanced(init_method: str, episodes: int = 1000,
                          rir_length: int = 4096, rt60_ms: float = 300.0) -> Dict:
    """
    Train Neural RIR agent with ENHANCED parameters.
    
    Improvements:
    - More episodes (1000 vs 600)
    - Better learning rate schedule
    - Larger hidden dimension
    """
    print(f"\n{'='*60}")
    print(f"Training ENHANCED Neural RIR Agent with {init_method}")
    print(f"{'='*60}")
    
    # Neural agent handles its own environment
    agent = NeuralRIRAgent(
        rir_length=rir_length,
        learning_rate=2e-4,         # 1e-4 → 2e-4 (faster learning)
        gamma=0.99
    )
    
    # Upgrade policy network to 1024 hidden_dim (from default 768)
    from src.neural_rir_agent import RIRPolicyNetwork
    agent.policy_net = RIRPolicyNetwork(rir_length, hidden_dim=1024)
    agent.optimizer = torch.optim.Adam(
        agent.policy_net.parameters(), 
        lr=2e-4, 
        weight_decay=1e-5
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
        true_rir = np.zeros(rir_length)
        if init_method == 'exponential_decay':
            # Exponential decay initialization
            true_rir = agent._get_exponential_decay_rir(rir_length, rt60_ms)
        else:
            # Random-like but still realistic for ground truth
            true_rir[0] = 1.0
            for i in range(1, min(200, rir_length)):
                true_rir[i] = np.random.uniform(0, 0.3) * np.exp(-i / 80)
        
        reverb = np.convolve(clean, true_rir, mode='same')
        
        # Initialize RIR based on method
        if init_method == 'exponential_decay':
            initial_rir = agent._get_exponential_decay_rir(rir_length, rt60_ms)
        else:
            initial_rir = np.random.normal(0, 0.1, rir_length)

        # Create environment and reset
        env = NeuralRIREnvironment(max_iterations=15, sample_rate=sample_rate, rir_length=rir_length)
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

        # Logging
        if episode % 50 == 0:
            print(f"Episode {episode}/{episodes}, Reward: {np.sum(rewards):.2f}, Correlation: {correlation:.3f}, Steps: {step+1}")
        
        training_stats.append({
            'episode': episode,
            'total_reward': np.sum(rewards),
            'steps': step + 1,
            'final_metrics': {
                'correlation': correlation
            }
        })
    
    # Get final RIR
    final_rir_est = env.current_rir
    sample_rate = 16000
    try:
        structural_drr = compute_rir_drr_metric(final_rir_est, sample_rate=sample_rate)
    except Exception:
        structural_drr = None
    
    results = {
        'agent_type': 'Neural_Enhanced',
        'init_method': init_method,
        'episodes': episodes,
        'training_stats': training_stats,
        'final_rir': final_rir_est,
        'rir_length': rir_length,
        'rt60_ms': rt60_ms,
        'final_metrics': {
            'avg_reward': np.mean([s['total_reward'] for s in training_stats[-50:]]),
            'avg_correlation': np.mean([s['final_metrics'].get('correlation', 0.0) 
                                       for s in training_stats[-50:]]),
            'avg_steps': np.mean([s['steps'] for s in training_stats[-50:]]),
            'structural_drr': structural_drr
        }
    }
    
    print(f"\nFinal metrics:")
    print(f"  Avg reward (last 50): {results['final_metrics']['avg_reward']:.2f}")
    print(f"  Avg correlation: {results['final_metrics']['avg_correlation']:.3f}")
    print(f"  Structural DRR: {structural_drr:.2f} dB" if structural_drr else "  Structural DRR: N/A")
    
    return results


def main():
    """Train enhanced versions of all agents."""
    
    # Configuration
    sample_rate = 16000
    rt60_ms = 300.0
    rir_lengths = [256, 512, 1024, 2048]  # Test all lengths
    
    # Output directory
    output_dir = Path('experiments/rir_length_enhanced')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("ENHANCED TRAINING - PERFORMANCE IMPROVEMENTS")
    print("=" * 80)
    print("\nImprovements:")
    print("  - QN: 8-10 bins (vs 5), 12 actions (vs 8), 1200 episodes (vs 600)")
    print("  - DQN: Deeper network, 50k buffer (vs 10k), 64 batch (vs 32)")
    print("  - Neural: 1024 hidden (vs 768), 1000 episodes (vs 600)")
    print()
    
    all_results = {}
    
    for rir_length in rir_lengths:
        rir_ms = (rir_length / sample_rate) * 1000
        print(f"\n{'='*80}")
        print(f"RIR LENGTH = {rir_length} samples ({rir_ms:.1f} ms)")
        print(f"{'='*80}")
        
        results_for_length = []
        
        for init_method in ['random', 'exponential_decay']:
            # Train all 3 enhanced agents
            results_for_length.append(train_qn_enhanced(init_method, 1200, rir_length, rt60_ms))
            results_for_length.append(train_dqn_enhanced(init_method, 1200, rir_length, rt60_ms))
            results_for_length.append(train_neural_enhanced(init_method, 1000, rir_length, rt60_ms))
        
        all_results[rir_length] = results_for_length
        
        # Save results
        length_dir = output_dir / f'rir_{rir_length}'
        length_dir.mkdir(exist_ok=True)
        
        with open(length_dir / 'all_results.pkl', 'wb') as f:
            pickle.dump(results_for_length, f)
        
        # Summary
        summary = {
            'rir_length_samples': rir_length,
            'rir_length_ms': rir_ms,
            'rt60_ms': rt60_ms,
            'enhanced': True,
            'results': [
                {
                    'method': f"{r['agent_type']}-{r['init_method']}",
                    'final_avg_reward': r['final_metrics']['avg_reward'],
                    'final_avg_correlation': r['final_metrics']['avg_correlation'],
                    'structural_drr': r['final_metrics'].get('structural_drr', None)
                }
                for r in results_for_length
            ]
        }
        
        with open(length_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Saved enhanced results to {length_dir}")
    
    # Save combined results
    with open(output_dir / 'all_results_combined.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\n{'='*80}")
    print("ENHANCED TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nTo plot results:")
    print(f"  python scripts/plot_rir_length_drr_improved.py --enhanced")


if __name__ == '__main__':
    main()
