#!/usr/bin/env python3
"""
Train Neural agent with different RT60 values and plot DRR gain.
Tests RT60 from 100ms to 800ms to understand performance across reverberation levels.
"""

import sys
import numpy as np
import pickle
import json
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neural_rir_agent import NeuralRIRAgent, NeuralRIREnvironment, compute_rir_drr_metric


def train_neural_rt60(rt60_ms: float, init_method: str = 'exponential_decay', 
                      episodes: int = 1200, rir_length: int = 2048) -> Dict:
    """
    Train Neural RIR agent with specific RT60 value.
    
    Args:
        rt60_ms: RT60 reverberation time in milliseconds
        init_method: Initialization method ('exponential_decay' or 'random')
        episodes: Number of training episodes
        rir_length: Length of RIR in samples
        
    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*60}")
    print(f"Training Neural Agent - RT60 = {rt60_ms} ms")
    print(f"{'='*60}")
    
    # Create agent
    agent = NeuralRIRAgent(
        rir_length=rir_length,
        learning_rate=2e-4,
        gamma=0.99
    )
    
    # Upgrade policy network
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
        
        # Generate true RIR with specific RT60
        true_rir = np.zeros(rir_length)
        if init_method == 'exponential_decay':
            # Exponential decay with specified RT60
            true_rir = agent._get_exponential_decay_rir(rir_length, rt60_ms)
        else:
            # Random initialization
            true_rir[0] = 1.0
            for i in range(1, min(200, rir_length)):
                true_rir[i] = np.random.uniform(0, 0.3) * np.exp(-i / 80)
        
        reverb = np.convolve(clean, true_rir, mode='same')
        
        # Initialize RIR
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
        
        # Train agent
        train_info = agent.end_episode()
        
        # Compute correlation with true RIR
        final_rir = rir_state
        min_len = min(len(final_rir), len(true_rir))
        correlation = np.corrcoef(final_rir[:min_len], true_rir[:min_len])[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # Logging
        if episode % 100 == 0:
            print(f"Episode {episode}/{episodes}, Reward: {np.sum(rewards):.2f}, " + 
                  f"Correlation: {correlation:.3f}, Steps: {step+1}")
        
        training_stats.append({
            'episode': episode,
            'total_reward': np.sum(rewards),
            'steps': step + 1,
            'final_metrics': {
                'correlation': correlation
            }
        })
    
    # Get final RIR and compute DRR
    final_rir_est = env.current_rir
    try:
        structural_drr = compute_rir_drr_metric(final_rir_est, sample_rate=sample_rate)
    except Exception:
        structural_drr = None
    
    results = {
        'agent_type': 'Neural_Enhanced',
        'init_method': init_method,
        'rt60_ms': rt60_ms,
        'episodes': episodes,
        'training_stats': training_stats,
        'final_rir': final_rir_est,
        'rir_length': rir_length,
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
    """Train Neural agent across different RT60 values and plot results."""
    
    print("=" * 80)
    print("NEURAL AGENT: DRR GAIN vs RT60 REVERBERATION TIME")
    print("=" * 80)
    
    # Configuration
    rt60_values = [100, 200, 300, 400, 500, 600, 700, 800]  # milliseconds
    rir_length = 2048  # 128ms at 16kHz
    episodes = 1200
    
    # Output directory
    output_dir = Path('experiments/neural_rt60_sweep')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Train for each RT60 value
    for rt60 in rt60_values:
        result = train_neural_rt60(
            rt60_ms=rt60,
            init_method='exponential_decay',
            episodes=episodes,
            rir_length=rir_length
        )
        all_results.append(result)
        
        # Save individual result
        result_path = output_dir / f'rt60_{rt60}ms.pkl'
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"✓ Saved result to {result_path}")
    
    # Save combined results
    with open(output_dir / 'all_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    # Extract DRR values
    drr_values = [r['final_metrics']['structural_drr'] for r in all_results]
    correlations = [r['final_metrics']['avg_correlation'] for r in all_results]
    rewards = [r['final_metrics']['avg_reward'] for r in all_results]
    
    # Create summary JSON
    summary = {
        'rt60_values_ms': rt60_values,
        'drr_values_db': drr_values,
        'correlations': correlations,
        'avg_rewards': rewards,
        'rir_length': rir_length,
        'episodes': episodes,
        'method': 'Neural-Exponential_Decay'
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Neural Agent Performance vs RT60 Reverberation Time\n' + 
                 f'(Exponential Decay Init, {episodes} episodes, {rir_length} samples)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: DRR vs RT60
    ax1 = axes[0, 0]
    ax1.plot(rt60_values, drr_values, 'o-', linewidth=2.5, markersize=10, 
            color='#d62728', label='Neural-Exp_Decay')
    ax1.set_xlabel('RT60 Reverberation Time (ms)', fontsize=11)
    ax1.set_ylabel('Structural DRR (dB)', fontsize=11)
    ax1.set_title('DRR Gain vs RT60', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=7, color='g', linestyle='--', linewidth=2, alpha=0.7, label='Target (7 dB)')
    ax1.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    ax1.legend(fontsize=10)
    
    # Plot 2: Correlation vs RT60
    ax2 = axes[0, 1]
    ax2.plot(rt60_values, correlations, 's-', linewidth=2.5, markersize=10, 
            color='#2ca02c')
    ax2.set_xlabel('RT60 Reverberation Time (ms)', fontsize=11)
    ax2.set_ylabel('Correlation', fontsize=11)
    ax2.set_title('Correlation with True RIR vs RT60', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # Plot 3: Reward vs RT60
    ax3 = axes[1, 0]
    ax3.plot(rt60_values, rewards, '^-', linewidth=2.5, markersize=10, 
            color='#ff7f0e')
    ax3.set_xlabel('RT60 Reverberation Time (ms)', fontsize=11)
    ax3.set_ylabel('Average Reward (last 50 episodes)', fontsize=11)
    ax3.set_title('Training Reward vs RT60', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = [['RT60 (ms)', 'DRR (dB)', 'Correlation', 'Reward']]
    for rt60, drr, corr, rew in zip(rt60_values, drr_values, correlations, rewards):
        success = '✓' if drr >= 7 else ''
        table_data.append([
            f'{rt60}',
            f'{drr:.2f} {success}',
            f'{corr:.3f}',
            f'{rew:.1f}'
        ])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.3, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight successful cases (DRR >= 7)
    for i in range(1, len(table_data)):
        if drr_values[i-1] >= 7:
            for j in range(4):
                table[(i, j)].set_facecolor('#c8e6c9')
    
    ax4.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'neural_drr_vs_rt60.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot to {plot_path}")
    
    # Copy to main experiments folder for easy access
    import shutil
    main_plot_path = Path('experiments/neural_drr_vs_rt60.png')
    shutil.copy(plot_path, main_plot_path)
    print(f"✓ Copied plot to {main_plot_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: DRR GAIN vs RT60")
    print("=" * 80)
    
    for rt60, drr, corr, rew in zip(rt60_values, drr_values, correlations, rewards):
        success_mark = '✓ EXCEEDS TARGET' if drr >= 7 else ''
        print(f"RT60 = {rt60:3d} ms: DRR = {drr:6.2f} dB, " + 
              f"Corr = {corr:.3f}, Reward = {rew:6.1f}  {success_mark}")
    
    print("\n" + "=" * 80)
    print(f"Success rate (DRR >= 7 dB): {sum(1 for d in drr_values if d >= 7)}/{len(drr_values)}")
    print(f"Average DRR: {np.mean(drr_values):.2f} dB")
    print(f"Best DRR: {np.max(drr_values):.2f} dB at RT60 = {rt60_values[np.argmax(drr_values)]} ms")
    print("=" * 80)
    
    plt.show()


if __name__ == '__main__':
    main()
