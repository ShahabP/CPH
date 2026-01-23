#!/usr/bin/env python3
"""
Train Neural agent with different RT60 values and RIR lengths.
Plot DRR gain vs RT60 for all RIR lengths in the same figure.
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


def train_neural_rt60(rt60_ms: float, rir_length: int, episodes: int = 300) -> Dict:
    """Train Neural RIR agent with specific RT60 and RIR length."""
    
    rir_ms = rir_length / 16.0
    print(f"  Training: RT60={rt60_ms}ms, RIR={rir_length} samples ({rir_ms:.0f}ms), {episodes} episodes", end='')
    
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
        true_rir = agent._get_exponential_decay_rir(rir_length, rt60_ms)
        reverb = np.convolve(clean, true_rir, mode='same')
        
        # Initialize RIR
        initial_rir = agent._get_exponential_decay_rir(rir_length, rt60_ms)
        
        # Create environment and reset
        env = NeuralRIREnvironment(max_iterations=15, sample_rate=sample_rate, rir_length=rir_length)
        env.reset(reverb, clean, initial_rir=initial_rir)
        
        # Run episode
        for step in range(env.max_iterations):
            rir_state, reward, terminated, info = env.step(agent)
            if terminated:
                break
        
        # Train agent
        agent.end_episode()
    
    # Get final RIR and compute DRR
    final_rir_est = env.current_rir
    try:
        structural_drr = compute_rir_drr_metric(final_rir_est, sample_rate=sample_rate)
    except Exception as e:
        structural_drr = 0.0
    
    print(f" → DRR: {structural_drr:.2f} dB")
    
    results = {
        'rt60_ms': rt60_ms,
        'rir_length': rir_length,
        'structural_drr': structural_drr,
        'final_rir': final_rir_est,
        'episodes': episodes
    }
    
    return results


def main():
    """Train Neural agent across different RT60 values and RIR lengths."""
    
    print("=" * 80)
    print("NEURAL AGENT: DRR vs RT60 for Different RIR Lengths")
    print("=" * 80)
    
    # Configuration
    rt60_values = [100, 200, 300, 400, 500, 600, 700, 800]  # milliseconds
    rir_lengths = [256, 512, 1024, 2048]  # samples
    episodes = 300  # Quick training
    
    # Output directory
    output_dir = Path('experiments/neural_rt60_rir_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  RT60 values: {rt60_values} ms")
    print(f"  RIR lengths: {rir_lengths} samples")
    print(f"  Episodes per training: {episodes}")
    print(f"  Total trainings: {len(rt60_values) * len(rir_lengths)} = {len(rt60_values)} RT60 × {len(rir_lengths)} RIR lengths")
    print(f"  Estimated time: ~45-60 minutes\n")
    
    # Store all results
    all_results = {rir_len: [] for rir_len in rir_lengths}
    
    total_count = 0
    total_trainings = len(rt60_values) * len(rir_lengths)
    
    # Train for each combination
    for rir_len in rir_lengths:
        rir_ms = rir_len / 16.0
        print(f"\n{'='*80}")
        print(f"RIR LENGTH: {rir_len} samples ({rir_ms:.0f} ms)")
        print(f"{'='*80}")
        
        for rt60 in rt60_values:
            total_count += 1
            print(f"[{total_count}/{total_trainings}]", end=' ')
            
            result = train_neural_rt60(
                rt60_ms=rt60,
                rir_length=rir_len,
                episodes=episodes
            )
            all_results[rir_len].append(result)
        
        # Save intermediate results for this RIR length
        with open(output_dir / f'results_rir_{rir_len}.pkl', 'wb') as f:
            pickle.dump(all_results[rir_len], f)
    
    # Save all results
    with open(output_dir / 'all_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    # Create summary
    summary = {
        'rt60_values_ms': rt60_values,
        'rir_lengths_samples': rir_lengths,
        'episodes': episodes,
        'results': {}
    }
    
    for rir_len in rir_lengths:
        drr_values = [r['structural_drr'] for r in all_results[rir_len]]
        summary['results'][str(rir_len)] = {
            'drr_values_db': drr_values,
            'average_drr': float(np.mean(drr_values)),
            'best_drr': float(np.max(drr_values)),
            'worst_drr': float(np.min(drr_values))
        }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - GENERATING PLOTS")
    print("=" * 80)
    
    # Create comprehensive plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Neural Agent: DRR vs RT60 for Different RIR Lengths\n' + 
                 f'(Exponential Decay Init, {episodes} episodes per training)', 
                 fontsize=14, fontweight='bold')
    
    # Define colors and markers for each RIR length
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '-', '--', '--']  # Solid for first two, dashed for last two
    
    # Plot 1: DRR vs RT60 for all RIR lengths
    ax1 = axes[0]
    
    for idx, rir_len in enumerate(rir_lengths):
        rir_ms = rir_len / 16.0
        drr_values = [r['structural_drr'] for r in all_results[rir_len]]
        
        ax1.plot(rt60_values, drr_values, 
                marker=markers[idx], linestyle=linestyles[idx], linewidth=2.5, markersize=10,
                color=colors[idx], label=f'RIR Length: {rir_len} samples ({rir_ms:.0f} ms)',
                alpha=0.8)
    
    ax1.set_xlabel('RT60 Reverberation Time (ms)', fontsize=12)
    ax1.set_ylabel('DRR Gain (dB)', fontsize=12)
    ax1.set_title('DRR Gain vs RT60', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    
    # Plot 2: Summary statistics table
    ax2 = axes[1]
    ax2.axis('off')
    
    table_data = [['RIR Length', 'Avg DRR', 'Best', 'Worst', 'Success']]
    
    for rir_len in rir_lengths:
        rir_ms = rir_len / 16.0
        drr_values = [r['structural_drr'] for r in all_results[rir_len]]
        avg_drr = np.mean(drr_values)
        best_drr = np.max(drr_values)
        worst_drr = np.min(drr_values)
        success_count = sum(1 for d in drr_values if d >= 7)
        
        table_data.append([
            f'{rir_len}\n({rir_ms:.0f} ms)',
            f'{avg_drr:.2f} dB',
            f'{best_drr:.2f} dB',
            f'{worst_drr:.2f} dB',
            f'{success_count}/{len(rt60_values)}'
        ])
    
    table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.2, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code rows
    for i, color in enumerate(colors):
        for j in range(5):
            table[(i+1, j)].set_facecolor(color)
            table[(i+1, j)].set_alpha(0.3)
    
    ax2.set_title('Performance Summary by RIR Length', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'neural_drr_vs_rt60_all_rir.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {plot_path}")
    
    # Copy to main experiments folder
    import shutil
    main_plot_path = Path('experiments/neural_drr_vs_rt60_all_rir.png')
    shutil.copy(plot_path, main_plot_path)
    print(f"✓ Copied plot to {main_plot_path}")
    
    # Print detailed summary
    print("\n" + "=" * 80)
    print("DETAILED SUMMARY: DRR vs RT60 for All RIR Lengths")
    print("=" * 80)
    
    for rir_len in rir_lengths:
        rir_ms = rir_len / 16.0
        print(f"\n{'─'*80}")
        print(f"RIR Length: {rir_len} samples ({rir_ms:.0f} ms)")
        print(f"{'─'*80}")
        
        drr_values = [r['structural_drr'] for r in all_results[rir_len]]
        
        for rt60, drr in zip(rt60_values, drr_values):
            success_mark = '✓' if drr >= 7 else '○'
            print(f"  RT60 = {rt60:3d} ms: DRR = {drr:6.2f} dB  {success_mark}")
        
        avg_drr = np.mean(drr_values)
        success_count = sum(1 for d in drr_values if d >= 7)
        print(f"\n  Average: {avg_drr:.2f} dB")
        print(f"  Success rate: {success_count}/{len(rt60_values)} ({success_count/len(rt60_values)*100:.0f}%)")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    
    all_drr = []
    for rir_len in rir_lengths:
        all_drr.extend([r['structural_drr'] for r in all_results[rir_len]])
    
    overall_avg = np.mean(all_drr)
    overall_success = sum(1 for d in all_drr if d >= 7)
    total_trainings = len(all_drr)
    
    print(f"Total trainings: {total_trainings}")
    print(f"Overall average DRR: {overall_avg:.2f} dB")
    print(f"Overall success rate: {overall_success}/{total_trainings} ({overall_success/total_trainings*100:.0f}%)")
    print(f"Best DRR achieved: {np.max(all_drr):.2f} dB")
    print(f"Worst DRR achieved: {np.min(all_drr):.2f} dB")
    print("=" * 80)
    
    plt.show()


if __name__ == '__main__':
    main()
