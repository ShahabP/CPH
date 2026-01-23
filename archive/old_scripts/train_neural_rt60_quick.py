#!/usr/bin/env python3
"""
Train Neural agent with different RT60 values and plot DRR gain.
Quick version with fewer episodes for faster results.
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
                      episodes: int = 300, rir_length: int = 2048) -> Dict:
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
    print(f"\nTraining Neural Agent - RT60 = {rt60_ms} ms ({episodes} episodes)")
    
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
        
        # Progress logging
        if episode % 50 == 0 or episode == episodes - 1:
            print(f"  Episode {episode}/{episodes} complete", end='\r')
    
    print(f"  Episode {episodes}/{episodes} complete ✓")
    
    # Get final RIR and compute DRR
    final_rir_est = env.current_rir
    try:
        structural_drr = compute_rir_drr_metric(final_rir_est, sample_rate=sample_rate)
    except Exception as e:
        print(f"  Warning: Could not compute DRR - {e}")
        structural_drr = 0.0
    
    results = {
        'rt60_ms': rt60_ms,
        'structural_drr': structural_drr,
        'final_rir': final_rir_est,
        'rir_length': rir_length,
        'episodes': episodes
    }
    
    print(f"  Final DRR: {structural_drr:.2f} dB")
    
    return results


def main():
    """Train Neural agent across different RT60 values and plot results."""
    
    print("=" * 80)
    print("NEURAL AGENT: DRR GAIN vs RT60 REVERBERATION TIME")
    print("=" * 80)
    print("\nQuick training mode: 300 episodes per RT60 value")
    print("Estimated time: ~15-20 minutes\n")
    
    # Configuration
    rt60_values = [100, 200, 300, 400, 500, 600, 700, 800]  # milliseconds
    rir_length = 2048  # 128ms at 16kHz
    episodes = 300  # Reduced from 1200 for faster results
    
    # Output directory
    output_dir = Path('experiments/neural_rt60_sweep')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Train for each RT60 value
    for idx, rt60 in enumerate(rt60_values):
        print(f"\n[{idx+1}/{len(rt60_values)}] RT60 = {rt60} ms")
        print("-" * 60)
        
        result = train_neural_rt60(
            rt60_ms=rt60,
            init_method='exponential_decay',
            episodes=episodes,
            rir_length=rir_length
        )
        all_results.append(result)
    
    # Save results
    with open(output_dir / 'all_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    # Extract DRR values
    drr_values = [r['structural_drr'] for r in all_results]
    
    # Create summary JSON
    summary = {
        'rt60_values_ms': rt60_values,
        'drr_values_db': drr_values,
        'rir_length': rir_length,
        'episodes': episodes,
        'method': 'Neural-Exponential_Decay'
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - GENERATING PLOTS")
    print("=" * 80)
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Neural Agent Performance vs RT60 Reverberation Time\n' + 
                 f'(Exponential Decay Init, {episodes} episodes, RIR length: {rir_length} samples)', 
                 fontsize=13, fontweight='bold')
    
    # Plot 1: DRR vs RT60
    ax1 = axes[0]
    ax1.plot(rt60_values, drr_values, 'o-', linewidth=3, markersize=12, 
            color='#d62728', label='Neural-Exp_Decay')
    ax1.set_xlabel('RT60 Reverberation Time (ms)', fontsize=12)
    ax1.set_ylabel('Structural DRR (dB)', fontsize=12)
    ax1.set_title('DRR Gain vs RT60', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=7, color='g', linestyle='--', linewidth=2, alpha=0.7, label='Target (7 dB)')
    ax1.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    ax1.legend(fontsize=11, loc='best')
    
    # Add value labels on points
    for rt60, drr in zip(rt60_values, drr_values):
        ax1.annotate(f'{drr:.1f}', xy=(rt60, drr), 
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9, alpha=0.8)
    
    # Plot 2: Summary table
    ax2 = axes[1]
    ax2.axis('off')
    
    table_data = [['RT60 (ms)', 'DRR (dB)', 'Status']]
    for rt60, drr in zip(rt60_values, drr_values):
        status = '✓ Success' if drr >= 7 else '○ Below target'
        table_data.append([
            f'{rt60}',
            f'{drr:.2f}',
            status
        ])
    
    # Add summary row
    avg_drr = np.mean(drr_values)
    success_count = sum(1 for d in drr_values if d >= 7)
    table_data.append(['', '', ''])
    table_data.append(['Average', f'{avg_drr:.2f}', f'{success_count}/{len(rt60_values)} ≥ 7dB'])
    
    table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.3, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight successful cases (DRR >= 7)
    for i in range(1, len(rt60_values) + 1):
        if drr_values[i-1] >= 7:
            for j in range(3):
                table[(i, j)].set_facecolor('#c8e6c9')
    
    # Style summary row
    for j in range(3):
        table[(len(table_data)-1, j)].set_facecolor('#e3f2fd')
        table[(len(table_data)-1, j)].set_text_props(weight='bold')
    
    ax2.set_title('Performance Summary', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'neural_drr_vs_rt60.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {plot_path}")
    
    # Copy to main experiments folder
    import shutil
    main_plot_path = Path('experiments/neural_drr_vs_rt60.png')
    shutil.copy(plot_path, main_plot_path)
    print(f"✓ Copied plot to {main_plot_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: DRR GAIN vs RT60")
    print("=" * 80)
    
    for rt60, drr in zip(rt60_values, drr_values):
        success_mark = '✓ EXCEEDS TARGET' if drr >= 7 else '○'
        print(f"RT60 = {rt60:3d} ms: DRR = {drr:6.2f} dB  {success_mark}")
    
    print("\n" + "-" * 80)
    print(f"Average DRR: {avg_drr:.2f} dB")
    print(f"Success rate (DRR >= 7 dB): {success_count}/{len(rt60_values)} ({success_count/len(rt60_values)*100:.0f}%)")
    print(f"Best DRR: {np.max(drr_values):.2f} dB at RT60 = {rt60_values[np.argmax(drr_values)]} ms")
    print(f"Worst DRR: {np.min(drr_values):.2f} dB at RT60 = {rt60_values[np.argmin(drr_values)]} ms")
    print("=" * 80)
    
    plt.show()


if __name__ == '__main__':
    main()
