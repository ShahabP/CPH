#!/usr/bin/env python3
"""
Regenerate enhanced training results - comprehensive DRR performance comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Professional publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


def regenerate_enhanced_training_results():
    """
    Comprehensive training results showing DRR performance across RIR lengths.
    """
    
    # Data
    rir_lengths_ms = [16, 32, 64, 128]
    
    # QN (Q-Network) performance
    qn_random = [6.4, -2.3, -7.2, -10.4]
    qn_exp = [6.5, -1.8, -7.0, -10.2]
    
    # DQN performance
    dqn_random = [6.0, -2.5, -7.5, -10.6]
    dqn_exp = [6.2, -2.0, -7.3, -10.5]
    
    # Neural (Ours) performance
    neural_random = [1.1, 2.9, 1.9, 1.4]
    neural_exp = [27.4, 26.8, 26.7, 24.6]
    
    # Best performance per RIR length
    best_drr = [27.4, 26.8, 26.7, 24.6]
    
    # Create figure with 6 subplots
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
    
    # Threshold line value
    threshold = 7.0
    
    # ============ TOP ROW: Agent-specific comparisons ============
    
    # 1. QN Agent Performance
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(rir_lengths_ms, qn_random, 'o-', linewidth=2.5, markersize=8,
            label='QN Random', color='#1f77b4', alpha=0.9)
    ax1.plot(rir_lengths_ms, qn_exp, 's-', linewidth=2.5, markersize=8,
            label='QN Exp Decay', color='#ff7f0e', alpha=0.9)
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.6)
    ax1.set_xlabel('RIR Length (ms)', fontweight='bold')
    ax1.set_ylabel('Structural DRR (dB)', fontweight='bold')
    ax1.set_title('QN Agent Performance (1200 episodes)', fontweight='bold', fontsize=11)
    ax1.legend(loc='lower left', framealpha=0.95, fontsize=8)
    ax1.grid(True, alpha=0.25, linestyle='--')
    ax1.set_ylim(-11, 8)
    
    # 2. DQN Agent Performance
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(rir_lengths_ms, dqn_random, 'o-', linewidth=2.5, markersize=8,
            label='DQN Random', color='#1f77b4', alpha=0.9)
    ax2.plot(rir_lengths_ms, dqn_exp, 's-', linewidth=2.5, markersize=8,
            label='DQN Exp Decay', color='#ff7f0e', alpha=0.9)
    ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.6)
    ax2.set_xlabel('RIR Length (ms)', fontweight='bold')
    ax2.set_ylabel('Structural DRR (dB)', fontweight='bold')
    ax2.set_title('DQN Agent Performance (1200 episodes)', fontweight='bold', fontsize=11)
    ax2.legend(loc='lower left', framealpha=0.95, fontsize=8)
    ax2.grid(True, alpha=0.25, linestyle='--')
    ax2.set_ylim(-11, 8)
    
    # 3. Neural Agent Performance
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(rir_lengths_ms, neural_random, 'o-', linewidth=2.5, markersize=8,
            label='Neural Random', color='#1f77b4', alpha=0.9)
    ax3.plot(rir_lengths_ms, neural_exp, 's-', linewidth=2.5, markersize=8,
            label='Neural Exp Decay', color='#ff7f0e', alpha=0.9)
    ax3.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.6)
    ax3.set_xlabel('RIR Length (ms)', fontweight='bold')
    ax3.set_ylabel('Structural DRR (dB)', fontweight='bold')
    ax3.set_title('Neural Agent Performance (1200 episodes)', fontweight='bold', fontsize=11)
    ax3.legend(loc='lower right', framealpha=0.95, fontsize=8)
    ax3.grid(True, alpha=0.25, linestyle='--')
    ax3.set_ylim(-2, 30)
    
    # ============ BOTTOM ROW: Initialization comparisons ============
    
    # 4. Random Initialization Comparison
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(rir_lengths_ms, qn_random, 'o-', linewidth=2.5, markersize=8,
            label='QN', color='#1f77b4', alpha=0.9)
    ax4.plot(rir_lengths_ms, dqn_random, 's-', linewidth=2.5, markersize=8,
            label='DQN', color='#ff7f0e', alpha=0.9)
    ax4.plot(rir_lengths_ms, neural_random, '^-', linewidth=2.5, markersize=8,
            label='Neural', color='#2ca02c', alpha=0.9)
    ax4.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.6)
    ax4.set_xlabel('RIR Length (ms)', fontweight='bold')
    ax4.set_ylabel('Structural DRR (dB)', fontweight='bold')
    ax4.set_title('Random Initialization Comparison', fontweight='bold', fontsize=11)
    ax4.legend(loc='lower left', framealpha=0.95, fontsize=8)
    ax4.grid(True, alpha=0.25, linestyle='--')
    ax4.set_ylim(-11, 8)
    
    # 5. Exponential Decay Initialization Comparison
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(rir_lengths_ms, qn_exp, 'o-', linewidth=2.5, markersize=8,
            label='QN', color='#1f77b4', alpha=0.9)
    ax5.plot(rir_lengths_ms, dqn_exp, 's-', linewidth=2.5, markersize=8,
            label='DQN', color='#ff7f0e', alpha=0.9)
    ax5.plot(rir_lengths_ms, neural_exp, '^-', linewidth=2.5, markersize=8,
            label='Neural', color='#2ca02c', alpha=0.9)
    ax5.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.6)
    ax5.set_xlabel('RIR Length (ms)', fontweight='bold')
    ax5.set_ylabel('Structural DRR (dB)', fontweight='bold')
    ax5.set_title('Exponential Decay Initialization Comparison', fontweight='bold', fontsize=11)
    ax5.legend(loc='lower left', framealpha=0.95, fontsize=8)
    ax5.grid(True, alpha=0.25, linestyle='--')
    ax5.set_ylim(-12, 30)
    
    # 6. Best Performance Summary Table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Create table
    table_data = [
        ['RIR Length', 'Best Method', 'DRR'],
        ['16 ms', 'Neural\nexponential decay', '27.4 dB'],
        ['32 ms', 'Neural\nexponential decay', '26.8 dB'],
        ['64 ms', 'Neural\nexponential decay', '26.7 dB'],
        ['128 ms', 'Neural\nexponential decay', '24.6 dB']
    ]
    
    table = ax6.table(cellText=table_data, cellLoc='center',
                     loc='center', bbox=[0, 0.1, 1, 0.85])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    
    # Style header row
    for i in range(3):
        cell = table[(0, i)]
        cell.set_facecolor('#2ca02c')
        cell.set_text_props(weight='bold', color='white', fontsize=10)
    
    # Style data rows
    for i in range(1, 5):
        for j in range(3):
            cell = table[(i, j)]
            cell.set_facecolor('#d4edda')
            cell.set_text_props(fontsize=9)
    
    ax6.set_title('Best Performance by RIR Length', fontweight='bold', fontsize=11, pad=20)
    
    # Main title
    fig.suptitle('Enhanced Training Results: DRR Performance by RIR Length\nAll Methods: 1200 episodes',
                fontsize=15, fontweight='bold', y=0.98)
    
    # Save to both locations
    output_path1 = Path(__file__).parent.parent / 'docs' / 'figures' / 'training_results_comprehensive.png'
    output_path2 = Path(__file__).parent.parent / 'experiments' / 'training_results_comprehensive.png'
    
    plt.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✓ Regenerated: training_results_comprehensive.png")
    print(f"  Saved to: docs/figures/")
    print(f"  Saved to: experiments/")
    
    plt.close()


def main():
    print("=" * 60)
    print("REGENERATING COMPREHENSIVE TRAINING RESULTS")
    print("=" * 60)
    print()
    
    regenerate_enhanced_training_results()
    
    print()
    print("=" * 60)
    print("✅ FIGURE REGENERATED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("Components:")
    print("  - QN agent performance (random vs exp decay)")
    print("  - DQN agent performance (random vs exp decay)")
    print("  - Neural agent performance (random vs exp decay)")
    print("  - Random initialization comparison (all methods)")
    print("  - Exponential initialization comparison (all methods)")
    print("  - Best performance summary table")
    print("  - 300 DPI publication quality")


if __name__ == "__main__":
    main()
