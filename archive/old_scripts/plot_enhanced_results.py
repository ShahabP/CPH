#!/usr/bin/env python3
"""
Plot enhanced training results from saved summary files.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results():
    """Load all enhanced results from summary files."""
    base_dir = Path("experiments/rir_length_enhanced")
    rir_lengths = [256, 512, 1024, 2048]
    
    results = {}
    for rir_len in rir_lengths:
        summary_path = base_dir / f"rir_{rir_len}" / "summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                data = json.load(f)
                results[rir_len] = data
    
    return results


def load_baseline_results():
    """Load baseline results for comparison."""
    baseline_path = Path("experiments/drr_summary.json")
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            return json.load(f)
    return None


def extract_drr_by_method(results, rir_lengths):
    """Extract DRR values organized by method."""
    methods = [
        'QN_Enhanced-random',
        'QN_Enhanced-exponential_decay',
        'DQN_Enhanced-random', 
        'DQN_Enhanced-exponential_decay',
        'Neural_Enhanced-random',
        'Neural_Enhanced-exponential_decay'
    ]
    
    drr_data = {method: [] for method in methods}
    
    for rir_len in rir_lengths:
        if rir_len in results:
            for result in results[rir_len]['results']:
                method = result['method']
                drr = result.get('structural_drr', 0)
                if method in drr_data:
                    drr_data[method].append(drr)
    
    return drr_data


def main():
    print("\n" + "=" * 80)
    print("PLOTTING ENHANCED TRAINING RESULTS")
    print("=" * 80)
    
    # Load data
    results = load_results()
    baseline_results = load_baseline_results()
    
    rir_lengths = [256, 512, 1024, 2048]
    rir_lengths_ms = [rl / 16.0 for rl in rir_lengths]
    
    # Extract DRR values
    drr_data = extract_drr_by_method(results, rir_lengths)
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Enhanced Training Results: DRR Performance by RIR Length\n' + 
                 'All Methods: 1200 episodes', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: QN methods
    ax = axes[0, 0]
    ax.plot(rir_lengths_ms, drr_data['QN_Enhanced-random'], 'o-', 
            linewidth=2.5, markersize=10, label='QN Random')
    ax.plot(rir_lengths_ms, drr_data['QN_Enhanced-exponential_decay'], 's-', 
            linewidth=2.5, markersize=10, label='QN Exp Decay')
    ax.set_xlabel('RIR Length (ms)', fontsize=11)
    ax.set_ylabel('Structural DRR (dB)', fontsize=11)
    ax.set_title('QN Agent Performance (1200 episodes)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=7, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (7 dB)')
    
    # Plot 2: DQN methods
    ax = axes[0, 1]
    ax.plot(rir_lengths_ms, drr_data['DQN_Enhanced-random'], 'o-', 
            linewidth=2.5, markersize=10, label='DQN Random')
    ax.plot(rir_lengths_ms, drr_data['DQN_Enhanced-exponential_decay'], 's-', 
            linewidth=2.5, markersize=10, label='DQN Exp Decay')
    ax.set_xlabel('RIR Length (ms)', fontsize=11)
    ax.set_ylabel('Structural DRR (dB)', fontsize=11)
    ax.set_title('DQN Agent Performance (1200 episodes)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=7, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Plot 3: Neural methods
    ax = axes[0, 2]
    ax.plot(rir_lengths_ms, drr_data['Neural_Enhanced-random'], 'o-', 
            linewidth=2.5, markersize=10, label='Neural Random')
    ax.plot(rir_lengths_ms, drr_data['Neural_Enhanced-exponential_decay'], 's-', 
            linewidth=2.5, markersize=10, label='Neural Exp Decay')
    ax.set_xlabel('RIR Length (ms)', fontsize=11)
    ax.set_ylabel('Structural DRR (dB)', fontsize=11)
    ax.set_title('Neural Agent Performance (1200 episodes)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=7, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Plot 4: Overall comparison - Random init
    ax = axes[1, 0]
    ax.plot(rir_lengths_ms, drr_data['QN_Enhanced-random'], 'o-', 
            linewidth=2.5, markersize=10, label='QN', color='#1f77b4')
    ax.plot(rir_lengths_ms, drr_data['DQN_Enhanced-random'], 's-', 
            linewidth=2.5, markersize=10, label='DQN', color='#ff7f0e')
    ax.plot(rir_lengths_ms, drr_data['Neural_Enhanced-random'], '^-', 
            linewidth=2.5, markersize=10, label='Neural', color='#2ca02c')
    ax.set_xlabel('RIR Length (ms)', fontsize=11)
    ax.set_ylabel('Structural DRR (dB)', fontsize=11)
    ax.set_title('Random Initialization Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=7, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Plot 5: Overall comparison - Exp decay init
    ax = axes[1, 1]
    ax.plot(rir_lengths_ms, drr_data['QN_Enhanced-exponential_decay'], 'o-', 
            linewidth=2.5, markersize=10, label='QN', color='#1f77b4')
    ax.plot(rir_lengths_ms, drr_data['DQN_Enhanced-exponential_decay'], 's-', 
            linewidth=2.5, markersize=10, label='DQN', color='#ff7f0e')
    ax.plot(rir_lengths_ms, drr_data['Neural_Enhanced-exponential_decay'], '^-', 
            linewidth=2.5, markersize=10, label='Neural', color='#2ca02c')
    ax.set_xlabel('RIR Length (ms)', fontsize=11)
    ax.set_ylabel('Structural DRR (dB)', fontsize=11)
    ax.set_title('Exponential Decay Initialization Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=7, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Plot 6: Best results summary table
    ax = axes[1, 2]
    ax.axis('off')
    
    # Find best method for each RIR length
    best_results = []
    for i, rir_len in enumerate(rir_lengths):
        best_drr = -999
        best_method = ""
        
        for method, drr_values in drr_data.items():
            if i < len(drr_values) and drr_values[i] > best_drr:
                best_drr = drr_values[i]
                best_method = method.replace('_Enhanced-', '\n')
        
        best_results.append([
            f"{rir_lengths_ms[i]:.0f} ms",
            best_method,
            f"{best_drr:.1f} dB"
        ])
    
    table_data = [['RIR Length', 'Best Method', 'DRR']] + best_results
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.4, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight rows that meet target
    for i, row in enumerate(best_results, 1):
        drr_val = float(row[2].replace(' dB', ''))
        if drr_val >= 7:
            for j in range(3):
                table[(i, j)].set_facecolor('#c8e6c9')
    
    ax.set_title('Best Performance by RIR Length', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save figure
    output_path = "experiments/drr_enhanced_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot to {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    all_drr_values = []
    for method, values in drr_data.items():
        all_drr_values.extend(values)
        avg = np.mean(values)
        std = np.std(values)
        max_val = np.max(values)
        print(f"\n{method:40s}")
        print(f"  Average: {avg:6.2f} dB ± {std:.2f}")
        print(f"  Range:   {np.min(values):6.2f} to {max_val:6.2f} dB")
        if max_val >= 7:
            print(f"  ✓ EXCEEDS 7 dB TARGET!")
    
    overall_avg = np.mean(all_drr_values)
    success_count = sum(1 for v in all_drr_values if v >= 7)
    success_rate = 100 * success_count / len(all_drr_values)
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Average DRR: {overall_avg:.2f} dB")
    print(f"  Range: {np.min(all_drr_values):.2f} to {np.max(all_drr_values):.2f} dB")
    print(f"  Success rate (≥7 dB): {success_count}/{len(all_drr_values)} ({success_rate:.1f}%)")
    
    print("\n" + "=" * 80)
    
    plt.show()


if __name__ == '__main__':
    main()
