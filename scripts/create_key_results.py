"""
Create a focused results summary with the two key plots:
1. Final learned RIRs for all methods
2. DRR improvement comparison
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def create_summary_figure():
    """Create a 2-panel figure with final RIRs and DRR results."""
    results_dir = Path('experiments/all_combinations')
    
    # Load results
    with open(results_dir / 'all_results.pkl', 'rb') as f:
        all_results = pickle.load(f)
    
    with open(results_dir / 'drr_summary.json', 'r') as f:
        drr_summary = json.load(f)
    
    # Create figure with 2 main panels
    fig = plt.figure(figsize=(20, 12))
    
    # Main title
    fig.suptitle('Reinforcement Learning for Room Impulse Response Estimation:\nKey Results', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    colors = {
        'QN-random': '#1f77b4',
        'QN-exponential_decay': '#aec7e8',
        'DQN-random': '#ff7f0e',
        'DQN-exponential_decay': '#ffbb78',
        'Neural-random': '#2ca02c',
        'Neural-exponential_decay': '#98df8a'
    }
    
    # ========== Panel 1: Final Learned RIRs (Top) ==========
    gs_top = fig.add_gridspec(3, 2, left=0.08, right=0.95, top=0.90, bottom=0.52, 
                              hspace=0.35, wspace=0.25)
    
    fig.text(0.5, 0.92, 'A. Final Learned Room Impulse Responses', 
             fontsize=16, fontweight='bold', ha='center')
    
    for idx, result in enumerate(all_results):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs_top[row, col])
        
        label = f"{result['agent_type']}-{result['init_method']}"
        final_rir = result['final_rir']
        
        # Plot RIR
        time_ms = np.arange(len(final_rir)) / 16.0  # Convert samples to ms (16 kHz)
        ax.plot(time_ms, final_rir, color=colors.get(label, 'gray'), linewidth=1.5, alpha=0.8)
        
        # Add metrics
        final_metrics = result['final_metrics']
        corr = final_metrics['avg_correlation']
        reward = final_metrics['avg_reward']
        
        title_text = f"{result['agent_type']} + {result['init_method'].replace('_', ' ').title()}"
        ax.set_title(title_text, fontweight='bold', fontsize=13)
        
        # Add text box with metrics
        textstr = f"Correlation: {corr:.3f}\nReward: {reward:.2f}"
        props = dict(boxstyle='round', facecolor=colors.get(label, 'gray'), alpha=0.15)
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        ax.set_xlabel('Time (ms)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Amplitude', fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # ========== Panel 2: DRR Comparison (Bottom) ==========
    gs_bottom = fig.add_gridspec(1, 2, left=0.08, right=0.95, top=0.43, bottom=0.08,
                                  hspace=0.3, wspace=0.35)
    
    fig.text(0.5, 0.45, 'B. Dereverberation Performance (DRR Improvement)', 
             fontsize=16, fontweight='bold', ha='center')
    
    # DRR Bar Chart
    ax_drr = fig.add_subplot(gs_bottom[0, 0])
    
    initial_drr = drr_summary['initial_drr']
    clean_drr = drr_summary['clean_reference_drr']
    
    labels = ['Reverberant\nInput'] + [f"{m['agent_type']}\n{m['init_method'][:3].upper()}" 
                                        for m in drr_summary['methods']]
    drr_values = [initial_drr] + [m['final_drr'] for m in drr_summary['methods']]
    colors_list = ['red'] + [colors.get(f"{m['agent_type']}-{m['init_method']}", 'gray') 
                             for m in drr_summary['methods']]
    
    bars = ax_drr.bar(range(len(labels)), drr_values, color=colors_list, alpha=0.75, 
                      edgecolor='black', linewidth=1.5)
    
    # Add reference lines
    ax_drr.axhline(y=initial_drr, color='red', linestyle='--', linewidth=2.5, 
                   alpha=0.6, label=f'Input: {initial_drr:.1f} dB', zorder=1)
    ax_drr.axhline(y=clean_drr, color='green', linestyle='--', linewidth=2.5, 
                   alpha=0.6, label=f'Clean Target: {clean_drr:.1f} dB', zorder=1)
    
    ax_drr.set_xticks(range(len(labels)))
    ax_drr.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax_drr.set_ylabel('DRR (dB)', fontweight='bold', fontsize=13)
    ax_drr.set_title('Direct-to-Reverberant Ratio after Dereverberation', 
                     fontweight='bold', fontsize=13)
    ax_drr.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax_drr.grid(True, axis='y', alpha=0.3, zorder=0)
    ax_drr.set_ylim([0, max(drr_values) * 1.3])
    
    # Add value labels
    for bar, val in zip(bars, drr_values):
        height = bar.get_height()
        ax_drr.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # DRR Gain Chart
    ax_gain = fig.add_subplot(gs_bottom[0, 1])
    
    gain_labels = [f"{m['agent_type']}\n{m['init_method'][:3].upper()}" 
                   for m in drr_summary['methods']]
    gain_values = [m['drr_gain'] for m in drr_summary['methods']]
    gain_colors = [colors.get(f"{m['agent_type']}-{m['init_method']}", 'gray') 
                   for m in drr_summary['methods']]
    
    bars_gain = ax_gain.bar(range(len(gain_labels)), gain_values, color=gain_colors, 
                            alpha=0.75, edgecolor='black', linewidth=1.5)
    
    ax_gain.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.8, zorder=1)
    ax_gain.set_xticks(range(len(gain_labels)))
    ax_gain.set_xticklabels(gain_labels, fontsize=11, fontweight='bold')
    ax_gain.set_ylabel('DRR Gain (dB)', fontweight='bold', fontsize=13)
    ax_gain.set_title('DRR Improvement Relative to Input', fontweight='bold', fontsize=13)
    ax_gain.grid(True, axis='y', alpha=0.3, zorder=0)
    
    # Color bars based on positive/negative
    for bar, val in zip(bars_gain, gain_values):
        if val < 0:
            bar.set_facecolor('lightcoral')
            bar.set_alpha(0.7)
        height = bar.get_height()
        va = 'bottom' if val >= 0 else 'top'
        y_offset = 0.05 if val >= 0 else -0.05
        ax_gain.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                    f'{val:+.2f}', ha='center', va=va, fontsize=10, fontweight='bold')
    
    # Add summary statistics box
    best_idx = np.argmax([m['final_drr'] for m in drr_summary['methods']])
    best_method = drr_summary['methods'][best_idx]
    best_corr_idx = np.argmax([m['rir_correlation'] for m in drr_summary['methods']])
    best_corr_method = drr_summary['methods'][best_corr_idx]
    
    summary_text = (
        f"Best DRR: {best_method['agent_type']} + {best_method['init_method']}\n"
        f"  → {best_method['final_drr']:.2f} dB ({best_method['drr_gain']:+.2f} dB)\n\n"
        f"Best RIR Correlation: {best_corr_method['agent_type']} + {best_corr_method['init_method']}\n"
        f"  → {best_corr_method['rir_correlation']:.3f}"
    )
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2)
    ax_gain.text(0.97, 0.03, summary_text, transform=ax_gain.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right', bbox=props,
                family='monospace')
    
    # Save figure
    plt.savefig(results_dir / 'key_results_summary.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved key results summary: {results_dir / 'key_results_summary.png'}")
    plt.close()
    
    # Print summary
    print("\n" + "="*80)
    print("KEY RESULTS SUMMARY")
    print("="*80)
    print(f"\nInitial Reverberant Speech DRR: {initial_drr:.2f} dB")
    print(f"Clean Reference Speech DRR: {clean_drr:.2f} dB")
    print(f"\nBest Dereverberation Performance:")
    print(f"  Method: {best_method['agent_type']} + {best_method['init_method']}")
    print(f"  Final DRR: {best_method['final_drr']:.2f} dB (Δ={best_method['drr_gain']:+.2f} dB)")
    print(f"\nBest RIR Estimation:")
    print(f"  Method: {best_corr_method['agent_type']} + {best_corr_method['init_method']}")
    print(f"  RIR Correlation: {best_corr_method['rir_correlation']:.3f}")
    print(f"  Final DRR: {best_corr_method['final_drr']:.2f} dB")
    print("="*80)


if __name__ == '__main__':
    create_summary_figure()
