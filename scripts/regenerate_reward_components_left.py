#!/usr/bin/env python3
"""
Regenerate reward components - left side only (bar chart).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Professional publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11


def regenerate_reward_components_left_only():
    """
    Reward components - bar chart only (left side).
    """
    
    components = ['DRR', 'Direct-to-\nTail', 'Early-to-\nLate', 
                  'Tail\nDecay', 'Energy\nConserv.', 'Smooth-\nness']
    weights = [1.0, 0.3, 0.2, 0.5, 0.1, 0.1]
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728', '#9467bd', '#8c564b']
    
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    # Bar chart of weights
    bars = ax.bar(range(len(components)), weights, color=colors, 
                   alpha=0.85, edgecolor='black', linewidth=2.0)
    
    ax.set_xlabel('Reward Component', fontsize=14, fontweight='bold', labelpad=12)
    ax.set_ylabel('Weight', fontsize=14, fontweight='bold', labelpad=12)
    ax.set_title('Composite Reward Function Components', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(components, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25, axis='y', linestyle='--', zorder=0)
    ax.set_ylim(0, 1.25)
    
    # Add values on bars
    for bar, w in zip(bars, weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.04,
                f'{w:.1f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=13)
    
    # Add formula annotation
    formula_text = (
        r"$R = w_{\mathrm{DRR}} \cdot r_{\mathrm{DRR}} + "
        r"w_{\mathrm{D/T}} \cdot r_{\mathrm{D/T}} + "
        r"w_{\mathrm{E/L}} \cdot r_{\mathrm{E/L}} + \ldots$"
    )
    ax.text(0.5, 0.95, formula_text,
           transform=ax.transAxes,
           fontsize=11,
           ha='center',
           va='top',
           bbox=dict(boxstyle='round,pad=0.6',
                    facecolor='#f8f9fa',
                    edgecolor='#2ca02c',
                    alpha=0.95,
                    linewidth=2))
    
    # Cleaner spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.8)
    ax.spines['bottom'].set_linewidth(1.8)
    
    # Better tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.3, length=6)
    
    # Add subtle background for primary component
    ax.axhspan(0.8, 1.25, facecolor='#d4edda', alpha=0.12, zorder=0)
    
    plt.tight_layout()
    
    # Save to both locations
    output_path1 = Path(__file__).parent.parent / 'docs' / 'figures' / 'reward_components.png'
    output_path2 = Path(__file__).parent.parent / 'experiments' / 'reward_components.png'
    
    plt.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✓ Regenerated (left side only): reward_components.png")
    print(f"  Saved to: docs/figures/")
    print(f"  Saved to: experiments/")
    
    plt.close()


def main():
    print("=" * 60)
    print("REGENERATING REWARD COMPONENTS - LEFT SIDE ONLY")
    print("=" * 60)
    print()
    
    regenerate_reward_components_left_only()
    
    print()
    print("=" * 60)
    print("✅ FIGURE REGENERATED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("Improvements:")
    print("  - Single panel (bar chart only)")
    print("  - Larger, clearer presentation")
    print("  - Reward formula annotation")
    print("  - Professional styling")
    print("  - 300 DPI publication quality")


if __name__ == "__main__":
    main()
