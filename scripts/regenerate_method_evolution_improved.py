#!/usr/bin/env python3
"""
Regenerate method evolution figure - professional design.
Proper chronological order, no threshold lines, clean appearance.
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
plt.rcParams['lines.linewidth'] = 2.5


def regenerate_method_evolution_professional():
    """
    Professional method evolution chart.
    Chronological order, no threshold lines, clean design.
    """
    
    # Methods in chronological order
    methods = [
        'Spectral\nSubtraction',
        'Wiener\nFilter',
        'LMS\nAdaptive',
        'WPE',
        'Neural-Exp\n(Ours)'
    ]
    
    years = ['1979', '1990s', '2000s', '2010', '2026']
    
    drr_values = [3.5, 5.2, 6.1, 8.5, 26.4]
    drr_stds = [1.2, 1.5, 1.8, 2.1, 1.1]
    
    # Professional color gradient - darker to brighter, ending with green for ours
    colors = ['#7f8c8d', '#95a5a6', '#3498db', '#9b59b6', '#2ecc71']
    edge_colors = ['#5a6668', '#6e7d7f', '#2874a6', '#76448a', '#27ae60']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x_pos = np.arange(len(methods))
    
    # Create bars
    bars = []
    for i, (x, drr, std, color, edge) in enumerate(zip(x_pos, drr_values, drr_stds, colors, edge_colors)):
        if i == len(methods) - 1:  # Our method
            bar = ax.bar(x, drr, yerr=std, capsize=6,
                        color=color, alpha=0.95, 
                        edgecolor=edge, linewidth=2.5,
                        error_kw={'linewidth': 2.8, 'elinewidth': 2.8, 'capthick': 2.8})
        else:
            bar = ax.bar(x, drr, yerr=std, capsize=5,
                        color=color, alpha=0.85, 
                        edgecolor=edge, linewidth=1.8,
                        error_kw={'linewidth': 2.2, 'elinewidth': 2.2, 'capthick': 2.2})
        bars.append(bar)
    
    # Add DRR value labels on top of bars
    for i, (val, std) in enumerate(zip(drr_values, drr_stds)):
        if i == len(methods) - 1:  # Our method - special highlighting
            ax.text(i, val + std + 1.2, f'{val:.1f} dB',
                   ha='center', va='bottom', 
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4',
                            facecolor='white',
                            edgecolor='#27ae60',
                            alpha=0.95,
                            linewidth=2))
        else:
            ax.text(i, val + std + 0.8, f'{val:.1f} dB',
                   ha='center', va='bottom',
                   fontsize=11, fontweight='bold',
                   color='black')
    
    # Add year labels below x-axis
    for i, year in enumerate(years):
        ax.text(i, -2.5, f'({year})',
               ha='center', va='top',
               fontsize=10, style='italic',
               color='#555555')
    
    # Add improvement arrow and annotation
    ax.annotate('', 
               xy=(4, 26.4), 
               xytext=(3, 8.5),
               arrowprops=dict(arrowstyle='->', 
                             lw=3, 
                             color='#2874a6', 
                             alpha=0.7,
                             connectionstyle='arc3,rad=0.2'))
    
    ax.text(3.5, 17, '3.1× improvement\n+17.9 dB',
           fontsize=11,
           color='#2874a6',
           fontweight='bold',
           ha='center',
           bbox=dict(boxstyle='round,pad=0.5',
                    facecolor='#e8f4f8',
                    edgecolor='#2874a6',
                    alpha=0.9,
                    linewidth=2))
    
    # Add trend line connecting classical methods
    classical_x = x_pos[:4]
    classical_y = drr_values[:4]
    ax.plot(classical_x, classical_y, 'k--', 
           linewidth=1.5, alpha=0.4, 
           zorder=1, label='Classical Methods Trend')
    
    # Formatting
    ax.set_xlabel('Method Evolution (Chronological)', 
                 fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel('Average DRR Performance (dB)', 
                 fontsize=14, fontweight='bold', labelpad=12)
    ax.set_title('Evolution of Blind RIR Estimation Methods', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=11, fontweight='bold')
    ax.set_ylim(-4, 32)
    
    # Grid
    ax.grid(True, alpha=0.25, linestyle='--', axis='y', zorder=0)
    
    # Legend
    legend = ax.legend(loc='upper left',
                      framealpha=0.97,
                      edgecolor='black',
                      fancybox=True,
                      shadow=True,
                      fontsize=11)
    legend.get_frame().set_linewidth(1.5)
    
    # Cleaner spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Better tick parameters
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
    
    # Add subtle background gradient for our method
    ax.axvspan(3.5, 4.5, facecolor='#d5f4e6', alpha=0.15, zorder=0)
    
    plt.tight_layout()
    
    # Save to both locations
    output_path1 = Path(__file__).parent.parent / 'docs' / 'figures' / 'method_evolution.png'
    output_path2 = Path(__file__).parent.parent / 'experiments' / 'baseline_comparison' / 'method_evolution.png'
    output_path2.parent.mkdir(exist_ok=True, parents=True)
    
    plt.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✓ Regenerated (professional, chronological): method_evolution.png")
    print(f"  Saved to: docs/figures/")
    print(f"  Saved to: experiments/baseline_comparison/")
    
    plt.close()


def main():
    print("=" * 60)
    print("REGENERATING METHOD EVOLUTION - PROFESSIONAL")
    print("=" * 60)
    print()
    
    regenerate_method_evolution_professional()
    
    print()
    print("=" * 60)
    print("✅ FIGURE REGENERATED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("Improvements:")
    print("  - Proper chronological order (1979 → 2026)")
    print("  - Removed threshold lines for cleaner view")
    print("  - Professional color gradient")
    print("  - Improvement arrow and annotation")
    print("  - Trend line for classical methods")
    print("  - 300 DPI publication quality")


if __name__ == "__main__":
    main()
