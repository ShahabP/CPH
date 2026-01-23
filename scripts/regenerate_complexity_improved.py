#!/usr/bin/env python3
"""
Regenerate complexity vs performance figure - improved visibility.
Remove threshold line, enhance clarity.
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
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['axes.axisbelow'] = True


def regenerate_complexity_vs_performance():
    """
    Improved complexity vs performance scatter plot.
    Using computational time (seconds) like the original graph.
    Clear, professional design without threshold line.
    """
    
    # Method data: [Computational Time (sec), DRR (dB)]
    # Time is for processing a 2.5s reverberant signal
    methods = {
        'Ours': {
            'comp_time': 2.5,
            'drr': 26.4,
            'color': '#2ca02c',
            'marker': 'o',
            'size': 400,
            'edgecolor': '#1a661a',
            'edgewidth': 2.5
        },
        'WPE': {
            'comp_time': 1.5,
            'drr': 8.5,
            'color': '#bebada',
            'marker': 'D',
            'size': 280,
            'edgecolor': '#8b7cb5',
            'edgewidth': 2
        },
        'LMS': {
            'comp_time': 0.8,
            'drr': 6.1,
            'color': '#fb8072',
            'marker': 's',
            'size': 260,
            'edgecolor': '#c95d51',
            'edgewidth': 2
        },
        'WF': {
            'comp_time': 0.12,
            'drr': 5.2,
            'color': '#ffffb3',
            'marker': '^',
            'size': 240,
            'edgecolor': '#cccc8f',
            'edgewidth': 2
        },
        'SS': {
            'comp_time': 0.05,
            'drr': 3.5,
            'color': '#8dd3c7',
            'marker': 'v',
            'size': 220,
            'edgecolor': '#5fa99a',
            'edgewidth': 2
        }
    }
    
    fig, ax = plt.subplots(figsize=(11, 8))
    
    # Plot each method
    for method_name, data in methods.items():
        ax.scatter(data['comp_time'], data['drr'],
                  s=data['size'],
                  c=data['color'],
                  marker=data['marker'],
                  alpha=0.8,
                  edgecolors=data['edgecolor'],
                  linewidths=data['edgewidth'],
                  label=method_name,
                  zorder=10 if method_name == 'Ours' else 5)
        
        # Add method labels and values
        if method_name == 'Ours':
            # Special annotation for our method
            ax.annotate(f'{method_name}\n({data["drr"]:.1f} dB, {data["comp_time"]:.1f}s)',
                       xy=(data['comp_time'], data['drr']),
                       xytext=(data['comp_time'] - 0.4, data['drr'] + 2.5),
                       fontsize=12,
                       fontweight='bold',
                       color='darkgreen',
                       bbox=dict(boxstyle='round,pad=0.5',
                                facecolor='lightgreen',
                                edgecolor='#1a661a',
                                alpha=0.8,
                                linewidth=2),
                       arrowprops=dict(arrowstyle='->',
                                     lw=2.5,
                                     color='green',
                                     alpha=0.8),
                       zorder=15)
        else:
            # Simple labels for other methods
            offset_x = 0.08 if method_name != 'SS' else 0.05
            offset_y = 0.8
            ax.text(data['comp_time'] + offset_x, data['drr'] + offset_y,
                   method_name,
                   fontsize=11,
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white',
                            edgecolor=data['edgecolor'],
                            alpha=0.85,
                            linewidth=1.2))
    
    # Pareto front line (connecting best trade-off points)
    pareto_x = [0.05, 2.5]
    pareto_y = [3.5, 26.4]
    ax.plot(pareto_x, pareto_y, 'b:', linewidth=2.5, alpha=0.5, zorder=3)
    
    # Trade-off annotation
    ax.text(1.2, 15, 'Performance-\nComplexity\nTrade-off',
           fontsize=12,
           color='blue',
           style='italic',
           fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5',
                    facecolor='lightyellow',
                    edgecolor='blue',
                    alpha=0.6,
                    linewidth=1.5))
    
    # Formatting
    ax.set_xlabel('Computational Time (seconds per 2.5s signal)',
                 fontsize=14, fontweight='bold', labelpad=12)
    ax.set_ylabel('DRR Performance (dB)',
                 fontsize=14, fontweight='bold', labelpad=12)
    ax.set_title('Computational Complexity vs Performance Trade-off',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits
    ax.set_xlim(-0.15, 3.0)
    ax.set_ylim(0, 30)
    
    # Legend with better positioning - moved to lower right to avoid overlap
    legend = ax.legend(loc='lower right',
                      framealpha=0.95,
                      edgecolor='black',
                      fancybox=True,
                      shadow=True,
                      ncol=1,
                      fontsize=12,
                      markerscale=0.8,
                      labelspacing=0.8,
                      handletextpad=0.8,
                      borderpad=1.2,
                      title='Methods',
                      title_fontsize=13)
    legend.get_frame().set_linewidth(1.5)
    # Make legend title bold
    plt.setp(legend.get_title(), fontweight='bold')
    
    # Cleaner spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Better tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    
    # Grid
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=1)
    
    plt.tight_layout()
    
    # Save to both locations
    output_path1 = Path(__file__).parent.parent / 'docs' / 'figures' / 'complexity_vs_performance.png'
    output_path2 = Path(__file__).parent.parent / 'experiments' / 'baseline_comparison' / 'complexity_vs_performance.png'
    output_path2.parent.mkdir(exist_ok=True, parents=True)
    
    plt.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✓ Regenerated (improved): complexity_vs_performance.png")
    print(f"  Saved to: docs/figures/")
    print(f"  Saved to: experiments/baseline_comparison/")
    
    plt.close()


def main():
    print("=" * 60)
    print("REGENERATING COMPLEXITY VS PERFORMANCE - IMPROVED")
    print("=" * 60)
    print()
    
    regenerate_complexity_vs_performance()
    
    print()
    print("=" * 60)
    print("✅ FIGURE REGENERATED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("Improvements:")
    print("  - Removed threshold line for cleaner view")
    print("  - Larger, clearer markers and labels")
    print("  - Better color scheme and contrast")
    print("  - Pareto frontier annotation")
    print("  - Professional publication quality (300 DPI)")


if __name__ == "__main__":
    main()
