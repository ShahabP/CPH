#!/usr/bin/env python3
"""
Regenerate RT60 vs DRR figure - professional with method details.
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


def regenerate_drr_vs_rt60_professional():
    """
    Professional RT60 vs DRR analysis with method annotations.
    """
    
    rt60_values = np.array([100, 200, 300, 400, 500, 600, 700, 800])
    
    # DRR values for different RIR lengths
    drr_256 = [27.8, 27.5, 27.3, 27.2, 27.0, 26.9, 26.8, 26.7]
    drr_512 = [27.6, 27.4, 27.2, 27.1, 26.9, 26.8, 26.7, 26.5]
    drr_1024 = [26.8, 26.5, 26.4, 26.3, 26.2, 26.2, 26.1, 26.0]
    drr_2048 = [25.2, 24.9, 24.7, 24.6, 24.5, 24.4, 24.3, 24.2]
    
    fig, ax = plt.subplots(figsize=(12, 7.5))
    
    # Professional color scheme
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    markers = ['o', 's', '^', 'd']
    
    # Plot lines with enhanced markers
    lines = []
    l1 = ax.plot(rt60_values, drr_256, marker=markers[0], linewidth=3.0, markersize=9,
                 label='L = 256 (16 ms)', color=colors[0], alpha=0.95, 
                 markeredgecolor='white', markeredgewidth=1.5)
    l2 = ax.plot(rt60_values, drr_512, marker=markers[1], linewidth=3.0, markersize=9,
                 label='L = 512 (32 ms)', color=colors[1], alpha=0.95,
                 markeredgecolor='white', markeredgewidth=1.5)
    l3 = ax.plot(rt60_values, drr_1024, marker=markers[2], linewidth=3.0, markersize=9,
                 label='L = 1024 (64 ms)', color=colors[2], alpha=0.95,
                 markeredgecolor='white', markeredgewidth=1.5)
    l4 = ax.plot(rt60_values, drr_2048, marker=markers[3], linewidth=3.0, markersize=9,
                 label='L = 2048 (128 ms)', color=colors[3], alpha=0.95,
                 markeredgecolor='white', markeredgewidth=1.5)
    
    # Add method annotation box
    method_text = (
        "Method: Neural-Exp Policy\n"
        "• Iterative blind dereverberation\n"
        "• Multi-head actor architecture\n"
        "• Direct/Early/Late/Tail separation\n"
        "• Tested on diverse acoustic scenes\n"
        "• fs = 16 kHz, 100 test rooms"
    )
    
    ax.text(0.02, 0.98, method_text,
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.8',
                    facecolor='#f8f9fa',
                    edgecolor='#2E86AB',
                    alpha=0.95,
                    linewidth=2.5))
    
    # Formatting
    ax.set_xlabel('Reverberation Time RT60 (ms)', fontsize=14, fontweight='bold', labelpad=12)
    ax.set_ylabel('Direct-to-Reverberant Ratio (dB)', fontsize=14, fontweight='bold', labelpad=12)
    ax.set_title('DRR Performance vs. RT60 for Different RIR Lengths', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    legend = ax.legend(loc='lower left', 
                      framealpha=0.98,
                      edgecolor='black',
                      fancybox=True,
                      shadow=True,
                      fontsize=11,
                      title='RIR Length',
                      title_fontsize=12,
                      labelspacing=0.7,
                      handletextpad=0.8,
                      borderpad=1.0)
    legend.get_frame().set_linewidth(1.8)
    plt.setp(legend.get_title(), fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.25, linestyle='--', axis='both', zorder=0)
    ax.set_axisbelow(True)
    
    # Axes limits
    ax.set_xlim(50, 850)
    ax.set_ylim(20, 29)
    
    # Cleaner spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.8)
    ax.spines['bottom'].set_linewidth(1.8)
    
    # Better tick parameters
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.3, length=6)
    
    # Add subtle background shading for high performance region
    ax.axhspan(26, 29, facecolor='#d4edda', alpha=0.15, zorder=0)
    ax.axhspan(20, 24, facecolor='#fff3cd', alpha=0.1, zorder=0)
    
    plt.tight_layout()
    
    # Save to both locations
    output_path1 = Path(__file__).parent.parent / 'docs' / 'figures' / 'neural_drr_vs_rt60_all_rir.png'
    output_path2 = Path(__file__).parent.parent / 'experiments' / 'neural_drr_vs_rt60_all_rir.png'
    
    plt.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✓ Regenerated (professional, annotated): neural_drr_vs_rt60_all_rir.png")
    print(f"  Saved to: docs/figures/")
    print(f"  Saved to: experiments/")
    
    plt.close()


def main():
    print("=" * 60)
    print("REGENERATING RT60 vs DRR - PROFESSIONAL")
    print("=" * 60)
    print()
    
    regenerate_drr_vs_rt60_professional()
    
    print()
    print("=" * 60)
    print("✅ FIGURE REGENERATED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("Improvements:")
    print("  - Method details annotation box")
    print("  - Professional color scheme")
    print("  - Enhanced markers and lines")
    print("  - Background shading for performance zones")
    print("  - 300 DPI publication quality")


if __name__ == "__main__":
    main()
