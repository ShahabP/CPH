#!/usr/bin/env python3
"""
Regenerate room dimensions comparison with box/violin plots.
Using rounded box plots like the original version.
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
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['lines.linewidth'] = 2


def regenerate_room_dimensions_boxplot():
    """
    Professional room dimensions comparison with box/violin plots.
    Rounded visualization like the original.
    """
    
    rooms = ['Small\nOffice', 'Medium\nRoom', 'Large\nRoom', 'Concert\nHall', 'Cathedral']
    volumes = [39, 90, 320, 1080, 5000]
    rt60s = [200, 350, 500, 700, 1200]
    # Updated DRR means (user requested decreases per room)
    # Decreases: Small Office -5 dB, Medium Room -6 dB, Large Room -9 dB,
    # Concert Hall -12 dB, Cathedral -11 dB
    drr_means = [12.50, 11.15, 7.66, 4.03, 4.05]
    drr_stds = [0.6, 0.5, 0.4, 0.6, 0.8]
    
    # Generate synthetic data for box plots based on mean/std
    np.random.seed(42)
    n_samples = 50  # Number of synthetic samples per room
    
    drr_data = []
    for mean, std in zip(drr_means, drr_stds):
        samples = np.random.normal(mean, std, n_samples)
        drr_data.append(samples)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Colors for violin/box plots - professional gradient
    colors = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087']
    
    # Create violin plots (rounded box plots)
    positions = np.arange(len(rooms))
    
    parts = ax.violinplot(drr_data, 
                         positions=positions,
                         widths=0.7,
                         showmeans=True,
                         showextrema=True,
                         showmedians=True)
    
    # Customize violin plot colors and edges
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_linewidth(1.2)
        pc.set_alpha(0.95)

    # Tidy up whiskers/medians
    parts['cbars'].set_edgecolor('black')
    parts['cbars'].set_linewidth(1.2)
    parts['cmaxes'].set_edgecolor('black')
    parts['cmaxes'].set_linewidth(1.2)
    parts['cmins'].set_edgecolor('black')
    parts['cmins'].set_linewidth(1.2)
    if 'cmedians' in parts:
        parts['cmedians'].set_edgecolor('#c62828')
        parts['cmedians'].set_linewidth(2.4)

    # Draw bold red mean bars centered on each violin (like the example)
    mean_bar_half = 0.28
    for i, (pos, mean) in enumerate(zip(positions, drr_means)):
        ax.hlines(mean, pos - mean_bar_half, pos + mean_bar_half,
              colors='#c62828', linewidth=3.2, zorder=5)

    # Mean value labels removed per user request (no boxed dB values on figure)
    
    # Add room specifications below
    for i, (pos, vol, rt) in enumerate(zip(positions, volumes, rt60s)):
        # Volume
        ax.text(pos, 19.5,
                f'V = {vol:,} m³',
                ha='center', va='center',
                fontsize=12,
                style='italic',
                bbox=dict(boxstyle='round,pad=0.3',
                         facecolor='#f5f5f5',
                         edgecolor='gray',
                         alpha=0.9,
                         linewidth=1))
        # RT60
        ax.text(pos, 18.0,
                f'RT60 = {rt} ms',
                ha='center', va='center',
                fontsize=12,
                style='italic',
                bbox=dict(boxstyle='round,pad=0.3',
                         facecolor='#f5f5f5',
                         edgecolor='gray',
                         alpha=0.9,
                         linewidth=1))
    
    
    
    # Formatting
    ax.set_xlabel('Room Type', fontsize=14, fontweight='bold', labelpad=12)
    ax.set_ylabel('Direct-to-Reverberant Ratio (dB)', fontsize=14, fontweight='bold', labelpad=12)
    ax.set_title('DRR Performance Across Diverse Room Geometries',
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(rooms, fontsize=14, fontweight='bold')
    
    # Legend
    legend = ax.legend(loc='upper right', framealpha=0.97,
                      fancybox=True, shadow=False, fontsize=14)
    legend.get_frame().set_linewidth(0)
    
    # Adjust y-limits to accommodate lowered DRR values
    ax.set_ylim(0, 14)
    ax.grid(True, alpha=0.25, linestyle='--', axis='y')
    
    # Cleaner spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
  
    
    plt.tight_layout()
    
    # Save to both locations
    output_path1 = Path(__file__).parent.parent / 'docs' / 'figures' / 'room_dimensions_comparison.png'
    output_path2 = Path(__file__).parent.parent / 'experiments' / 'room_dimensions_comparison.png'
    
    plt.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✓ Regenerated (box/violin plots): room_dimensions_comparison.png")
    print(f"  Saved to: docs/figures/")
    print(f"  Saved to: experiments/")
    
    plt.close()


def main():
    print("=" * 60)
    print("REGENERATING ROOM DIMENSIONS - BOX/VIOLIN PLOTS")
    print("=" * 60)
    print()
    
    regenerate_room_dimensions_boxplot()
    
    print()
    print("=" * 60)
    print("✅ FIGURE REGENERATED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("Style: Rounded box/violin plots")
    print("Quality: 300 DPI, publication-ready")


if __name__ == "__main__":
    main()
