#!/usr/bin/env python3
"""
Plot the final estimated RIRs from Neural agent training.
Compares random vs exponential decay initialization across all RIR lengths.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_neural_results(rir_length):
    """Load Neural agent results for a given RIR length."""
    base_dir = Path("experiments/rir_length_enhanced")
    pkl_path = base_dir / f"rir_{rir_length}" / "all_results.pkl"
    
    if not pkl_path.exists():
        print(f"Warning: {pkl_path} not found")
        return None, None
    
    with open(pkl_path, 'rb') as f:
        all_results = pickle.load(f)
    
    neural_random = None
    neural_exp = None
    
    for result in all_results:
        if result['agent_type'] == 'Neural_Enhanced':
            if result['init_method'] == 'random':
                neural_random = result
            elif result['init_method'] == 'exponential_decay':
                neural_exp = result
    
    return neural_random, neural_exp


def plot_rir_comparison(ax, rir_random, rir_exp, rir_length, sample_rate=16000):
    """Plot RIR comparison on given axis."""
    time_ms = np.arange(rir_length) / sample_rate * 1000
    
    # Plot exponential decay (the successful one)
    if rir_exp is not None:
        ax.plot(time_ms, rir_exp, '-', linewidth=1.5, color='#d62728', 
               label='Exp Decay Init', alpha=0.8)
    
    # Plot random initialization
    if rir_random is not None:
        ax.plot(time_ms, rir_random, '-', linewidth=1.5, color='#1f77b4', 
               label='Random Init', alpha=0.6)
    
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Add zero line
    ax.axhline(y=0, color='k', linestyle=':', linewidth=0.5, alpha=0.5)


def compute_drr(rir, sample_rate=16000):
    """Compute DRR from RIR."""
    direct_window_samples = int(0.0025 * sample_rate)
    peak_idx = np.argmax(np.abs(rir))
    
    direct_start = peak_idx
    direct_end = min(peak_idx + direct_window_samples, len(rir))
    reverb_start = direct_end
    
    direct_energy = np.sum(rir[direct_start:direct_end]**2)
    reverb_energy = np.sum(rir[reverb_start:]**2)
    
    if reverb_energy < 1e-10:
        return 100.0
    if direct_energy < 1e-10:
        return -100.0
    
    return 10 * np.log10(direct_energy / reverb_energy)


def main():
    print("\n" + "="*80)
    print("PLOTTING FINAL RIRs FROM NEURAL AGENT")
    print("="*80)
    
    rir_lengths = [256, 512, 1024, 2048]
    rir_lengths_ms = [rl / 16.0 for rl in rir_lengths]
    sample_rate = 16000
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Main title
    fig.suptitle('Neural Agent: Final Estimated Room Impulse Responses\n' + 
                 '(After 1200 Training Episodes)', 
                 fontsize=16, fontweight='bold')
    
    # Create grid: 2 rows x 4 columns
    # Top row: Time domain plots
    # Bottom row: Zoom on early reflections
    
    for idx, (rir_len, rir_ms) in enumerate(zip(rir_lengths, rir_lengths_ms)):
        # Load results
        neural_random, neural_exp = load_neural_results(rir_len)
        
        if neural_random is None and neural_exp is None:
            continue
        
        rir_random = neural_random['final_rir'] if neural_random else None
        rir_exp = neural_exp['final_rir'] if neural_exp else None
        
        drr_random = compute_drr(rir_random) if rir_random is not None else 0.0
        drr_exp = compute_drr(rir_exp) if rir_exp is not None else 0.0
        
        # Top row: Full RIR
        ax1 = plt.subplot(2, 4, idx + 1)
        plot_rir_comparison(ax1, rir_random, rir_exp, rir_len, sample_rate)
        ax1.set_title(f'RIR Length: {rir_len} samples ({rir_ms:.0f} ms)\n' + 
                     f'DRR: Rand={drr_random:.1f} dB, Exp={drr_exp:.1f} dB', 
                     fontsize=11, fontweight='bold')
        
        # Bottom row: Zoomed view (first 50ms or first quarter, whichever is smaller)
        ax2 = plt.subplot(2, 4, idx + 5)
        zoom_samples = min(int(0.05 * sample_rate), rir_len // 4)
        zoom_time_ms = np.arange(zoom_samples) / sample_rate * 1000
        
        if rir_exp is not None:
            ax2.plot(zoom_time_ms, rir_exp[:zoom_samples], '-', linewidth=2, 
                    color='#d62728', label='Exp Decay Init', alpha=0.8)
        
        if rir_random is not None:
            ax2.plot(zoom_time_ms, rir_random[:zoom_samples], '-', linewidth=2, 
                    color='#1f77b4', label='Random Init', alpha=0.6)
        
        ax2.set_xlabel('Time (ms)', fontsize=10)
        ax2.set_ylabel('Amplitude', fontsize=10)
        ax2.set_title(f'Early Reflections - {rir_len} samples\n(first {zoom_samples} samples shown)', 
                     fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        ax2.axhline(y=0, color='k', linestyle=':', linewidth=0.5, alpha=0.5)
        
        # Highlight the peak
        if rir_exp is not None:
            peak_idx = np.argmax(np.abs(rir_exp[:zoom_samples]))
            peak_time = peak_idx / sample_rate * 1000
            ax2.axvline(x=peak_time, color='r', linestyle='--', 
                       linewidth=1, alpha=0.5, label='Peak')
    
    plt.tight_layout()
    
    # Save figure
    output_path = "experiments/neural_final_rirs.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved RIR plots to {output_path}")
    
    # Print RIR statistics
    print("\n" + "="*80)
    print("RIR STATISTICS")
    print("="*80)
    
    for rir_len, rir_ms in zip(rir_lengths, rir_lengths_ms):
        neural_random, neural_exp = load_neural_results(rir_len)
        
        if neural_random is None and neural_exp is None:
            continue
        
        print(f"\nRIR Length: {rir_len} samples ({rir_ms:.0f} ms)")
        print("-" * 60)
        
        for name, result in [('Random Init', neural_random), 
                             ('Exp Decay Init', neural_exp)]:
            if result:
                rir = result['final_rir']
                drr = compute_drr(rir)
                peak_idx = np.argmax(np.abs(rir))
                peak_val = rir[peak_idx]
                rms = np.sqrt(np.mean(rir**2))
                energy = np.sum(rir**2)
                
                print(f"  {name:20s}:")
                print(f"    Peak: {peak_val:8.4f} at sample {peak_idx:4d} " + 
                      f"({peak_idx/sample_rate*1000:.2f} ms)")
                print(f"    RMS:  {rms:8.4f}")
                print(f"    Energy: {energy:8.4f}")
                print(f"    DRR:  {drr:8.2f} dB")
    
    print("\n" + "="*80)
    
    plt.show()


if __name__ == '__main__':
    main()
