#!/usr/bin/env python3
"""
Analyze and visualize RIR structure for natural acoustic properties.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_rir(rir_path):
    """Load RIR from numpy file."""
    return np.load(rir_path)

def analyze_rir_structure(rir, sample_rate=16000):
    """Analyze RIR for acoustic realism metrics."""
    metrics = {}
    
    # Basic properties
    metrics['length'] = len(rir)
    metrics['peak_idx'] = np.argmax(np.abs(rir))
    metrics['peak_value'] = np.abs(rir[metrics['peak_idx']])
    
    # Time-based analysis (assuming 16kHz sample rate)
    metrics['peak_time_ms'] = metrics['peak_idx'] / sample_rate * 1000
    
    # Energy distribution analysis
    total_energy = np.sum(rir ** 2)
    
    # Direct sound (first 5 samples ~0.3ms)
    direct_energy = rir[0] ** 2
    metrics['direct_energy_pct'] = (direct_energy / total_energy) * 100
    
    # Early reflections (0.3-50ms)
    early_end = min(int(0.05 * sample_rate), len(rir))  # 50ms
    early_energy = np.sum(rir[1:early_end] ** 2)
    metrics['early_energy_pct'] = (early_energy / total_energy) * 100
    
    # Late reverberation (50-200ms)  
    late_start = early_end
    late_end = min(int(0.2 * sample_rate), len(rir))  # 200ms
    if late_end > late_start:
        late_energy = np.sum(rir[late_start:late_end] ** 2)
        metrics['late_energy_pct'] = (late_energy / total_energy) * 100
    else:
        metrics['late_energy_pct'] = 0
    
    # Tail (200ms+)
    if len(rir) > late_end:
        tail_energy = np.sum(rir[late_end:] ** 2)
        metrics['tail_energy_pct'] = (tail_energy / total_energy) * 100
    else:
        metrics['tail_energy_pct'] = 0
    
    # Decay analysis
    if len(rir) > 64:
        # RT60 estimation (rough)
        envelope = np.abs(rir)
        # Find -60dB point
        target_level = metrics['peak_value'] * 0.001  # -60dB
        decay_idx = np.where(envelope[metrics['peak_idx']:] < target_level)[0]
        if len(decay_idx) > 0:
            rt60_samples = decay_idx[0] + metrics['peak_idx']
            metrics['rt60_ms'] = rt60_samples / sample_rate * 1000
        else:
            metrics['rt60_ms'] = None
    
    # Structural realism score
    score = 0
    # Direct sound should dominate but not overwhelm
    if 20 <= metrics['direct_energy_pct'] <= 60:
        score += 25
    # Early reflections should be present but moderate
    if 10 <= metrics['early_energy_pct'] <= 35:
        score += 25
    # Late reverb should exist
    if metrics['late_energy_pct'] >= 5:
        score += 25
    # Peak should be very early
    if metrics['peak_time_ms'] <= 1.0:
        score += 25
    
    metrics['realism_score'] = score
    
    return metrics

def plot_rir_analysis(rir_path, output_dir):
    """Create detailed RIR analysis plots."""
    rir = load_rir(rir_path)
    metrics = analyze_rir_structure(rir)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'RIR Analysis: {rir_path.name}', fontsize=14, fontweight='bold')
    
    # Time domain plot
    axes[0, 0].plot(rir, 'b-', linewidth=0.8)
    axes[0, 0].axvline(metrics['peak_idx'], color='r', linestyle='--', alpha=0.7, label='Peak')
    axes[0, 0].set_title('Time Domain')
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Amplitude') 
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Energy envelope
    envelope = np.abs(rir)
    axes[0, 1].semilogy(envelope, 'g-', linewidth=1.0)
    axes[0, 1].set_title('Energy Envelope (Log Scale)')
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('|Amplitude| (log)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Energy distribution pie chart
    energies = [
        metrics['direct_energy_pct'],
        metrics['early_energy_pct'], 
        metrics['late_energy_pct'],
        metrics['tail_energy_pct']
    ]
    labels = ['Direct\n(0-0.3ms)', 'Early\n(0.3-50ms)', 'Late\n(50-200ms)', 'Tail\n(200ms+)']
    colors = ['red', 'orange', 'yellow', 'lightblue']
    
    axes[1, 0].pie(energies, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Energy Distribution')
    
    # Metrics text
    metrics_text = f"""Structural Analysis:
Peak Time: {metrics['peak_time_ms']:.2f} ms
Direct Energy: {metrics['direct_energy_pct']:.1f}%
Early Energy: {metrics['early_energy_pct']:.1f}%
Late Energy: {metrics['late_energy_pct']:.1f}%
Tail Energy: {metrics['tail_energy_pct']:.1f}%
RT60: {metrics['rt60_ms']:.1f} ms (est.)
Realism Score: {metrics['realism_score']}/100"""
    
    axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / f"rir_analysis_{rir_path.stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved analysis plot: {output_path}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Analyze RIR structure for acoustic realism")
    parser.add_argument("--rir-dir", type=str, required=True, help="Directory containing RIR files")
    parser.add_argument("--output-dir", type=str, default="experiments/rir_analysis", help="Output directory")
    parser.add_argument("--method", type=str, choices=['dqn', 'neural', 'both'], default='both', help="Which method to analyze")
    
    args = parser.parse_args()
    
    rir_dir = Path(args.rir_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    methods_to_analyze = []
    if args.method in ['dqn', 'both']:
        methods_to_analyze.append('dqn')
    if args.method in ['neural', 'both']:
        methods_to_analyze.append('neural')
    
    all_metrics = []
    
    for method in methods_to_analyze:
        method_dir = rir_dir / method
        if not method_dir.exists():
            print(f"Method directory not found: {method_dir}")
            continue
            
        print(f"\nAnalyzing {method.upper()} RIRs...")
        
        # Get the most recent RIRs (final estimates)
        rir_files = sorted(method_dir.glob("rir_*.npy"))
        
        if not rir_files:
            print(f"No RIR files found in {method_dir}")
            continue
            
        # Analyze last few RIRs to see final structure
        for rir_file in rir_files[-3:]:  # Last 3 RIRs
            try:
                metrics = plot_rir_analysis(rir_file, output_dir)
                metrics['method'] = method
                metrics['filename'] = rir_file.name
                all_metrics.append(metrics)
                
                print(f"  {rir_file.name}: Realism Score = {metrics['realism_score']}/100")
                
            except Exception as e:
                print(f"Error analyzing {rir_file}: {e}")
    
    # Summary comparison
    if len(all_metrics) > 0:
        print(f"\n=== RIR Structure Summary ===")
        dqn_metrics = [m for m in all_metrics if m['method'] == 'dqn']
        neural_metrics = [m for m in all_metrics if m['method'] == 'neural']
        
        if dqn_metrics:
            avg_dqn_realism = np.mean([m['realism_score'] for m in dqn_metrics])
            avg_dqn_direct = np.mean([m['direct_energy_pct'] for m in dqn_metrics])
            print(f"DQN - Avg Realism: {avg_dqn_realism:.1f}/100, Direct Energy: {avg_dqn_direct:.1f}%")
            
        if neural_metrics:
            avg_neural_realism = np.mean([m['realism_score'] for m in neural_metrics])
            avg_neural_direct = np.mean([m['direct_energy_pct'] for m in neural_metrics])
            print(f"Neural - Avg Realism: {avg_neural_realism:.1f}/100, Direct Energy: {avg_neural_direct:.1f}%")

if __name__ == "__main__":
    main()