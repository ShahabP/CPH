#!/usr/bin/env python3
"""
RIR Estimation vs Input Speech Length Analysis
==============================================

This script explains the relationship between the estimated RIR length 
and the input reverberant speech length in the RL-based RIR estimation system.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_rir_vs_speech_length():
    """Analyze how RIR estimation relates to input speech length."""
    
    print("🎯 RIR Estimation vs Input Reverberant Speech Length")
    print("=" * 65)
    print()
    
    print("📊 KEY RELATIONSHIP:")
    print("   • RIR Length: FIXED at 1024 samples (64ms at 16kHz)")
    print("   • Speech Length: VARIABLE (typically 1-3 seconds)")
    print("   • Ratio: Speech is ~16-48x longer than RIR")
    print()
    
    print("🔬 TECHNICAL DETAILS:")
    print()
    
    # Configuration values from the codebase
    sample_rate = 16000
    rir_length = 1024
    speech_duration_sec = 3.0  # From rl_framework
    demo_duration_sec = 2.0    # From neural_rir_agent demo
    
    speech_samples = int(speech_duration_sec * sample_rate)
    demo_samples = int(demo_duration_sec * sample_rate)
    
    rir_duration_ms = (rir_length / sample_rate) * 1000
    
    print(f"📏 DIMENSIONS:")
    print(f"   • Sample Rate: {sample_rate:,} Hz")
    print(f"   • RIR Length: {rir_length} samples ({rir_duration_ms:.1f} ms)")
    print(f"   • Speech Length (training): {speech_samples:,} samples ({speech_duration_sec:.1f} sec)")
    print(f"   • Speech Length (demo): {demo_samples:,} samples ({demo_duration_sec:.1f} sec)")
    print()
    
    print(f"📐 RATIOS:")
    speech_to_rir_ratio = speech_samples / rir_length
    demo_to_rir_ratio = demo_samples / rir_length
    print(f"   • Training Speech:RIR = {speech_to_rir_ratio:.1f}:1")
    print(f"   • Demo Speech:RIR = {demo_to_rir_ratio:.1f}:1")
    print()
    
    print("🔄 CONVOLUTION PROCESS:")
    print("   reverb_audio = np.convolve(clean_audio, estimated_rir, mode='same')")
    print("   ")
    print("   • Input: clean_audio (48,000 samples)")
    print("   • Filter: estimated_rir (1,024 samples)")
    print("   • Output: reverb_audio (48,000 samples) - same length as input")
    print()
    
    print("🧠 RL ESTIMATION STRATEGY:")
    print("   1. Agent receives ENTIRE reverberant speech segment")
    print("   2. Estimates FIXED-LENGTH RIR (1024 samples)")
    print("   3. Uses RIR to dereverberate ENTIRE speech")
    print("   4. Computes reward based on full speech quality")
    print()
    
    print("⏰ ACOUSTIC SIGNIFICANCE:")
    print("   • RIR captures room acoustics up to 64ms")
    print("   • Covers: direct sound + early reflections + initial decay")
    print("   • 64ms is sufficient for most room acoustic modeling")
    print("   • Late reverberation (>64ms) handled by tail modeling")
    print()
    
    print("📊 DEREVERBERATION MECHANICS:")
    print("   From _dereverberate_with_rir() method:")
    print("   ")
    print("   1. Pad RIR to speech length: rir_padded = zeros(speech_len + rir_len - 1)")
    print("   2. FFT both signals: reverb_fft, rir_fft")
    print("   3. Wiener deconvolution: wiener_filter = rir_conj / (|rir|² + λ)")
    print("   4. Apply filter: clean_fft = reverb_fft * wiener_filter")
    print("   5. Return to time domain: dereverberated[:speech_length]")
    print()
    
    print("🎯 PRACTICAL IMPLICATIONS:")
    print("   • RIR estimation is INDEPENDENT of speech length")
    print("   • Same 1024-sample RIR can dereverberate any speech length")
    print("   • Agent learns from ENTIRE speech segment quality")
    print("   • Longer speech provides more training signal")
    print()
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('RIR vs Speech Length Relationship', fontsize=16, fontweight='bold')
    
    # 1. Length comparison
    categories = ['RIR\n(64ms)', 'Demo Speech\n(2.0s)', 'Training Speech\n(3.0s)']
    lengths = [rir_length, demo_samples, speech_samples]
    colors = ['red', 'blue', 'green']
    
    bars = ax1.bar(categories, lengths, color=colors, alpha=0.7)
    ax1.set_ylabel('Samples')
    ax1.set_title('Length Comparison (Linear Scale)')
    ax1.set_yscale('log')
    
    # Add value labels
    for bar, length in zip(bars, lengths):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{length:,}', ha='center', va='bottom')
    
    # 2. Time domain visualization
    t_rir = np.arange(rir_length) / sample_rate * 1000  # ms
    t_speech = np.arange(speech_samples) / sample_rate  # seconds
    
    # Synthetic RIR
    rir_demo = np.zeros(rir_length)
    rir_demo[0] = 1.0
    for i in range(1, 200):
        rir_demo[i] = 0.3 * np.exp(-i / 50) * (1 + 0.1 * np.random.randn())
    
    ax2.plot(t_rir, rir_demo, 'r-', linewidth=2, label='RIR (64ms)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Room Impulse Response - Full 64ms Duration')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Speech visualization (first 200ms only for visibility)
    speech_show_samples = int(0.2 * sample_rate)  # 200ms
    t_speech_show = np.arange(speech_show_samples) / sample_rate * 1000  # ms
    
    # Synthetic speech
    speech_demo = 0.2 * np.sin(2 * np.pi * 440 * t_speech_show/1000) * np.exp(-t_speech_show/1000)
    
    ax3.plot(t_speech_show, speech_demo, 'b-', linewidth=1, label='Speech (first 200ms of 3000ms)')
    ax3.axvspan(0, rir_duration_ms, alpha=0.3, color='red', label=f'RIR Duration ({rir_duration_ms:.1f}ms)')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Reverberant Speech - Showing First 200ms of Full 3-Second Signal')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    
    # Save visualization
    viz_dir = Path("experiments/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    save_path = viz_dir / "rir_vs_speech_length.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"📈 Visualization saved to: {save_path}")
    print()
    
    print("🔑 SUMMARY:")
    print("   • RIR: Fixed 1024 samples (64ms) - captures room acoustics")
    print("   • Speech: Variable length (1-3 seconds) - provides training data")  
    print("   • Agent uses ENTIRE speech to evaluate RIR quality")
    print("   • Same RIR can dereverberate any length speech")
    print("   • Longer speech = more robust reward signal")

if __name__ == "__main__":
    analyze_rir_vs_speech_length()