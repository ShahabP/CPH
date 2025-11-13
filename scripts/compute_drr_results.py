"""
Compute DRR (Direct-to-Reverberant Ratio) for dereverberated speech using final learned RIRs.
This demonstrates the practical dereverberation performance of each method.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
import json


def generate_synthetic_reverberant_speech(sample_rate=16000, duration=2.0, rt60_ms=600):
    """
    Generate realistic synthetic reverberant speech signal for testing.
    Uses formant-based synthesis to create speech-like spectral characteristics.
    
    Args:
        sample_rate: Sampling rate in Hz
        duration: Duration in seconds
        rt60_ms: Reverberation time in milliseconds
        
    Returns:
        Tuple of (reverberant_speech, clean_speech, true_rir)
    """
    # Generate clean speech with realistic formant structure
    t = np.linspace(0, duration, int(sample_rate * duration))
    clean_speech = np.zeros_like(t)
    
    # Simulate speech with time-varying formants (vowel-like sounds)
    # Typical vowel formants for /a/, /e/, /i/, /o/, /u/
    vowel_formants = [
        ([730, 1090, 2440], [60, 70, 110]),   # /a/ as in "father"
        ([530, 1840, 2480], [60, 90, 120]),   # /e/ as in "bed"
        ([270, 2290, 3010], [40, 90, 100]),   # /i/ as in "beet"
        ([570, 840, 2410], [70, 80, 100]),    # /o/ as in "boat"
        ([300, 870, 2240], [40, 80, 100]),    # /u/ as in "boot"
    ]
    
    # Create segments with different vowels
    segment_duration = duration / len(vowel_formants)
    for idx, (formants, bandwidths) in enumerate(vowel_formants):
        start_idx = int(idx * segment_duration * sample_rate)
        end_idx = int((idx + 1) * segment_duration * sample_rate)
        segment_t = t[start_idx:end_idx]
        
        # Generate fundamental frequency (F0) with vibrato
        f0_base = 120 + 10 * np.sin(2 * np.pi * 5 * segment_t)  # 120 Hz with vibrato
        
        # Generate voiced source (glottal pulse train)
        source = np.zeros_like(segment_t)
        for harmonic in range(1, 20):
            amplitude = 1.0 / harmonic  # Natural harmonic decay
            source += amplitude * np.sin(2 * np.pi * f0_base * harmonic * segment_t)
        
        # Apply formant filters (resonances)
        for formant_freq, bandwidth in zip(formants, bandwidths):
            # Simple resonance modeling using bandpass characteristics
            resonance = np.exp(-(segment_t - segment_t[len(segment_t)//2])**2 / (2 * 0.1**2))
            formant_contribution = resonance * np.sin(2 * np.pi * formant_freq * segment_t)
            source += 0.3 * formant_contribution
        
        clean_speech[start_idx:end_idx] = source
    
    # Add realistic speech envelope (amplitude modulation)
    # Simulate syllable rate ~4 Hz
    syllable_envelope = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
    # Add faster modulations for phonetic detail
    phonetic_envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 12 * t + np.sin(2 * np.pi * 3 * t))
    clean_speech *= syllable_envelope * phonetic_envelope
    
    # Add slight noise for naturalness (breath noise, aspiration)
    noise = 0.02 * np.random.randn(len(clean_speech))
    clean_speech += noise
    
    # Normalize
    clean_speech = clean_speech / (np.max(np.abs(clean_speech)) + 1e-8)
    
    # Generate true RIR
    rir_length = 4096  # 256ms at 16kHz
    true_rir = np.zeros(rir_length)
    
    # Direct sound
    true_rir[0] = 1.0
    
    # Early reflections (10-50ms)
    early_start = int(0.01 * sample_rate)  # 10ms
    early_end = int(0.05 * sample_rate)    # 50ms
    for i in range(early_start, early_end, 50):
        if i < rir_length:
            true_rir[i] = 0.3 * np.exp(-i / (sample_rate * 0.02))
    
    # Late reverberation (exponential decay)
    rt60_samples = (rt60_ms / 1000.0) * sample_rate
    decay_constant = 3 * np.log(10) / rt60_samples  # -60dB decay
    
    for i in range(early_end, rir_length):
        decay = np.exp(-decay_constant * (i - early_end))
        # Add random fluctuations
        true_rir[i] = decay * np.random.randn() * 0.05
    
    # Normalize RIR
    true_rir = true_rir / np.sqrt(np.sum(true_rir ** 2))
    
    # Convolve to create reverberant speech
    reverberant_speech = signal.fftconvolve(clean_speech, true_rir, mode='same')
    reverberant_speech = reverberant_speech / (np.max(np.abs(reverberant_speech)) + 1e-8)
    
    return reverberant_speech, clean_speech, true_rir


def dereverberate_with_rir(reverb_audio, rir):
    """
    Dereverberate audio using estimated RIR via Wiener deconvolution.
    """
    # Pad RIR to avoid circular convolution artifacts
    rir_padded = np.zeros(len(reverb_audio) + len(rir) - 1)
    rir_padded[:len(rir)] = rir
    
    # Convert to frequency domain
    reverb_fft = np.fft.fft(reverb_audio, n=len(rir_padded))
    rir_fft = np.fft.fft(rir_padded)
    
    # Wiener deconvolution with regularization
    regularization = 0.01
    rir_conj = np.conj(rir_fft)
    rir_power = np.abs(rir_fft) ** 2
    
    # Wiener filter: H* / (|H|^2 + λ)
    wiener_filter = rir_conj / (rir_power + regularization)
    
    # Apply filter
    clean_fft = reverb_fft * wiener_filter
    
    # Convert back to time domain and trim to original length
    dereverberated = np.real(np.fft.ifft(clean_fft))[:len(reverb_audio)]
    
    return dereverberated


def compute_speech_drr(audio, sample_rate=16000):
    """
    Compute DRR (Direct-to-Reverberant Ratio) on speech signal.
    """
    if len(audio) < sample_rate // 10:  # Less than 100ms
        return -20.0
    
    # Frame-based analysis
    frame_size = int(0.025 * sample_rate)  # 25ms frames
    hop_size = int(0.01 * sample_rate)     # 10ms hop
    
    frames = []
    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i + frame_size]
        frames.append(frame)
    
    if len(frames) < 5:
        return -20.0
    
    # Compute frame energies
    frame_energies = [np.sum(frame ** 2) for frame in frames]
    
    # Find direct sound (strongest frames in first 200ms)
    max_direct_frames = min(20, len(frame_energies))  # First 200ms
    direct_energy = np.max(frame_energies[:max_direct_frames])
    
    # Estimate reverberation energy (mean of remaining frames)
    if len(frame_energies) > max_direct_frames:
        reverb_frames = frame_energies[max_direct_frames:]
        reverb_energy = np.mean(reverb_frames)
    else:
        reverb_energy = np.mean(frame_energies) * 0.1  # Assume 10% reverb
    
    # Compute DRR
    drr_db = 10 * np.log10((direct_energy + 1e-12) / (reverb_energy + 1e-12))
    
    return float(drr_db)


def plot_drr_comparison(results, reverb_speech, clean_speech, results_dir):
    """
    Plot comprehensive DRR comparison with waveforms and spectrograms.
    """
    sample_rate = 16000
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(5, 4, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle('Dereverberation Performance: Final Learned RIRs', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    colors = {
        'QN-random': '#1f77b4',
        'QN-exponential_decay': '#aec7e8',
        'DQN-random': '#ff7f0e',
        'DQN-exponential_decay': '#ffbb78',
        'Neural-random': '#2ca02c',
        'Neural-exponential_decay': '#98df8a'
    }
    
    # Row 0: Reference signals
    ax_ref_wave = fig.add_subplot(gs[0, :2])
    time_s = np.arange(len(reverb_speech)) / sample_rate
    ax_ref_wave.plot(time_s, reverb_speech, label='Reverberant', color='red', alpha=0.7, linewidth=0.8)
    ax_ref_wave.plot(time_s, clean_speech, label='Clean Reference', color='blue', alpha=0.7, linewidth=0.8)
    ax_ref_wave.set_xlabel('Time (s)', fontweight='bold')
    ax_ref_wave.set_ylabel('Amplitude', fontweight='bold')
    ax_ref_wave.set_title('Input Signals', fontweight='bold', fontsize=12)
    ax_ref_wave.legend()
    ax_ref_wave.grid(True, alpha=0.3)
    
    # Compute DRR for reference signals
    drr_reverb = compute_speech_drr(reverb_speech, sample_rate)
    drr_clean = compute_speech_drr(clean_speech, sample_rate)
    
    # DRR bar chart
    ax_drr_bar = fig.add_subplot(gs[0, 2:])
    labels = ['Reverberant\nInput', 'Clean\nReference'] + \
             [f"{r['agent_type']}\n{r['init_method'][:3]}" for r in results]
    drr_values = [drr_reverb, drr_clean] + [r['drr_improvement']['final_drr'] for r in results]
    colors_list = ['red', 'blue'] + [colors.get(f"{r['agent_type']}-{r['init_method']}", 'gray') 
                                      for r in results]
    
    bars = ax_drr_bar.bar(range(len(labels)), drr_values, color=colors_list, alpha=0.7, edgecolor='black')
    ax_drr_bar.axhline(y=drr_reverb, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Input DRR')
    ax_drr_bar.axhline(y=drr_clean, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Target DRR')
    ax_drr_bar.set_xticks(range(len(labels)))
    ax_drr_bar.set_xticklabels(labels, fontsize=9, rotation=0)
    ax_drr_bar.set_ylabel('DRR (dB)', fontweight='bold')
    ax_drr_bar.set_title('DRR Comparison: All Methods', fontweight='bold', fontsize=12)
    ax_drr_bar.legend(loc='upper left', fontsize=9)
    ax_drr_bar.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, drr_values):
        height = bar.get_height()
        ax_drr_bar.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Rows 1-4: Dereverberated waveforms for each method (2 per row)
    for idx, result in enumerate(results):
        row = 1 + idx // 2
        col_offset = (idx % 2) * 2
        
        label = f"{result['agent_type']}-{result['init_method']}"
        dereverb_audio = result['drr_improvement']['dereverberated_audio']
        final_drr = result['drr_improvement']['final_drr']
        drr_gain = result['drr_improvement']['drr_gain']
        
        # Waveform plot
        ax_wave = fig.add_subplot(gs[row, col_offset:col_offset+2])
        ax_wave.plot(time_s, dereverb_audio, color=colors.get(label, 'gray'), linewidth=0.8, alpha=0.8)
        ax_wave.plot(time_s, clean_speech, color='blue', linewidth=0.5, alpha=0.3, linestyle='--', label='Clean ref')
        ax_wave.set_xlabel('Time (s)', fontweight='bold', fontsize=9)
        ax_wave.set_ylabel('Amplitude', fontweight='bold', fontsize=9)
        ax_wave.set_title(f'{label}\nDRR: {final_drr:.1f} dB (Δ={drr_gain:+.1f} dB)', 
                         fontweight='bold', fontsize=10)
        ax_wave.legend(fontsize=7, loc='upper right')
        ax_wave.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'drr_dereverberation_results.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved DRR comparison plot: {results_dir / 'drr_dereverberation_results.png'}")
    plt.close()


def main():
    """Main function to compute DRR improvements."""
    results_dir = Path('experiments/all_combinations')
    
    # Load training results
    print("Loading training results...")
    with open(results_dir / 'all_results.pkl', 'rb') as f:
        all_results = pickle.load(f)
    
    print(f"Loaded {len(all_results)} trained models")
    
    # Generate test reverberant speech
    print("\nGenerating synthetic reverberant speech (RT60=400ms)...")
    reverb_speech, clean_speech, true_rir = generate_synthetic_reverberant_speech(
        sample_rate=16000, duration=2.0, rt60_ms=400
    )
    
    # Compute initial DRR
    initial_drr = compute_speech_drr(reverb_speech, sample_rate=16000)
    clean_drr = compute_speech_drr(clean_speech, sample_rate=16000)
    print(f"Reverberant speech DRR: {initial_drr:.2f} dB")
    print(f"Clean speech DRR: {clean_drr:.2f} dB")
    
    # Process each method
    print("\nDereverberating with learned RIRs...")
    print("="*80)
    
    drr_summary = {
        'initial_drr': float(initial_drr),
        'clean_reference_drr': float(clean_drr),
        'methods': []
    }
    
    for result in all_results:
        label = f"{result['agent_type']}-{result['init_method']}"
        final_rir = result['final_rir']
        
        # Dereverberate using learned RIR
        dereverberated = dereverberate_with_rir(reverb_speech, final_rir)
        
        # Compute DRR of dereverberated speech
        final_drr = compute_speech_drr(dereverberated, sample_rate=16000)
        drr_gain = final_drr - initial_drr
        
        # Store results
        result['drr_improvement'] = {
            'initial_drr': float(initial_drr),
            'final_drr': float(final_drr),
            'drr_gain': float(drr_gain),
            'dereverberated_audio': dereverberated
        }
        
        # Add to summary
        drr_summary['methods'].append({
            'agent_type': result['agent_type'],
            'init_method': result['init_method'],
            'initial_drr': float(initial_drr),
            'final_drr': float(final_drr),
            'drr_gain': float(drr_gain),
            'rir_correlation': result['final_metrics']['avg_correlation']
        })
        
        print(f"{label:40s} | Initial: {initial_drr:6.2f} dB → Final: {final_drr:6.2f} dB | "
              f"Gain: {drr_gain:+6.2f} dB | RIR Corr: {result['final_metrics']['avg_correlation']:.3f}")
    
    print("="*80)
    
    # Save DRR summary
    with open(results_dir / 'drr_summary.json', 'w') as f:
        json.dump(drr_summary, f, indent=2)
    print(f"\nSaved DRR summary: {results_dir / 'drr_summary.json'}")
    
    # Create comprehensive plots
    print("\nGenerating DRR comparison plots...")
    plot_drr_comparison(all_results, reverb_speech, clean_speech, results_dir)
    
    # Find best method
    best_method = max(all_results, key=lambda x: x['drr_improvement']['final_drr'])
    best_label = f"{best_method['agent_type']}-{best_method['init_method']}"
    best_drr = best_method['drr_improvement']['final_drr']
    best_gain = best_method['drr_improvement']['drr_gain']
    
    print("\n" + "="*80)
    print(f"BEST DEREVERBERATION PERFORMANCE:")
    print(f"  Method: {best_label}")
    print(f"  Final DRR: {best_drr:.2f} dB (Δ={best_gain:+.2f} dB)")
    print(f"  RIR Correlation: {best_method['final_metrics']['avg_correlation']:.3f}")
    print("="*80)
    
    print("\n✓ All DRR analysis complete!")


if __name__ == '__main__':
    main()
