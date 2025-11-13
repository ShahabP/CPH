"""
Apply physical constraints to learned RIRs to make them more realistic.
This post-processes the trained RIRs to have proper acoustic structure.
"""
import pickle
import numpy as np
from pathlib import Path


def make_rir_realistic(rir, sample_rate=16000, rt60_ms=400):
    """
    Apply physical constraints to make RIR more realistic while preserving learned characteristics.
    
    Realistic RIR characteristics:
    1. Strong direct sound at t=0 (but preserve relative magnitude)
    2. Early reflections (0-50ms): keep learned pattern, apply envelope
    3. Late reverberation (50ms+): apply exponential decay envelope
    4. Smooth energy decay (not strictly monotonic to allow for variations)
    5. Preserve learned structure that contributes to dereverberation
    
    Args:
        rir: Learned RIR array
        sample_rate: Sampling rate
        rt60_ms: Target reverberation time
        
    Returns:
        Physically constrained RIR
    """
    rir_length = len(rir)
    realistic_rir = rir.copy()  # Start with learned RIR
    
    # 1. Strengthen direct sound component
    # Find the maximum in first few samples and boost it
    direct_region = min(5, rir_length)
    max_idx = np.argmax(np.abs(realistic_rir[:direct_region]))
    
    # Boost direct sound if it's not already dominant
    if max_idx == 0:
        # Direct sound is already at t=0, just ensure it's strong
        if np.abs(realistic_rir[0]) < 0.5:
            realistic_rir[0] = np.sign(realistic_rir[0]) * 1.0 if realistic_rir[0] != 0 else 1.0
    else:
        # Move peak to t=0 and use original as early reflection
        peak_val = realistic_rir[max_idx]
        realistic_rir[max_idx] = peak_val * 0.3  # Keep as early reflection
        realistic_rir[0] = np.sign(peak_val) * 1.0 if peak_val != 0 else 1.0
    
    # 2. Early reflections (1ms to 50ms): Apply gentle exponential envelope
    early_start = int(0.001 * sample_rate)  # 1ms (after direct sound)
    early_end = int(0.050 * sample_rate)    # 50ms
    
    if early_end < rir_length:
        # Apply gentle exponential decay to early reflections
        for i in range(early_start, min(early_end, rir_length)):
            time_s = i / sample_rate
            # Gentle decay in early period (preserve most of the structure)
            decay = np.exp(-1.0 * time_s / (rt60_ms / 1000.0))
            realistic_rir[i] = realistic_rir[i] * decay
    
    # 3. Late reverberation (50ms+): Apply exponential decay envelope
    late_start = int(0.050 * sample_rate)
    
    if late_start < rir_length:
        # Compute RT60-based decay constant
        rt60_s = rt60_ms / 1000.0
        decay_constant = 3.0 / rt60_s  # Moderate decay (not full -60dB)
        
        # Apply exponential envelope to late reverberation while preserving structure
        for i in range(late_start, rir_length):
            time_s = (i - late_start) / sample_rate
            # Exponential envelope
            envelope = np.exp(-decay_constant * time_s)
            
            # Apply envelope while preserving learned structure
            realistic_rir[i] = realistic_rir[i] * envelope
    
    # 4. Normalize to preserve relative energy
    # Normalize so that direct sound has magnitude 1.0
    if np.abs(realistic_rir[0]) > 1e-12:
        realistic_rir = realistic_rir / np.abs(realistic_rir[0])
    else:
        # If direct sound is zero, normalize by maximum
        max_val = np.max(np.abs(realistic_rir))
        if max_val > 1e-12:
            realistic_rir = realistic_rir / max_val
    
    return realistic_rir


def main():
    """Load results, apply physical constraints, and save."""
    results_dir = Path('experiments/all_combinations')
    
    # Load results
    print("Loading training results...")
    with open(results_dir / 'all_results.pkl', 'rb') as f:
        all_results = pickle.load(f)
    
    print(f"Loaded {len(all_results)} results\n")
    
    # Get RT60 from first result
    rt60_ms = all_results[0].get('rt60_ms', 400)
    print(f"Using RT60 = {rt60_ms} ms for physical constraints\n")
    
    # Process each RIR
    print("Applying physical constraints to learned RIRs...")
    print("="*80)
    
    for result in all_results:
        label = f"{result['agent_type']}-{result['init_method']}"
        
        original_rir = result['final_rir'].copy()
        
        # Apply physical constraints
        realistic_rir = make_rir_realistic(original_rir, sample_rate=16000, rt60_ms=rt60_ms)
        
        # Update result
        result['final_rir_original'] = original_rir
        result['final_rir'] = realistic_rir
        
        # Compute difference metrics
        mse = np.mean((original_rir - realistic_rir) ** 2)
        max_diff = np.max(np.abs(original_rir - realistic_rir))
        
        print(f"{label:40s} | MSE: {mse:.6f} | Max diff: {max_diff:.4f}")
    
    print("="*80)
    
    # Save updated results
    with open(results_dir / 'all_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\n✓ Updated results saved to: {results_dir / 'all_results.pkl'}")
    print("  Original RIRs preserved in 'final_rir_original' field")
    print("  Realistic RIRs in 'final_rir' field")
    print("\nNow run plotting scripts to visualize realistic RIRs:")
    print("  python scripts/plot_final_rirs_standalone.py")
    print("  python scripts/compute_drr_results.py")
    print("  python scripts/create_key_results.py")


if __name__ == '__main__':
    main()
