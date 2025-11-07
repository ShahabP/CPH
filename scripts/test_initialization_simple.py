#!/usr/bin/env python3
"""
Standalone test of RIR initialization methods
"""

import numpy as np
import matplotlib.pyplot as plt

def get_exponential_decay_rir(rir_length=1024):
    """Generate exponential decay RIR initialization.

    Creates a physically plausible RIR with:
    - Strong direct sound at t=0
    - Exponential decay following typical room acoustics
    - Realistic decay constants for early and late reflections
    """
    rir = np.zeros(rir_length)

    # Direct sound (strong impulse at t=0)
    rir[0] = 1.0

    # Early reflections (first 200 samples ~12.5ms)
    early_decay = 200  # samples
    for i in range(1, min(early_decay, rir_length)):
        # Add some early reflections with decreasing amplitude
        reflection_strength = 0.3 * np.exp(-i / 100)  # Fast initial decay
        if np.random.random() < 0.1:  # Sparse early reflections
            rir[i] += reflection_strength * (0.5 + np.random.random())

    # Late reverberation (exponential tail)
    for i in range(early_decay, rir_length):
        # Exponential decay with realistic RT60 characteristics
        decay_rate = 50  # samples (faster decay = shorter RT60)
        amplitude = 0.1 * np.exp(-i / decay_rate)
        rir[i] = amplitude * (0.8 + 0.4 * np.random.random())  # Add some variation

    # Normalize to ensure direct sound is prominent
    if np.max(np.abs(rir)) > 0:
        rir = rir / np.max(np.abs(rir))

    return rir

def test_initialization_methods():
    """Test both RIR initialization methods."""

    print("🧪 Testing RIR Initialization Methods")
    print("=" * 40)

    np.random.seed(42)  # For reproducible results

    # Test random initialization
    print("Testing Random Initialization:")
    random_rir = np.random.normal(0, 0.1, 1024)
    print(f"  Shape: {random_rir.shape}")
    print(".3f")
    print(".3f")
    print(".3f")

    # Test exponential decay initialization
    print("\nTesting Exponential Decay Initialization:")
    exp_rir = get_exponential_decay_rir(1024)
    print(f"  Shape: {exp_rir.shape}")
    print(".3f")
    print(".3f")
    print(".3f")

    # Compare characteristics
    print("\n📊 Comparison:")
    print(".3f")
    print(".3f")
    print(".3f")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('RIR Initialization Methods Comparison')

    time_ms = np.arange(len(random_rir)) / 16  # 16kHz to ms

    ax1.plot(time_ms, random_rir, 'r-', alpha=0.7, label='Random')
    ax1.set_title('Random Initialization\n(Gaussian noise, σ=0.1)')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 64)

    ax2.plot(time_ms, exp_rir, 'b-', alpha=0.7, label='Exp Decay')
    ax2.set_title('Exponential Decay Initialization\n(Physically plausible RIR)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 64)

    plt.tight_layout()

    # Save plot
    import os
    os.makedirs('experiments/initialization_comparison', exist_ok=True)
    plt.savefig('experiments/initialization_comparison/init_methods_test.png', dpi=150, bbox_inches='tight')
    print(f"\n📈 Plot saved to: experiments/initialization_comparison/init_methods_test.png")

    print("\n✅ Initialization methods test completed successfully!")
    print("\nKey Differences:")
    print("• Random: Unbiased exploration, may need more iterations to converge")
    print("• Exp Decay: Physically plausible starting point, faster convergence")

if __name__ == "__main__":
    test_initialization_methods()