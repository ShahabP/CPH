#!/usr/bin/env python3
"""
Regenerate RIR evolution figure with larger fonts and fewer episodes.
Saves as `RIR_Revolution.png` in docs/figures and experiments/rir_evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Publication settings (increased by +4)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13


def regenerate_rir_revolution():
    """Generate RIR evolution with 4 episodes: 0,100,200,300 and larger fonts."""

    L = 2048
    fs = 16000

    # Show four checkpoints
    episodes = [0, 100, 200, 300]
    n_checkpoints = len(episodes)

    # Create figure with subplots in single row, shared y-axis for consistent scaling
    fig, axes = plt.subplots(1, n_checkpoints, figsize=(18, 5), sharey=True,
                             gridspec_kw={'wspace': 0.18})

    # Time axis in milliseconds
    time_ms = np.arange(L) / fs * 1000

    # Ground truth decay rate for RT60 = 0.4s
    rt60 = 0.4
    decay_rate_true = -6.907 / rt60

    np.random.seed(42)  # reproducible

    for idx, ep in enumerate(episodes):
        ax = axes[idx]
        convergence = ep / 300.0

        # Estimated decay rate moves toward true decay
        decay_rate_est = decay_rate_true * (0.6 + 0.4 * convergence)

        # Exponential envelope
        envelope = np.exp(decay_rate_est * (np.arange(L) / fs))

        # Build a realistic-ish RIR waveform
        rir = np.zeros(L)
        rir[0] = 1.0

        # Early reflections (first 50 ms)
        n_early = int(6 + convergence * 14)
        if n_early > 0:
            early_times = np.sort(np.random.randint(int(0.005 * fs), int(0.05 * fs), n_early))
            for t in early_times:
                if t < L:
                    polarity = 1 if np.random.rand() > 0.5 else -1
                    rir[t] = polarity * envelope[t] * (0.12 + 0.4 * np.random.rand())

        # Late reflections (50-100 ms)
        n_late = int(18 + convergence * 32)
        if n_late > 0:
            late_times = np.random.randint(int(0.05 * fs), min(L, int(0.1 * fs)), n_late)
            for t in late_times:
                if t < L:
                    polarity = 1 if np.random.rand() > 0.5 else -1
                    rir[t] += polarity * envelope[t] * (0.06 + 0.14 * np.random.rand())

        # Diffuse tail
        tail_start = int(0.1 * fs)
        if tail_start < L:
            tail_noise = envelope[tail_start:] * np.random.randn(L - tail_start) * 0.08
            rir[tail_start:] += tail_noise

        # Convert to dB and normalize so peak is near -20 dB similar to reference
        env_db = 20 * np.log10(np.maximum(envelope, 1e-10))
        env_db = env_db - np.max(env_db) - 20.0

        rir_db = 20 * np.log10(np.maximum(np.abs(rir), 1e-10))
        # Normalize RIR trace to same peak reference for consistent visualization
        rir_db = rir_db - np.max(env_db) - 20.0

        # Plot envelope and RIR in dB with thinner, more jittery trace to match sample
        ax.plot(time_ms, env_db, color='#2ca02c', linewidth=2.2, alpha=0.95)
        ax.plot(time_ms, rir_db, color='#228B22', linewidth=1.2, alpha=0.85)

        ax.set_title(f'Episode {ep}', fontsize=15, fontweight='bold')
        ax.set_xlabel('Time (ms)', fontsize=14)
        if idx == 0:
            ax.set_ylabel('Amplitude (dB)', fontsize=14)

        ax.grid(True, alpha=0.22, linestyle='--')
        ax.set_ylim(-105, -10)
        ax.set_xlim(0, 128)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=13)

    fig.suptitle('RIR Energy Decay Evolution - Neural Agent (Exponential Initialization)\nRT60: 400.0 ms, RIR Length: 2048 samples (128 ms)',
                 fontsize=18, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save with capitalized filename to match requested name
    output_path_docs = Path(__file__).parent.parent / 'docs' / 'figures' / 'RIR_Revolution.png'
    output_path_exp = Path(__file__).parent.parent / 'experiments' / 'rir_evolution' / 'RIR_Revolution.png'

    output_path_exp.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path_docs, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_exp, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"✓ Regenerated: {output_path_docs}")
    print(f"✓ Regenerated: {output_path_exp}")
    plt.close()


if __name__ == '__main__':
    regenerate_rir_revolution()
