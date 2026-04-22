import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.neural_rir_agent import NeuralRIREnvironment, NeuralRIRAgent, compute_rir_drr_metric
from pesq import pesq
from pystoi import stoi


def spsa_optimize(initial_h, reverb, agent, iterations=15, a=0.1, c=0.01, sample_rate=16000):
    """Simple SPSA-based optimizer to approximate gradient ascent on the reward."""
    h = initial_h.copy()
    for k in range(iterations):
        # gain sequences (Robbins-Monro style)
        ak = a / (1 + k * 0.03)
        ck = c / (1 + k * 0.01)
        # random perturbation vector
        v = np.random.choice([1.0, -1.0], size=h.shape)
        h_plus = h + ck * v
        h_minus = h - ck * v
        r_plus = agent.compute_reward(h_plus, reverb, None, sample_rate)
        r_minus = agent.compute_reward(h_minus, reverb, None, sample_rate)
        g_hat = (r_plus - r_minus) / (2.0 * ck) * v
        h = h + ak * g_hat
        # projection: enforce causality and normalization similar to paper
        h[:0] = h[:0]  # no-op to keep slice semantics
        # normalize
        norm = np.linalg.norm(h) + 1e-12
        h = h / norm
    return h


def run_baseline_demo():
    sample_rate = 16000
    durations = [1.0, 2.0]
    rir_lengths = [512, 1024]
    results = []

    for dur in durations:
        for rl in rir_lengths:
            t = np.linspace(0, dur, int(dur * sample_rate), endpoint=False)
            clean = 0.5 * np.sin(2 * np.pi * 220 * t) * np.exp(-t * 1.5)

            env = NeuralRIREnvironment(max_iterations=10, rir_length=rl, sample_rate=sample_rate)
            agent = NeuralRIRAgent(rir_length=rl, learning_rate=1e-3)

            true_rir = agent._get_exponential_decay_rir(rl, rt60_ms=300.0)
            reverb = np.convolve(clean, true_rir)[: len(clean)]

            # initialize with exponential prior
            h0 = agent._get_exponential_decay_rir(rl, rt60_ms=300.0)

            # run SPSA-based TTO (baseline)
            h_opt = spsa_optimize(h0, reverb, agent, iterations=15, a=0.2, c=0.01, sample_rate=sample_rate)

            estimated_drr = compute_rir_drr_metric(h_opt, sample_rate)
            dereverb = agent._dereverberate_with_rir(reverb, h_opt)

            try:
                pesq_score = pesq(sample_rate, clean.astype('float32'), dereverb.astype('float32'), 'wb')
            except Exception:
                pesq_score = float('nan')
            try:
                stoi_score = stoi(clean.astype('float32'), dereverb.astype('float32'), sample_rate, extended=False)
            except Exception:
                stoi_score = float('nan')

            results.append({
                'duration_s': dur,
                'rir_length': rl,
                'estimated_drr_db': float(estimated_drr),
                'pesq': float(pesq_score),
                'stoi': float(stoi_score)
            })

    df = pd.DataFrame(results)
    out_csv = 'experiments/baseline_tto_demo_results.csv'
    df.to_csv(out_csv, index=False)
    print('Baseline TTO demo results saved to', out_csv)
    print(df)

if __name__ == '__main__':
    run_baseline_demo()
