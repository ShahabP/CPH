import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.neural_rir_agent import NeuralRIREnvironment, NeuralRIRAgent, compute_rir_drr_metric
from pesq import pesq
from pystoi import stoi


def spsa_optimize(initial_h, reverb, agent, iterations=15, a=0.1, c=0.01, sample_rate=16000):
    h = initial_h.copy()
    for k in range(iterations):
        ak = a / (1 + k * 0.03)
        ck = c / (1 + k * 0.01)
        v = np.random.choice([1.0, -1.0], size=h.shape)
        h_plus = h + ck * v
        h_minus = h - ck * v
        r_plus = agent.compute_reward(h_plus, reverb, None, sample_rate)
        r_minus = agent.compute_reward(h_minus, reverb, None, sample_rate)
        g_hat = (r_plus - r_minus) / (2.0 * ck) * v
        h = h + ak * g_hat
        # normalization
        norm = np.linalg.norm(h) + 1e-12
        h = h / norm
    return h


def run_single_case():
    sample_rate = 16000
    duration = 2.0
    rir_length = 1024
    N = 20  # episodes
    K = 15  # steps per episode

    # Create a speech-like test signal
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    clean = 0.5 * np.sin(2 * np.pi * 220 * t) * np.exp(-t * 1.2)

    # True RIR (for simulation)
    env = NeuralRIREnvironment(max_iterations=K, rir_length=rir_length, sample_rate=sample_rate)

    # Use agent helper to generate exponential prior RIR
    base_agent = NeuralRIRAgent(rir_length=rir_length)
    true_rir = base_agent._get_exponential_decay_rir(rir_length, rt60_ms=400.0)
    reverb = np.convolve(clean, true_rir)[: len(clean)]

    # Neural-Exp: run N independent episodes (fresh agent per episode)
    neural_results = []
    for e in range(N):
        np.random.seed(42 + e)
        agent = NeuralRIRAgent(rir_length=rir_length)
        env = NeuralRIREnvironment(max_iterations=K, rir_length=rir_length, sample_rate=sample_rate)
        state = env.reset(reverb, clean, initial_rir=base_agent._get_exponential_decay_rir(rir_length))
        for k in range(K):
            state, reward, terminated, info = env.step(agent)
            if terminated:
                break
        final_rir = state
        dereverb = agent._dereverberate_with_rir(reverb, final_rir)
        try:
            pesq_score = pesq(sample_rate, clean.astype('float32'), dereverb.astype('float32'), 'wb')
        except Exception:
            pesq_score = float('nan')
        try:
            stoi_score = stoi(clean.astype('float32'), dereverb.astype('float32'), sample_rate, extended=False)
        except Exception:
            stoi_score = float('nan')
        estimated_drr = compute_rir_drr_metric(final_rir, sample_rate)
        neural_results.append({'episode': e, 'drr': float(estimated_drr), 'pesq': float(pesq_score), 'stoi': float(stoi_score)})

    neural_df = pd.DataFrame(neural_results)
    best_neural = neural_df.loc[neural_df['pesq'].idxmax()]

    # Baseline TTO: run SPSA starting from same exponential init
    np.random.seed(100)
    init_h = base_agent._get_exponential_decay_rir(rir_length)
    baseline_agent = NeuralRIRAgent(rir_length=rir_length)
    h_opt = spsa_optimize(init_h, reverb, baseline_agent, iterations=50, a=0.2, c=0.01, sample_rate=sample_rate)
    dereverb_b = baseline_agent._dereverberate_with_rir(reverb, h_opt)
    try:
        pesq_b = pesq(sample_rate, clean.astype('float32'), dereverb_b.astype('float32'), 'wb')
    except Exception:
        pesq_b = float('nan')
    try:
        stoi_b = stoi(clean.astype('float32'), dereverb_b.astype('float32'), sample_rate, extended=False)
    except Exception:
        stoi_b = float('nan')
    drr_b = compute_rir_drr_metric(h_opt, sample_rate)

    # Summary
    out = {
        'neural_best_episode': int(best_neural['episode']),
        'neural_best_drr': float(best_neural['drr']),
        'neural_best_pesq': float(best_neural['pesq']),
        'neural_best_stoi': float(best_neural['stoi']),
        'baseline_drr': float(drr_b),
        'baseline_pesq': float(pesq_b),
        'baseline_stoi': float(stoi_b),
        'neural_all': neural_df.to_dict(orient='records')
    }

    out_csv = 'experiments/single_case_check.json'
    import json
    with open(out_csv, 'w') as f:
        json.dump(out, f, indent=2)

    print('Single-case check saved to', out_csv)
    print(out)


if __name__ == '__main__':
    run_single_case()
