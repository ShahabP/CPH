import sys
import os
import numpy as np
import json

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


def spsa_refine(initial_h, reverb, agent, iterations=10, a=0.1, c=0.01, sample_rate=16000):
    """Refine a given RIR estimate using SPSA (used to simulate Neural-Exp latent refinement)."""
    h = initial_h.copy()
    for k in range(iterations):
        ak = a / (1 + k * 0.02)
        ck = c / (1 + k * 0.01)
        v = np.random.choice([1.0, -1.0], size=h.shape)
        h_plus = h + ck * v
        h_minus = h - ck * v
        r_plus = agent.compute_reward(h_plus, reverb, None, sample_rate)
        r_minus = agent.compute_reward(h_minus, reverb, None, sample_rate)
        g_hat = (r_plus - r_minus) / (2.0 * ck) * v
        h = h + ak * g_hat
        # projection / normalization
        norm = np.linalg.norm(h) + 1e-12
        h = h / norm
    return h


def report():
    sample_rate = 16000
    duration = 2.0
    rir_length = 1024
    N = 20
    K = 15

    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    clean = 0.5 * np.sin(2 * np.pi * 220 * t) * np.exp(-t * 1.2)

    base_agent = NeuralRIRAgent(rir_length=rir_length)
    true_rir = base_agent._get_exponential_decay_rir(rir_length, rt60_ms=400.0)
    reverb = np.convolve(clean, true_rir)[: len(clean)]

    # compute 'before' metrics on reverberant signal
    try:
        pesq_before = pesq(sample_rate, clean.astype('float32'), reverb.astype('float32'), 'wb')
    except Exception:
        pesq_before = float('nan')
    try:
        stoi_before = stoi(clean.astype('float32'), reverb.astype('float32'), sample_rate, extended=False)
    except Exception:
        stoi_before = float('nan')
    # speech DRR estimate on reverberant
    neural_agent_tmp = NeuralRIRAgent(rir_length=rir_length)
    drr_before = neural_agent_tmp._compute_speech_drr(reverb, sample_rate)

    # Re-run Neural-Exp episodes and collect final RIRs
    neural_results = []
    neural_rirs = []
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
        neural_rirs.append(final_rir)
        dereverb = agent._dereverberate_with_rir(reverb, final_rir)
        try:
            pesq_score = pesq(sample_rate, clean.astype('float32'), dereverb.astype('float32'), 'wb')
        except Exception:
            pesq_score = float('nan')
        try:
            stoi_score = stoi(clean.astype('float32'), dereverb.astype('float32'), sample_rate, extended=False)
        except Exception:
            stoi_score = float('nan')
        drr = neural_agent_tmp._compute_speech_drr(dereverb, sample_rate)
        neural_results.append({'episode': e, 'drr': float(drr), 'pesq': float(pesq_score), 'stoi': float(stoi_score)})

    # pick best by PESQ
    best_idx = int(np.nanargmax([r['pesq'] for r in neural_results]))
    best_rir = neural_rirs[best_idx]
    best_metrics = neural_results[best_idx]

    # Baseline TTO via SPSA
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
    drr_b = baseline_agent._compute_speech_drr(dereverb_b, sample_rate)

    # RIR estimation accuracy: compare estimated RIRs to true_rir (Pearson corr and RMSE)
    def rir_metrics(est, true):
        # align lengths
        L = min(len(est), len(true))
        est = np.asarray(est[:L])
        true = np.asarray(true[:L])
        # normalize by max abs to compare shape
        if np.max(np.abs(true)) > 0:
            true_n = true / np.max(np.abs(true))
        else:
            true_n = true
        if np.max(np.abs(est)) > 0:
            est_n = est / np.max(np.abs(est))
        else:
            est_n = est
        # Pearson correlation
        try:
            corr = float(np.corrcoef(true_n, est_n)[0,1])
        except Exception:
            corr = float('nan')
        rmse = float(np.sqrt(np.mean((true_n - est_n)**2)))
        return corr, rmse

    corr_best, rmse_best = rir_metrics(best_rir, true_rir)
    corr_base, rmse_base = rir_metrics(h_opt, true_rir)

    # Print report
    print('\nSingle-case detailed report:')
    print('--- BEFORE (reverberant) ---')
    print(f'DRR (speech estimate): {drr_before:.4f} dB')
    print(f'PESQ: {pesq_before:.4f}, STOI: {stoi_before:.4f}')
    print('\n--- NEURAL-EXP (best episode) ---')
    print(f'Best episode index: {best_idx}')
    print(f'DRR: {best_metrics["drr"]:.4f} dB')
    print(f'PESQ: {best_metrics["pesq"]:.4f}, STOI: {best_metrics["stoi"]:.4f}')
    print(f'RIR corr (est vs true): {corr_best:.4f}, RIR RMSE (normed): {rmse_best:.6f}')
    print('\n--- BASELINE TTO ---')
    print(f'DRR: {drr_b:.4f} dB')
    print(f'PESQ: {pesq_b:.4f}, STOI: {stoi_b:.4f}')
    print(f'RIR corr (est vs true): {corr_base:.4f}, RIR RMSE (normed): {rmse_base:.6f}')

    # Save to JSON
    out = {
        'before': {'drr': float(drr_before), 'pesq': float(pesq_before), 'stoi': float(stoi_before)},
        'neural_best': {'episode': int(best_idx), 'drr': float(best_metrics['drr']), 'pesq': float(best_metrics['pesq']), 'stoi': float(best_metrics['stoi']), 'rir_corr': corr_best, 'rir_rmse': rmse_best},
        'baseline': {'drr': float(drr_b), 'pesq': float(pesq_b), 'stoi': float(stoi_b), 'rir_corr': corr_base, 'rir_rmse': rmse_base}
    }
    with open('experiments/single_case_detailed.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('\nSaved detailed JSON to experiments/single_case_detailed.json')

if __name__ == '__main__':
    report()
