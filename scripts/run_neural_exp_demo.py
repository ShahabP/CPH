import sys
import os
import numpy as np
import pandas as pd

# Add project root to path so `src` can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.neural_rir_agent import NeuralRIREnvironment, NeuralRIRAgent, compute_rir_drr_metric
from pesq import pesq
from pystoi import stoi


def run_demo():
    sample_rate = 16000
    durations = [1.0, 2.0]
    rir_lengths = [512, 1024]
    results = []

    for dur in durations:
        for rl in rir_lengths:
            # synthetic clean signal: speech-like chirp
            t = np.linspace(0, dur, int(dur * sample_rate), endpoint=False)
            clean = 0.5 * np.sin(2 * np.pi * 220 * t) * np.exp(-t * 1.5)

            # create true RIR using environment helper
            env = NeuralRIREnvironment(max_iterations=10, rir_length=rl, sample_rate=sample_rate)
            agent = NeuralRIRAgent(rir_length=rl, learning_rate=1e-3)

            true_rir = agent._get_exponential_decay_rir(rl, rt60_ms=300.0)
            reverb = np.convolve(clean, true_rir)[: len(clean)]

            state = env.reset(reverb, clean, initial_rir=None)

            for step in range(env.max_iterations):
                state, reward, terminated, info = env.step(agent)
                if terminated:
                    break
            final_rir = state
            estimated_drr = compute_rir_drr_metric(final_rir, sample_rate)

            # compute dereverberated audio using final estimated RIR
            dereverb = agent._dereverberate_with_rir(reverb, final_rir)

            # compute PESQ and STOI against clean (clipped/padded as needed)
            try:
                # pesq requires 16000 or 8000 sr and signals in float32
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
                'final_rir_energy': float(np.sum(final_rir ** 2)),
                'pesq': float(pesq_score),
                'stoi': float(stoi_score),
                'last_reward': float(agent.episode_rewards[-1]) if len(agent.episode_rewards) else 0.0
            })

    df = pd.DataFrame(results)
    out_csv = 'experiments/neural_exp_demo_results.csv'
    df.to_csv(out_csv, index=False)
    print('Demo results saved to', out_csv)
    print(df)


if __name__ == '__main__':
    run_demo()
