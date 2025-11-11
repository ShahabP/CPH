"""
Quick training run with DRY RIR optimization - fewer episodes for fast verification.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
from pathlib import Path

from src.neural_rir_agent import NeuralRIRAgent, NeuralRIREnvironment, compute_rir_drr_metric

def quick_train():
    """Quick training to verify positive DRR."""
    
    print("="*60)
    print("Quick DRY RIR Training Test")
    print("="*60)
    
    rt60_values = [100, 500, 1000]  # Just 3 RT60 values
    rir_length = 6400
    episodes = 50  # Reduced from 200
    sample_rate = 16000
    
    results = {}
    
    for rt60_ms in rt60_values:
        print(f"\n=== RT60 = {rt60_ms} ms ===")
        
        # Create agent
        agent = NeuralRIRAgent(
            rir_length=rir_length,
            learning_rate=3e-4,
            gamma=0.95,
            update_scale=0.05
        )
        
        # Training loop
        final_drrs = []
        
        for episode in range(episodes):
            # Generate synthetic data
            duration = 2.0
            t = np.linspace(0, duration, int(duration * sample_rate))
            clean = 0.2 * np.sum([
                np.sin(2 * np.pi * f * t) * np.exp(-t * 0.5)
                for f in [440, 880, 1320]
            ], axis=0)
            
            # True RIR with exponential decay
            true_rir = agent._get_exponential_decay_rir(rir_length, rt60_ms)
            reverb = np.convolve(clean, true_rir, mode='same')
            
            # Initialize RIR
            initial_rir = agent._get_exponential_decay_rir(rir_length, rt60_ms)
            
            # Create environment
            env = NeuralRIREnvironment(max_iterations=10, sample_rate=sample_rate, rir_length=rir_length)
            env.reset(reverb, clean, initial_rir=initial_rir)
            
            # Run episode
            for step in range(env.max_iterations):
                rir_state, reward, terminated, info = env.step(agent)
                if terminated:
                    break
            
            # Train agent
            agent.end_episode()
            
            # Record final DRR
            final_drr = compute_rir_drr_metric(rir_state, sample_rate)
            final_drrs.append(final_drr)
            
            if episode % 10 == 0:
                print(f"  Episode {episode}: Final DRR = {final_drr:.2f} dB")
        
        # Compute average DRR over last 20 episodes
        avg_drr = np.mean(final_drrs[-20:])
        min_drr = np.min(final_drrs[-20:])
        max_drr = np.max(final_drrs[-20:])
        
        results[rt60_ms] = {
            'avg_drr': avg_drr,
            'min_drr': min_drr,
            'max_drr': max_drr
        }
        
        print(f"\n  Final 20 episodes:")
        print(f"    Avg DRR: {avg_drr:.2f} dB")
        print(f"    Min DRR: {min_drr:.2f} dB")
        print(f"    Max DRR: {max_drr:.2f} dB")
        
        if avg_drr > 0:
            print(f"    ✓ POSITIVE DRR achieved!")
        else:
            print(f"    ✗ Still negative DRR")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - DRY RIR Results")
    print("="*60)
    print(f"{'RT60 (ms)':<12} {'Avg DRR (dB)':<15} {'Status':<10}")
    print("-"*60)
    
    for rt60_ms in rt60_values:
        avg_drr = results[rt60_ms]['avg_drr']
        status = "✓ POSITIVE" if avg_drr > 0 else "✗ Negative"
        print(f"{rt60_ms:<12} {avg_drr:>14.2f} {status:<10}")
    
    print("="*60)
    
    return results

if __name__ == '__main__':
    results = quick_train()
