"""
Quick test of updated neural agent with dry RIR bias.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.neural_rir_agent import NeuralRIRAgent, NeuralRIREnvironment, compute_rir_drr_metric

def test_dry_rir():
    """Test neural agent with dry RIR optimization."""
    
    print("="*60)
    print("Testing Neural RIR Agent with DRY optimization")
    print("="*60)
    
    # Parameters
    rir_length = 6400  # 400ms @ 16kHz
    rt60_ms = 300.0
    sample_rate = 16000
    
    # Create agent
    agent = NeuralRIRAgent(
        rir_length=rir_length,
        learning_rate=3e-4,
        gamma=0.95,
        update_scale=0.05
    )
    
    # Generate test exponential decay RIR
    initial_rir = agent._get_exponential_decay_rir(rir_length, rt60_ms)
    
    print(f"\nInitial RIR (exponential_decay, RT60={rt60_ms}ms):")
    print(f"  Direct sound: {initial_rir[0]:.4f}")
    print(f"  Early energy (1-320): {np.sum(initial_rir[1:320]**2):.6f}")
    print(f"  Late energy (320-1600): {np.sum(initial_rir[320:1600]**2):.6f}")
    print(f"  Tail energy (1600+): {np.sum(initial_rir[1600:]**2):.6f}")
    
    drr_init = compute_rir_drr_metric(initial_rir, sample_rate)
    print(f"  Structural DRR: {drr_init:.2f} dB")
    
    # Generate synthetic test data
    duration = 2.0
    t = np.linspace(0, duration, int(duration * sample_rate))
    clean = 0.2 * np.sum([
        np.sin(2 * np.pi * f * t) * np.exp(-t * 0.5)
        for f in [440, 880, 1320]
    ], axis=0)
    
    # True RIR (also dry)
    true_rir = agent._get_exponential_decay_rir(rir_length, rt60_ms)
    reverb = np.convolve(clean, true_rir, mode='same')
    
    # Create environment
    env = NeuralRIREnvironment(max_iterations=10, sample_rate=sample_rate, rir_length=rir_length)
    env.reset(reverb, clean, initial_rir=initial_rir)
    
    # Run a few steps
    print(f"\nRunning {env.max_iterations} training steps...")
    rewards = []
    for step in range(env.max_iterations):
        rir_state, reward, terminated, info = env.step(agent)
        rewards.append(reward)
        
        if step % 3 == 0:
            drr_step = compute_rir_drr_metric(rir_state, sample_rate)
            print(f"  Step {step}: Reward={reward:.2f}, Structural DRR={drr_step:.2f} dB")
        
        if terminated:
            break
    
    # Final RIR
    final_rir = rir_state
    drr_final = compute_rir_drr_metric(final_rir, sample_rate)
    
    print(f"\nFinal RIR:")
    print(f"  Direct sound: {final_rir[0]:.4f}")
    print(f"  Early energy (1-320): {np.sum(final_rir[1:320]**2):.6f}")
    print(f"  Late energy (320-1600): {np.sum(final_rir[320:1600]**2):.6f}")
    print(f"  Tail energy (1600+): {np.sum(final_rir[1600:]**2):.6f}")
    print(f"  Structural DRR: {drr_final:.2f} dB")
    
    print(f"\nTotal reward: {sum(rewards):.2f}")
    print(f"DRR improvement: {drr_final - drr_init:.2f} dB")
    
    if drr_final > 0:
        print(f"\n✓ SUCCESS: Achieved POSITIVE DRR ({drr_final:.2f} dB)")
    else:
        print(f"\n✗ Still negative DRR ({drr_final:.2f} dB) - may need more training")
    
    return drr_final

if __name__ == '__main__':
    final_drr = test_dry_rir()
