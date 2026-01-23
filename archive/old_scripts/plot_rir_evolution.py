#!/usr/bin/env python3
"""
Plot RIR evolution during Neural agent training with exponential initialization.
Saves RIR estimates at checkpoints (every 50 episodes) and creates visualization.
"""

import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neural_rir_agent import NeuralRIRAgent, NeuralRIREnvironment, compute_rir_drr_metric


def train_with_checkpoints(episodes: int = 300, rir_length: int = 2048, 
                           rt60_ms: float = 400.0, checkpoint_interval: int = 50):
    """
    Train Neural RIR agent and save RIR estimates at regular intervals.
    
    Args:
        episodes: Total number of training episodes
        rir_length: Length of RIR in samples
        rt60_ms: RT60 reverberation time in milliseconds
        checkpoint_interval: Save RIR every N episodes
        
    Returns:
        Dictionary with checkpointed RIRs and metadata
    """
    print(f"\nTraining Neural Agent with Exponential Initialization")
    print(f"  RT60: {rt60_ms} ms")
    print(f"  RIR length: {rir_length} samples ({rir_length/16:.0f} ms)")
    print(f"  Episodes: {episodes}")
    print(f"  Checkpoint interval: every {checkpoint_interval} episodes\n")
    
    # Create agent
    agent = NeuralRIRAgent(
        rir_length=rir_length,
        learning_rate=2e-4,
        gamma=0.99
    )
    
    # Upgrade policy network
    from src.neural_rir_agent import RIRPolicyNetwork
    agent.policy_net = RIRPolicyNetwork(rir_length, hidden_dim=1024)
    agent.optimizer = torch.optim.Adam(
        agent.policy_net.parameters(), 
        lr=2e-4, 
        weight_decay=1e-5
    )
    
    # Training setup
    sample_rate = 16000
    
    # Generate true RIR once (same for all episodes)
    true_rir = agent._get_exponential_decay_rir(rir_length, rt60_ms)
    
    # Storage for checkpoints
    checkpoints = {
        'episodes': [],
        'rirs': [],
        'drrs': [],
        'true_rir': true_rir,
        'initial_rir': None,
        'rt60_ms': rt60_ms,
        'rir_length': rir_length,
        'sample_rate': sample_rate
    }
    
    # Training loop
    for episode in range(episodes):
        # Generate synthetic reverberant speech
        duration = 2.5
        t = np.linspace(0, duration, int(duration * sample_rate))
        clean = 0.2 * np.sum([
            np.sin(2 * np.pi * f * t) * np.exp(-t * 0.5)
            for f in [440, 880, 1320]
        ], axis=0)
        
        reverb = np.convolve(clean, true_rir, mode='same')
        
        # Initialize RIR with exponential decay
        initial_rir = agent._get_exponential_decay_rir(rir_length, rt60_ms)
        
        # Save initial RIR from first episode
        if episode == 0:
            checkpoints['initial_rir'] = initial_rir.copy()
            checkpoints['episodes'].append(0)
            checkpoints['rirs'].append(initial_rir.copy())
            try:
                drr = compute_rir_drr_metric(initial_rir, sample_rate=sample_rate)
            except:
                drr = 0.0
            checkpoints['drrs'].append(drr)
            print(f"Episode 0 (Initial): DRR = {drr:.2f} dB")
        
        # Create environment and reset
        env = NeuralRIREnvironment(max_iterations=15, sample_rate=sample_rate, rir_length=rir_length)
        env.reset(reverb, clean, initial_rir=initial_rir)
        
        # Run episode
        for step in range(env.max_iterations):
            rir_state, reward, terminated, info = env.step(agent)
            if terminated:
                break
        
        # Train agent
        agent.end_episode()
        
        # Save checkpoint
        if (episode + 1) % checkpoint_interval == 0 or episode == episodes - 1:
            final_rir = env.current_rir
            try:
                drr = compute_rir_drr_metric(final_rir, sample_rate=sample_rate)
            except Exception as e:
                drr = 0.0
            
            checkpoints['episodes'].append(episode + 1)
            checkpoints['rirs'].append(final_rir.copy())
            checkpoints['drrs'].append(drr)
            
            print(f"Episode {episode + 1:3d}: DRR = {drr:.2f} dB")
    
    print(f"\nTraining complete! Saved {len(checkpoints['episodes'])} checkpoints.")
    
    return checkpoints


def plot_rir_evolution(checkpoints, save_path=None):
    """
    Plot RIR evolution across training episodes.
    
    Args:
        checkpoints: Dictionary with RIR checkpoints
        save_path: Optional path to save the figure
    """
    episodes = checkpoints['episodes']
    rirs = checkpoints['rirs']
    drrs = checkpoints['drrs']
    true_rir = checkpoints['true_rir']
    sample_rate = checkpoints['sample_rate']
    rir_length = checkpoints['rir_length']
    rt60_ms = checkpoints['rt60_ms']
    
    n_checkpoints = len(episodes)
    
    # Create figure with single row for energy decay curves
    fig, axes = plt.subplots(1, n_checkpoints, figsize=(18, 4))
    
    # Time axis in milliseconds
    time_ms = np.arange(rir_length) / sample_rate * 1000
    
    # Plot each checkpoint - energy decay only
    for idx, (ep, rir, drr) in enumerate(zip(episodes, rirs, drrs)):
        ax = axes[idx] if n_checkpoints > 1 else axes
        
        # Compute energy envelope
        energy = rir**2
        # Smooth with moving average
        window_size = max(1, rir_length // 100)
        energy_smooth = np.convolve(energy, np.ones(window_size)/window_size, mode='same')
        energy_db = 10 * np.log10(energy_smooth + 1e-10)
        
        ax.plot(time_ms, energy_db, 'g-', linewidth=1.5, alpha=0.8)
        ax.set_title(f'Episode {ep}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (ms)', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Energy (dB)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
    
    # Overall title
    fig.suptitle(f'RIR Energy Decay Evolution - Neural Agent (Exponential Initialization)\n' +
                f'RT60: {rt60_ms} ms, RIR Length: {rir_length} samples ({rir_length/16:.0f} ms)',
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    return fig


def main():
    """Run training with checkpoints and create visualization."""
    
    # Configuration
    episodes = 300
    rir_length = 2048
    rt60_ms = 400.0
    checkpoint_interval = 50
    
    # Create output directory
    output_dir = Path("experiments/rir_evolution")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train with checkpoints
    checkpoints = train_with_checkpoints(
        episodes=episodes,
        rir_length=rir_length,
        rt60_ms=rt60_ms,
        checkpoint_interval=checkpoint_interval
    )
    
    # Save checkpoints
    checkpoint_file = output_dir / "rir_checkpoints.pkl"
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoints, f)
    print(f"\nCheckpoints saved to: {checkpoint_file}")
    
    # Create visualization
    plot_path = output_dir / "rir_evolution.png"
    fig = plot_rir_evolution(checkpoints, save_path=plot_path)
    
    # Show plot
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("Training Summary:")
    print("="*60)
    print(f"{'Episode':<10} {'DRR (dB)':<12}")
    print("-"*22)
    for ep, drr in zip(checkpoints['episodes'], checkpoints['drrs']):
        print(f"{ep:<10} {drr:>10.2f}")
    print("="*60)


if __name__ == '__main__':
    main()
