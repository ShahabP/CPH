#!/usr/bin/env python3
"""
Train Neural RIR Agent across different room dimensions.

Generates RIRs for different room sizes and tests the agent's ability to
estimate RIRs across various acoustic conditions. Uses multiple random seeds
for statistical robustness.
"""

import sys
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neural_rir_agent import NeuralRIRAgent, NeuralRIREnvironment, compute_rir_drr_metric


# Define different room configurations
ROOM_CONFIGS = {
    'Small Office': {
        'dimensions': (4.0, 3.5, 2.8),  # L x W x H in meters
        'rt60_ms': 200,
        'description': 'Small office/bedroom'
    },
    'Medium Room': {
        'dimensions': (6.0, 5.0, 3.0),
        'rt60_ms': 350,
        'description': 'Living room/classroom'
    },
    'Large Room': {
        'dimensions': (10.0, 8.0, 4.0),
        'rt60_ms': 500,
        'description': 'Conference room/auditorium'
    },
    'Hall': {
        'dimensions': (15.0, 12.0, 6.0),
        'rt60_ms': 700,
        'description': 'Large hall/gymnasium'
    },
    'Cathedral': {
        'dimensions': (25.0, 20.0, 10.0),
        'rt60_ms': 1200,
        'description': 'Very large space/cathedral'
    }
}


def calculate_room_volume(dimensions: Tuple[float, float, float]) -> float:
    """Calculate room volume in cubic meters."""
    return dimensions[0] * dimensions[1] * dimensions[2]


def generate_room_rir(dimensions: Tuple[float, float, float], 
                      rt60_ms: float,
                      rir_length: int,
                      sample_rate: int = 16000,
                      seed: int = None) -> np.ndarray:
    """
    Generate a realistic RIR based on room dimensions and RT60.
    
    Uses exponential decay with room-specific characteristics.
    
    Args:
        dimensions: (length, width, height) in meters
        rt60_ms: Reverberation time in milliseconds
        rir_length: Length of RIR in samples
        sample_rate: Audio sampling rate
        seed: Random seed for reproducibility
        
    Returns:
        Room impulse response
    """
    if seed is not None:
        np.random.seed(seed)
    
    rir = np.zeros(rir_length)
    
    # Direct sound at t=0
    rir[0] = 1.0
    
    # Calculate decay constant based on RT60
    decay_constant = -6.91 / (rt60_ms * sample_rate / 1000.0)
    
    # Room volume affects reflection density
    volume = calculate_room_volume(dimensions)
    reflection_probability = min(0.15, 0.05 + volume / 1000)  # More reflections in larger rooms
    
    # Generate reflections with exponential decay
    for i in range(1, rir_length):
        if np.random.random() < reflection_probability:
            # Reflection amplitude with exponential decay
            amplitude = np.random.uniform(0.1, 0.4) * np.exp(decay_constant * i)
            rir[i] = amplitude * np.random.choice([-1, 1])  # Random phase
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(rir))
    if max_val > 1.0:
        rir = rir / max_val
    
    return rir


def train_single_room(room_name: str,
                     room_config: Dict,
                     rir_length: int,
                     episodes: int,
                     seed: int) -> Dict:
    """
    Train Neural RIR agent for a single room configuration.
    
    Args:
        room_name: Name of room type
        room_config: Room configuration dict
        rir_length: RIR length in samples
        episodes: Number of training episodes
        seed: Random seed
        
    Returns:
        Training results dictionary
    """
    print(f"\n{'='*70}")
    print(f"Room: {room_name} | Seed: {seed}")
    print(f"Dimensions: {room_config['dimensions']} m | RT60: {room_config['rt60_ms']} ms")
    print(f"{'='*70}")
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    sample_rate = 16000
    rt60_ms = room_config['rt60_ms']
    dimensions = room_config['dimensions']
    
    # Create agent
    agent = NeuralRIRAgent(
        rir_length=rir_length,
        learning_rate=2e-4,
        gamma=0.99
    )
    
    # Upgrade to larger network
    from src.neural_rir_agent import RIRPolicyNetwork
    agent.policy_net = RIRPolicyNetwork(rir_length, hidden_dim=1024)
    agent.optimizer = torch.optim.Adam(
        agent.policy_net.parameters(),
        lr=2e-4,
        weight_decay=1e-5
    )
    
    # Training loop
    training_stats = []
    drr_history = []
    
    for episode in range(episodes):
        # Generate clean speech
        duration = 2.5
        t = np.linspace(0, duration, int(duration * sample_rate))
        clean = 0.2 * np.sum([
            np.sin(2 * np.pi * f * t) * np.exp(-t * 0.5)
            for f in [440, 880, 1320]
        ], axis=0)
        
        # Generate room-specific RIR
        true_rir = generate_room_rir(
            dimensions=dimensions,
            rt60_ms=rt60_ms,
            rir_length=rir_length,
            sample_rate=sample_rate,
            seed=seed + episode  # Different RIR each episode
        )
        
        # Create reverberant speech
        reverb = np.convolve(clean, true_rir, mode='same')
        
        # Initialize with exponential decay estimate
        initial_rir = agent._get_exponential_decay_rir(rir_length, rt60_ms)
        
        # Create environment and reset
        env = NeuralRIREnvironment(max_iterations=15, sample_rate=sample_rate, rir_length=rir_length)
        env.reset(reverb, clean, initial_rir=initial_rir)
        
        # Run episode
        rewards = []
        for step in range(env.max_iterations):
            rir_state, reward, terminated, info = env.step(agent)
            rewards.append(reward)
            if terminated:
                break
        
        # Train agent
        train_info = agent.end_episode()
        
        # Compute metrics
        final_rir = rir_state
        min_len = min(len(final_rir), len(true_rir))
        correlation = np.corrcoef(final_rir[:min_len], true_rir[:min_len])[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        structural_drr = compute_rir_drr_metric(final_rir, sample_rate)
        drr_history.append(structural_drr)
        
        # Log progress
        if episode % 50 == 0 or episode == episodes - 1:
            avg_drr = np.mean(drr_history[-50:]) if len(drr_history) >= 50 else np.mean(drr_history)
            print(f"Episode {episode:3d}/{episodes} | "
                  f"Reward: {np.sum(rewards):6.2f} | "
                  f"Corr: {correlation:.3f} | "
                  f"DRR: {structural_drr:6.2f} dB | "
                  f"Avg DRR (last 50): {avg_drr:6.2f} dB")
        
        training_stats.append({
            'episode': episode,
            'total_reward': np.sum(rewards),
            'steps': step + 1,
            'correlation': correlation,
            'structural_drr': structural_drr
        })
    
    # Final statistics
    final_drr = np.mean(drr_history[-50:])
    final_corr = np.mean([s['correlation'] for s in training_stats[-50:]])
    
    results = {
        'room_name': room_name,
        'room_config': room_config,
        'seed': seed,
        'rir_length': rir_length,
        'episodes': episodes,
        'training_stats': training_stats,
        'final_rir': final_rir,
        'true_rir': true_rir,
        'final_metrics': {
            'avg_drr': final_drr,
            'avg_correlation': final_corr,
            'std_drr': np.std(drr_history[-50:]),
            'min_drr': np.min(drr_history[-50:]),
            'max_drr': np.max(drr_history[-50:])
        }
    }
    
    print(f"\n{'='*70}")
    print(f"Final Results - {room_name} (Seed {seed}):")
    print(f"  Average DRR (last 50 eps): {final_drr:.2f} ± {results['final_metrics']['std_drr']:.2f} dB")
    print(f"  Average Correlation: {final_corr:.3f}")
    print(f"  DRR Range: [{results['final_metrics']['min_drr']:.2f}, {results['final_metrics']['max_drr']:.2f}] dB")
    print(f"{'='*70}\n")
    
    return results


def train_all_rooms(rir_length: int = 2048,
                   episodes: int = 300,
                   seeds: List[int] = [42, 123, 456]) -> Dict:
    """
    Train across all room configurations with multiple seeds.
    
    Args:
        rir_length: RIR length in samples
        episodes: Episodes per training
        seeds: List of random seeds
        
    Returns:
        Dictionary of all results
    """
    all_results = {}
    
    print("\n" + "="*70)
    print("NEURAL AGENT: RIR ESTIMATION ACROSS ROOM DIMENSIONS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  RIR length: {rir_length} samples ({rir_length/16000*1000:.1f} ms)")
    print(f"  Episodes per room: {episodes}")
    print(f"  Random seeds: {seeds}")
    print(f"  Number of rooms: {len(ROOM_CONFIGS)}")
    print(f"  Total trainings: {len(ROOM_CONFIGS) * len(seeds)}")
    print()
    
    for room_name, room_config in ROOM_CONFIGS.items():
        all_results[room_name] = []
        
        for seed in seeds:
            result = train_single_room(
                room_name=room_name,
                room_config=room_config,
                rir_length=rir_length,
                episodes=episodes,
                seed=seed
            )
            all_results[room_name].append(result)
    
    return all_results


def plot_room_comparison(all_results: Dict, output_dir: Path):
    """
    Generate box plots comparing performance across room dimensions.
    
    Args:
        all_results: Results dictionary from train_all_rooms
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for box plots
    room_names = list(ROOM_CONFIGS.keys())
    drr_data = []
    room_info = []
    
    for room_name in room_names:
        results = all_results[room_name]
        
        # Collect final DRR values across seeds
        drr_values = [r['final_metrics']['avg_drr'] for r in results]
        drr_data.append(drr_values)
        
        # Room info for labels
        config = ROOM_CONFIGS[room_name]
        volume = calculate_room_volume(config['dimensions'])
        room_info.append(f"{room_name}\n({volume:.0f} m³, RT60={config['rt60_ms']}ms)")
    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    
    # Box plot for DRR (clean, no colors)
    bp = ax.boxplot(drr_data, tick_labels=room_info, patch_artist=True,
                    notch=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='darkred', markersize=10, 
                                 markeredgecolor='white', markeredgewidth=1.5),
                    medianprops=dict(color='black', linewidth=2.5),
                    boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5, color='black'),
                    capprops=dict(linewidth=1.5, color='black'),
                    flierprops=dict(marker='o', markerfacecolor='gray', markersize=6, 
                                  alpha=0.5, markeredgecolor='black'))
    
    ax.set_ylabel('DRR Gain (dB)', fontsize=14, fontweight='bold')
    ax.set_title('Neural Agent: RIR Estimation Performance Across Room Dimensions', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Add mean values as text
    for i, (room_name, drr_vals) in enumerate(zip(room_names, drr_data)):
        mean_drr = np.mean(drr_vals)
        std_drr = np.std(drr_vals)
        ax.text(i+1, mean_drr + 1.5, f'{mean_drr:.1f}±{std_drr:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         edgecolor='black', alpha=0.9, linewidth=1.5))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'room_dimensions_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot to {plot_path}")
    
    # Also save to main experiments folder
    main_plot = Path('experiments') / 'room_dimensions_comparison.png'
    plt.savefig(main_plot, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {main_plot}")
    
    plt.show()


def save_results(all_results: Dict, output_dir: Path):
    """Save results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results as pickle
    with open(output_dir / 'all_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    # Save summary as JSON
    summary = {}
    for room_name, results in all_results.items():
        summary[room_name] = {
            'config': ROOM_CONFIGS[room_name],
            'volume_m3': calculate_room_volume(ROOM_CONFIGS[room_name]['dimensions']),
            'seeds': [r['seed'] for r in results],
            'drr_values': [r['final_metrics']['avg_drr'] for r in results],
            'correlation_values': [r['final_metrics']['avg_correlation'] for r in results],
            'mean_drr': np.mean([r['final_metrics']['avg_drr'] for r in results]),
            'std_drr': np.std([r['final_metrics']['avg_drr'] for r in results]),
            'mean_correlation': np.mean([r['final_metrics']['avg_correlation'] for r in results]),
            'std_correlation': np.std([r['final_metrics']['avg_correlation'] for r in results]),
        }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Saved results to {output_dir}")


def print_summary_table(all_results: Dict):
    """Print summary table of results."""
    print("\n" + "="*90)
    print("SUMMARY: NEURAL AGENT PERFORMANCE ACROSS ROOM DIMENSIONS")
    print("="*90)
    print(f"{'Room Type':<15} {'Volume (m³)':<12} {'RT60 (ms)':<10} {'Mean DRR (dB)':<15} {'Mean Correlation':<18}")
    print("-"*90)
    
    for room_name in ROOM_CONFIGS.keys():
        results = all_results[room_name]
        config = ROOM_CONFIGS[room_name]
        volume = calculate_room_volume(config['dimensions'])
        
        mean_drr = np.mean([r['final_metrics']['avg_drr'] for r in results])
        std_drr = np.std([r['final_metrics']['avg_drr'] for r in results])
        mean_corr = np.mean([r['final_metrics']['avg_correlation'] for r in results])
        std_corr = np.std([r['final_metrics']['avg_correlation'] for r in results])
        
        print(f"{room_name:<15} {volume:<12.0f} {config['rt60_ms']:<10} "
              f"{mean_drr:>6.2f} ± {std_drr:<4.2f}   {mean_corr:>5.3f} ± {std_corr:<5.3f}")
    
    print("="*90)


if __name__ == '__main__':
    # Configuration
    RIR_LENGTH = 2048
    EPISODES = 300
    SEEDS = [42, 123, 456]
    
    # Output directory
    output_dir = Path('experiments/room_dimensions')
    
    # Train across all rooms
    all_results = train_all_rooms(
        rir_length=RIR_LENGTH,
        episodes=EPISODES,
        seeds=SEEDS
    )
    
    # Save results
    save_results(all_results, output_dir)
    
    # Print summary table
    print_summary_table(all_results)
    
    # Generate plots
    plot_room_comparison(all_results, output_dir)
    
    print("\n" + "="*90)
    print("TRAINING COMPLETE!")
    print("="*90)
