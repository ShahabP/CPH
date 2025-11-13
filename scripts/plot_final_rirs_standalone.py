"""
Standalone script to plot final RIRs from saved results.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_final_rirs(all_results, results_dir):
    """
    Plot the final learned RIR for each agent-initialization combination.
    
    Args:
        all_results: List of results dictionaries
        results_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Final Learned RIRs for All Methods', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    colors = {
        'QN-random': '#1f77b4',
        'QN-exponential_decay': '#aec7e8',
        'DQN-random': '#ff7f0e',
        'DQN-exponential_decay': '#ffbb78',
        'Neural-random': '#2ca02c',
        'Neural-exponential_decay': '#98df8a'
    }
    
    for idx, result in enumerate(all_results):
        ax = axes[idx]
        label = f"{result['agent_type']}-{result['init_method']}"
        
        # Get final RIR
        final_rir = result.get('final_rir', None)
        if final_rir is None:
            ax.text(0.5, 0.5, 'No RIR data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label, fontweight='bold')
            continue
        
        # Plot time domain
        time_ms = np.arange(len(final_rir)) / 16.0  # Convert samples to ms (16 kHz)
        ax.plot(time_ms, final_rir, color=colors.get(label, 'gray'), linewidth=1.5)
        
        # Add metrics to title
        final_metrics = result.get('final_metrics', {})
        corr = final_metrics.get('avg_correlation', 0.0)
        reward = final_metrics.get('avg_reward', 0.0)
        
        ax.set_title(f"{label}\nCorr: {corr:.3f}, Reward: {reward:.2f}", fontweight='bold', fontsize=10)
        ax.set_xlabel('Time (ms)', fontweight='bold')
        ax.set_ylabel('Amplitude', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Mark the decay envelope
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'final_rirs_all_methods.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved final RIRs plot: {results_dir / 'final_rirs_all_methods.png'}")
    plt.close()


if __name__ == '__main__':
    results_dir = Path('experiments/all_combinations')
    
    # Load results
    with open(results_dir / 'all_results.pkl', 'rb') as f:
        all_results = pickle.load(f)
    
    print(f"Loaded {len(all_results)} results")
    
    # Check if final_rir is present
    for result in all_results:
        label = f"{result['agent_type']}-{result['init_method']}"
        has_rir = 'final_rir' in result
        if has_rir:
            rir_shape = result['final_rir'].shape
            print(f"{label}: RIR shape = {rir_shape}")
        else:
            print(f"{label}: No final_rir found!")
    
    # Plot final RIRs
    plot_final_rirs(all_results, results_dir)
    print("\nDone!")
