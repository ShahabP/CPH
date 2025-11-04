"""Evaluation script for trained RIR estimation models."""

import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, Any, List

from src.rl_framework import RIREstimationEnv, DQNAgent
from src.audio_processing import AudioProcessor
from src.rir_estimation import evaluate_rir_quality

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path: str, config: Dict[str, Any]) -> DQNAgent:
    """Load trained model."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create agent
    state_dim = config['agent']['state_dim'] if 'state_dim' in config['agent'] else 1000  # fallback
    action_dim = config['agent']['action_dim'] if 'action_dim' in config['agent'] else 64  # fallback
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=config['agent'].get('learning_rate', 1e-3),
        gamma=config['agent'].get('gamma', 0.99),
        epsilon=0.0  # No exploration during evaluation
    )
    
    # Load weights
    agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
    
    return agent


def evaluate_model(env: RIREstimationEnv, agent: DQNAgent, 
                  num_episodes: int = 10) -> List[Dict[str, Any]]:
    """Evaluate trained model."""
    evaluation_results = []
    
    for episode in range(num_episodes):
        logger.info(f"Evaluating episode {episode + 1}/{num_episodes}")
        
        # Reset environment
        state, info = env.reset()
        episode_history = []
        total_reward = 0.0
        
        # Run episode
        for step in range(env.max_iterations):
            action = agent.act(state, training=False)  # No exploration
            next_state, reward, terminated, truncated, info = env.step(action)
            
            episode_history.append({
                'step': step,
                'reward': reward,
                'metrics': info['metrics'].copy(),
                'improvement': info.get('improvement', {}).copy()
            })
            
            state = next_state
            total_reward += reward
            
            if terminated or truncated:
                break
        
        # Store episode results
        final_metrics = episode_history[-1]['metrics'] if episode_history else {}
        episode_result = {
            'episode': episode,
            'total_reward': total_reward,
            'final_metrics': final_metrics,
            'history': episode_history,
            'convergence_step': len(episode_history)
        }
        
        evaluation_results.append(episode_result)
        
        logger.info(f"Episode {episode}: Reward={total_reward:.2f}, "
                   f"Final Correlation={final_metrics.get('correlation', 0.0):.3f}")
    
    return evaluation_results


def generate_evaluation_report(results: List[Dict[str, Any]], 
                             output_dir: Path):
    """Generate comprehensive evaluation report."""
    # Compute summary statistics
    total_rewards = [r['total_reward'] for r in results]
    final_correlations = [r['final_metrics'].get('correlation', 0.0) for r in results]
    final_mses = [r['final_metrics'].get('mse', float('inf')) for r in results]
    convergence_steps = [r['convergence_step'] for r in results]
    
    summary_stats = {
        'num_episodes': len(results),
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'avg_correlation': np.mean(final_correlations),
        'std_correlation': np.std(final_correlations),
        'avg_mse': np.mean(final_mses),
        'std_mse': np.std(final_mses),
        'avg_convergence_steps': np.mean(convergence_steps),
        'std_convergence_steps': np.std(convergence_steps)
    }
    
    logger.info("Evaluation Summary:")
    logger.info(f"  Average Reward: {summary_stats['avg_reward']:.3f} ± {summary_stats['std_reward']:.3f}")
    logger.info(f"  Average Correlation: {summary_stats['avg_correlation']:.3f} ± {summary_stats['std_correlation']:.3f}")
    logger.info(f"  Average MSE: {summary_stats['avg_mse']:.6f} ± {summary_stats['std_mse']:.6f}")
    logger.info(f"  Average Convergence Steps: {summary_stats['avg_convergence_steps']:.1f} ± {summary_stats['std_convergence_steps']:.1f}")
    
    # Save summary statistics
    np.savez(output_dir / 'evaluation_summary.npz', **summary_stats)
    
    # Plot results
    create_evaluation_plots(results, output_dir)
    
    return summary_stats


def create_evaluation_plots(results: List[Dict[str, Any]], output_dir: Path):
    """Create evaluation plots."""
    plt.style.use('default')
    
    # Plot 1: Reward vs Episode
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    rewards = [r['total_reward'] for r in results]
    plt.plot(rewards, 'b-', marker='o', markersize=4)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Final Correlation vs Episode
    plt.subplot(2, 2, 2)
    correlations = [r['final_metrics'].get('correlation', 0.0) for r in results]
    plt.plot(correlations, 'r-', marker='s', markersize=4)
    plt.title('Final Correlation per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Correlation')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Convergence Steps
    plt.subplot(2, 2, 3)
    steps = [r['convergence_step'] for r in results]
    plt.bar(range(len(steps)), steps, alpha=0.7, color='green')
    plt.title('Convergence Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps to Convergence')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Learning Progress (example for first episode)
    plt.subplot(2, 2, 4)
    if results and 'history' in results[0]:
        history = results[0]['history']
        steps = [h['step'] for h in history]
        correlations = [h['metrics'].get('correlation', 0.0) for h in history]
        plt.plot(steps, correlations, 'purple', marker='d', markersize=4)
        plt.title('Learning Progress (Episode 1)')
        plt.xlabel('Step')
        plt.ylabel('Correlation')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Distribution plots
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist([r['total_reward'] for r in results], bins=10, alpha=0.7, color='blue')
    plt.title('Distribution of Total Rewards')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist([r['final_metrics'].get('correlation', 0.0) for r in results], 
             bins=10, alpha=0.7, color='red')
    plt.title('Distribution of Final Correlations')
    plt.xlabel('Final Correlation')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Evaluation plots saved to {output_dir}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained RIR estimation model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='evaluation_results/',
                       help='Path to output directory')
    parser.add_argument('--num-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--save-audio', action='store_true',
                       help='Save audio samples during evaluation')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading model and creating environment...")
    
    # Load model
    agent = load_model(args.model, config)
    
    # Create environment
    from train import create_environment  # Import from training script
    env = create_environment(config)
    
    logger.info(f"Starting evaluation with {args.num_episodes} episodes...")
    
    # Evaluate model
    results = evaluate_model(env, agent, args.num_episodes)
    
    # Generate report
    summary_stats = generate_evaluation_report(results, output_dir)
    
    # Save detailed results
    np.savez(output_dir / 'detailed_results.npz', results=results)
    
    logger.info(f"Evaluation completed. Results saved to {output_dir}")


if __name__ == '__main__':
    main()