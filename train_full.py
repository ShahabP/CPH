"""
Config-driven training script for RL-based RIR estimation.

Reads YAML config, initializes environment and agent, runs training,
and saves artifacts. WandB logging is optional via config.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import yaml

# Optional heavy deps
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

# Make src importable
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rl_framework import (
    RIREstimationEnv,
    StreamingRIREstimationEnv,
    DQNAgent,
    train_rir_agent,
)  # type: ignore
from neural_rir_agent import NeuralRIRAgent, NeuralRIREnvironment, compute_rir_drr_metric  # type: ignore
from audio_processing import AudioProcessor  # type: ignore
from dereverberation import BlindDereverberation  # type: ignore
from rir_estimation import DeconvolutionRIREstimator  # type: ignore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def maybe_init_wandb(config: Dict[str, Any]):
    if not config.get('use_wandb', False):
        return None
    try:
        import wandb  # type: ignore

        wandb.init(
            project=config.get('project_name', 'rir-rl-estimation'),
            config=config,
            name=config.get('experiment_name', 'rir_rl_experiment')
        )
        return wandb
    except Exception as e:
        logger.warning(f"WandB init failed or not installed: {e}")
        return None


def create_environment(config: Dict[str, Any]) -> RIREstimationEnv:
    env_cfg = config.get('environment', {})
    audio_cfg = config.get('audio_processing', {})
    dereverb_cfg = config.get('dereverberation', {})
    rir_cfg = config.get('rir_estimation', {})

    env_cls = RIREstimationEnv
    if config.get('streaming', {}).get('enabled', False):
        env_cls = StreamingRIREstimationEnv

    env = env_cls(
        max_iterations=env_cfg.get('max_iterations', 10),
        rir_length=env_cfg.get('rir_length', 1024),
        feature_dim=env_cfg.get('feature_dim', 513),
        action_space_type=env_cfg.get('action_space_type', 'continuous')
    )

    env.audio_processor = AudioProcessor(
        sample_rate=audio_cfg.get('sample_rate', 16000),
        n_fft=audio_cfg.get('n_fft', 1024),
        hop_length=audio_cfg.get('hop_length', 256)
    )

    env.dereverberation = BlindDereverberation(
        method=dereverb_cfg.get('method', 'spectral_subtraction'),
        sample_rate=audio_cfg.get('sample_rate', 16000)
    )

    env.rir_estimator = DeconvolutionRIREstimator(
        method=rir_cfg.get('method', 'wiener_deconv'),
        regularization=rir_cfg.get('regularization', 1e-3)
    )

    return env


def create_agent(env: RIREstimationEnv, config: Dict[str, Any]) -> DQNAgent:
    agent_cfg = config.get('agent', {})
    state_dim = env.observation_space.shape[0]
    if hasattr(env.action_space, 'n'):
        action_dim = env.action_space.n
    else:
        action_dim = int(agent_cfg.get('discrete_actions', 16))

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=float(agent_cfg.get('learning_rate', 1e-3)),
        gamma=float(agent_cfg.get('gamma', 0.99)),
        epsilon=float(agent_cfg.get('epsilon', 1.0)),
        epsilon_decay=float(agent_cfg.get('epsilon_decay', 0.995)),
        epsilon_min=float(agent_cfg.get('epsilon_min', 0.01)),
        memory_size=int(agent_cfg.get('memory_size', 10000)),
    )
    return agent


def main():
    parser = argparse.ArgumentParser(description='Train RL agent for RIR estimation (config-driven)')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to YAML config')
    parser.add_argument('--output-dir', type=str, default='experiments', help='Output directory for artifacts')
    parser.add_argument('--stream-dir', type=str, default=None, help='Directory containing incoming .wav segments for streaming mode')
    parser.add_argument('--watch', action='store_true', help='Continuously watch stream-dir for new files')
    parser.add_argument('--neural', action='store_true', help='Use Neural RIR agent instead of DQN')
    args = parser.parse_args()

    config = load_config(args.config)

    # Seeds
    np.random.seed(int(config.get('seed', 42)))
    if torch is not None:
        try:
            torch.manual_seed(int(config.get('seed', 42)))
        except Exception:
            pass

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # WandB (optional)
    wandb = maybe_init_wandb(config)

    # Create env and agent
    if args.neural:
        # Use Neural RIR approach
        rir_length = config.get('environment', {}).get('rir_length', 1024)
        max_iterations = config.get('environment', {}).get('max_iterations', 20)
        env = NeuralRIREnvironment(max_iterations=max_iterations, rir_length=rir_length)
        agent = NeuralRIRAgent(rir_length=rir_length, learning_rate=1e-3)
    else:
        # Use DQN approach
        env = create_environment(config)
        agent = create_agent(env, config)

    # If streaming mode enabled, process incoming segments iteratively
    stream_cfg = config.get('streaming', {})
    stream_enabled = bool(stream_cfg.get('enabled', False) or args.stream_dir)
    if stream_enabled and not args.neural:
        from glob import glob
        import time
        sr = int(config.get('audio_processing', {}).get('sample_rate', 16000))
        ap = AudioProcessor(sample_rate=sr)
        stream_dir = args.stream_dir or stream_cfg.get('stream_dir', 'data/stream')
        poll = float(stream_cfg.get('poll_interval_sec', 2.0))
        processed = set()
        out_dir = Path(args.output_dir)

        # If env is continuous and agent is DQN, build a discrete->continuous codebook
        is_cont = hasattr(env.action_space, 'shape')
        codebook = None
        if is_cont and isinstance(agent, DQNAgent):
            act_dim = int(np.prod(env.action_space.shape))
            rng = np.random.RandomState(0)
            codebook = rng.uniform(low=-1.0, high=1.0, size=(agent.action_dim, act_dim)).astype(np.float32)
            codebook[0] = 0.0

        logger.info(f"Streaming mode: watching {stream_dir} (watch={args.watch})")
        try:
            while True:
                wavs = sorted(glob(os.path.join(stream_dir, '*.wav')))
                new_files = [w for w in wavs if w not in processed]
                for w in new_files:
                    try:
                        audio, _ = ap.load_audio(w, target_sr=sr)
                        # Enqueue new segment; clean and true_rir unknown in real scenarios
                        if isinstance(env, StreamingRIREstimationEnv):
                            env.push_segment(audio, None, None)
                            state, info = env.start_next_segment()
                        else:
                            # Fallback: run a normal episode with provided reverb
                            state, info = env.reset(options={'reverb_audio': audio})

                        # Train on this segment for max_iterations
                        total_reward = 0.0
                        seg_name = Path(w).stem
                        for step in range(env.max_iterations):
                            a = agent.act(state, training=True)
                            if codebook is not None and not hasattr(a, '__len__'):
                                action = codebook[int(a) % codebook.shape[0]]
                            else:
                                action = a
                            next_state, reward, terminated, truncated, info = env.step(action)
                            agent.remember(state, action, reward, next_state, terminated or truncated)
                            state = next_state
                            total_reward += reward
                            
                            # Save per-step global RIR if available
                            if hasattr(env, 'global_rir_estimate') and env.global_rir_estimate is not None:
                                step_file = out_dir / f'{seg_name}_step_{step:02d}_rir.npy'
                                np.save(step_file, env.global_rir_estimate)
                            
                            if terminated or truncated:
                                break
                        agent.replay()

                        # Save global RIR estimate if available
                        if hasattr(env, 'global_rir_estimate') and env.global_rir_estimate is not None:
                            np.save(out_dir / (Path(w).stem + '_rir.npy'), env.global_rir_estimate)
                        processed.add(w)
                        logger.info(f"Processed segment {w}, reward={total_reward:.3f}")
                    except Exception as e:
                        logger.warning(f"Failed to process {w}: {e}")

                if not args.watch:
                    break
                time.sleep(poll)
        finally:
            pass
        stats = []
    elif stream_enabled and args.neural:
        # Neural streaming mode (simplified)
        from glob import glob
        import time
        sr = int(config.get('audio_processing', {}).get('sample_rate', 16000))
        ap = AudioProcessor(sample_rate=sr)
        stream_dir = args.stream_dir or stream_cfg.get('stream_dir', 'data/stream')
        
        logger.info(f"Neural streaming mode: {stream_dir}")
        wavs = sorted(glob(os.path.join(stream_dir, '*.wav')))
        stats = []
        
        for w in wavs[:4]:  # Process first 4 segments
            audio, _ = ap.load_audio(w, target_sr=sr)
            state = env.reset(audio)
            total_reward = 0
            seg_name = Path(w).stem
            
            for step in range(env.max_iterations):
                next_state, reward, terminated, info = env.step(agent)
                total_reward += reward
                
                # Save per-step RIR
                step_file = out_dir / f'{seg_name}_step_{step:02d}_rir_neural.npy'
                np.save(step_file, env.current_rir)
                
                if terminated:
                    break
            
            agent.end_episode()
            logger.info(f"Processed {seg_name}, reward={total_reward:.3f}")
            stats.append({'total_reward': total_reward, 'steps': step+1})
    else:
        # Non-streaming episodic training
        train_cfg = config.get('training', {})
        episodes = int(train_cfg.get('episodes', 20))
        max_steps = int(train_cfg.get('max_steps', 50))

        if args.neural:
            # Neural episodic training
            logger.info(f"Starting Neural training: episodes={episodes}, max_steps={max_steps}")
            stats = []
            
            # Generate synthetic data for training
            for episode in range(episodes):
                # Simple synthetic scenario
                sr = 16000
                t = np.linspace(0, 1.0, sr)
                clean = 0.2 * np.sin(2 * np.pi * 440 * t) * np.exp(-t * 0.5)
                
                # Random RIR for variety
                rir_len = env.rir_length
                true_rir = np.zeros(rir_len)
                true_rir[0] = 1.0
                for i in range(1, min(200, rir_len)):
                    true_rir[i] = 0.1 * np.exp(-i / 50) * np.random.randn()
                
                reverb = np.convolve(clean, true_rir, mode='same')
                
                state = env.reset(reverb, clean)
                total_reward = 0
                
                for step in range(max_steps):
                    next_state, reward, terminated, info = env.step(agent)
                    total_reward += reward
                    if terminated:
                        break
                
                agent.end_episode()
                stats.append({'total_reward': total_reward, 'steps': step+1})
                
                if episode % 5 == 0:
                    logger.info(f"Episode {episode}, Reward: {total_reward:.2f}, Steps: {step+1}")
        else:
            # DQN episodic training
            logger.info(f"Starting DQN training: episodes={episodes}, max_steps={max_steps}")
            stats = train_rir_agent(env=env, agent=agent, episodes=episodes, max_steps=max_steps)

    # Save stats
    np.savez(out_dir / 'training_results.npz', training_stats=stats)

    # Save model (if torch available)
    if torch is not None:
        try:
            import torch as _torch  # for mypy friendliness
            _torch.save({
                'q_network_state_dict': agent.q_network.state_dict(),
                'target_network_state_dict': agent.target_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'config': config,
            }, out_dir / 'final_model.pth')
        except Exception as e:
            logger.warning(f"Skipping model save: {e}")

    if wandb is not None:
        try:
            final = stats[-1]
            wandb.log({
                'final_reward': final.get('total_reward', 0.0),
                'final_steps': final.get('steps', 0),
                'final_correlation': final.get('final_metrics', {}).get('correlation', 0.0)
            })
            wandb.finish()
        except Exception as e:
            logger.warning(f"WandB logging failed: {e}")


if __name__ == '__main__':
    main()
