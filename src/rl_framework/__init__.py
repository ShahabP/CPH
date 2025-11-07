"""
Reinforcement Learning Framework for RIR Estimation

This module implements the RL environment, agent, and training loop for
iterative improvement of room impulse response estimation.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Any, Optional, List
from collections import deque
import random
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RIREstimationState:
    """State representation for RIR estimation environment."""
    reverb_features: np.ndarray
    dereverberated_features: np.ndarray
    current_rir_estimate: np.ndarray
    estimation_history: List[np.ndarray]
    iteration: int
    quality_metrics: Dict[str, float]


class RIREstimationEnv(gym.Env):
    """
    Gymnasium environment for reinforcement learning based RIR estimation.
    
    The environment provides:
    - State: Current audio features and RIR estimate
    - Action: Parameters for dereverberation and RIR estimation algorithms  
    - Reward: Based on RIR estimation quality and improvement
    """
    
    def __init__(self, 
                 max_iterations: int = 10,
                 rir_length: int = 1024,
                 feature_dim: int = 513,
                 action_space_type: str = 'continuous',
                 rir_init_method: str = 'random'):
        """
        Initialize RIR estimation environment.
        
        Args:
            max_iterations: Maximum iterations per episode
            rir_length: Length of RIR to estimate
            feature_dim: Dimension of audio features
            action_space_type: Type of action space ('continuous' or 'discrete')
            rir_init_method: RIR initialization method ('random' or 'exponential_decay')
        """
        super().__init__()
        
        self.max_iterations = max_iterations
        self.rir_length = rir_length
        self.feature_dim = feature_dim
        self.rir_init_method = rir_init_method
        
        # State space: [reverb_features, dereverb_features, current_rir, metrics]
        state_dim = 2 * feature_dim + rir_length + 10  # +10 for metrics
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # Action space: parameters for algorithms
        if action_space_type == 'continuous':
            # Continuous actions: [dereverb_params, rir_estimation_params]
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(6,), dtype=np.float32
            )
            # Actions: [alpha, beta, step_size, regularization, method_weight_1, method_weight_2]
        else:
            # Discrete actions: algorithm choices and parameter settings
            self.action_space = spaces.MultiDiscrete([3, 3, 5, 5])  # method choices and param levels
        
        # Initialize components (would be injected in practice)
        self.audio_processor = None
        self.dereverberation = None
        self.rir_estimator = None
        
        # Episode state
        self.current_state = None
        self.iteration = 0
        self.episode_history = []
        # Streaming support fields (used by subclass too)
        self._stream_queue = []  # list of tuples: (reverb, clean, true_rir)
        self.global_rir_estimate = None  # persists across segments
        
    def reset(self, seed: Optional[int] = None, 
             options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment for new episode.
        
        Args:
            seed: Random seed
            options: Additional options including audio data
            
        Returns:
            Tuple of (initial_observation, info)
        """
        super().reset(seed=seed)
        
        if options and 'reverb_audio' in options:
            self.reverb_audio = options['reverb_audio']
            self.clean_audio = options.get('clean_audio', None)
            self.true_rir = options.get('true_rir', None)
        elif self._stream_queue:
            # Pop next streaming segment if available
            seg = self._stream_queue.pop(0)
            self.reverb_audio, self.clean_audio, self.true_rir = seg
        else:
            # Generate synthetic data for training
            self.reverb_audio, self.clean_audio, self.true_rir = self._generate_synthetic_data()
        
        # Extract initial features
        reverb_features = self._extract_features(self.reverb_audio)
        
        # Initialize with basic dereverberation
        initial_dereverb = self._apply_basic_dereverberation(self.reverb_audio)
        dereverb_features = self._extract_features(initial_dereverb)
        
        # Initialize RIR estimate
        # Use global estimate if available to warm-start new segment
        if self.global_rir_estimate is not None:
            initial_rir = self.global_rir_estimate.copy()
        else:
            initial_rir = self._get_initial_rir_estimate(self.reverb_audio, initial_dereverb, self.rir_init_method)
        
        # Compute initial quality metrics
        initial_metrics = self._compute_quality_metrics(initial_rir)
        
        # Create initial state
        self.current_state = RIREstimationState(
            reverb_features=reverb_features,
            dereverberated_features=dereverb_features,
            current_rir_estimate=initial_rir,
            estimation_history=[initial_rir.copy()],
            iteration=0,
            quality_metrics=initial_metrics
        )
        
        self.iteration = 0
        self.episode_history = [initial_metrics.copy()]
        
        observation = self._get_observation()
        info = {'metrics': initial_metrics, 'iteration': 0}
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.iteration += 1
        
        # Parse action to algorithm parameters
        dereverb_params, rir_params = self._parse_action(action)
        
        # Apply dereverberation with new parameters
        enhanced_audio = self._apply_dereverberation(
            self.reverb_audio, dereverb_params
        )
        enhanced_features = self._extract_features(enhanced_audio)
        
        # Estimate RIR with new parameters
        new_rir = self._estimate_rir(
            self.reverb_audio, enhanced_audio, rir_params
        )
        
        # Compute quality metrics
        new_metrics = self._compute_quality_metrics(new_rir)
        
        # Update state
        self.current_state.dereverberated_features = enhanced_features
        self.current_state.current_rir_estimate = new_rir
        self.current_state.estimation_history.append(new_rir.copy())
        self.current_state.iteration = self.iteration
        self.current_state.quality_metrics = new_metrics
        
        # Compute reward
        reward = self._compute_reward(new_metrics)
        
        # Check termination conditions
        terminated = self._is_terminated(new_metrics)
        truncated = self.iteration >= self.max_iterations
        
        # Store episode history
        self.episode_history.append(new_metrics.copy())
        
        observation = self._get_observation()
        info = {
            'metrics': new_metrics,
            'iteration': self.iteration,
            'improvement': self._compute_improvement()
        }
        
        return observation, reward, terminated, truncated, info
    
    def _generate_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic reverberant audio for training."""
        # Generate clean speech signal (placeholder)
        duration = 3.0  # seconds
        sample_rate = 16000
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Simple synthetic speech-like signal
        clean_audio = np.sum([
            np.sin(2 * np.pi * f * t) * np.exp(-t * 0.5)
            for f in [440, 880, 1320]
        ], axis=0) * 0.1
        
        # Generate synthetic RIR
        rir_length = 1024
        true_rir = np.zeros(rir_length)
        true_rir[0] = 1.0  # Direct path
        
        # Add reflections with exponential decay
        decay_times = [100, 200, 300, 500]
        amplitudes = [0.3, 0.2, 0.1, 0.05]
        for delay, amp in zip(decay_times, amplitudes):
            if delay < rir_length:
                true_rir[delay] = amp * np.exp(-delay / 200)
        
        # Convolve to create reverberant audio
        reverb_audio = np.convolve(clean_audio, true_rir, mode='same')
        
        return reverb_audio, clean_audio, true_rir
    
    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract features from audio signal."""
        # Placeholder: compute magnitude spectrum
        from scipy import signal as scipy_signal
        f, t, Zxx = scipy_signal.stft(audio, nperseg=1024, noverlap=512)
        magnitude = np.abs(Zxx)
        
        # Global pooling for fixed-size representation
        features = np.mean(magnitude, axis=1)
        
        return features
    
    def _apply_basic_dereverberation(self, audio: np.ndarray) -> np.ndarray:
        """Apply basic dereverberation as initial processing."""
        # Placeholder: simple spectral subtraction
        return audio * 0.8  # Simple scaling as placeholder
    
    def _get_initial_rir_estimate(self, reverb_audio: np.ndarray, 
                                 clean_audio: np.ndarray,
                                 init_method: str = "random") -> np.ndarray:
        """Get initial RIR estimate using specified method.
        
        Args:
            reverb_audio: Reverberant audio signal
            clean_audio: Clean audio signal (if available)
            init_method: Initialization method ("random" or "exponential_decay")
            
        Returns:
            Initial RIR estimate
        """
        if init_method == "exponential_decay":
            return self._get_exponential_decay_rir()
        else:  # default to random
            return np.random.normal(0, 0.1, self.rir_length)
    
    def _get_exponential_decay_rir(self) -> np.ndarray:
        """Generate exponential decay RIR initialization.
        
        Creates a physically plausible RIR with:
        - Strong direct sound at t=0
        - Exponential decay following typical room acoustics
        - Realistic decay constants for early and late reflections
        """
        rir = np.zeros(self.rir_length)
        
        # Direct sound (strong impulse at t=0)
        rir[0] = 1.0
        
        # Early reflections (first 200 samples ~12.5ms)
        early_decay = 200  # samples
        for i in range(1, min(early_decay, self.rir_length)):
            # Add some early reflections with decreasing amplitude
            reflection_strength = 0.3 * np.exp(-i / 100)  # Fast initial decay
            if np.random.random() < 0.1:  # Sparse early reflections
                rir[i] += reflection_strength * (0.5 + np.random.random())
        
        # Late reverberation (exponential tail)
        for i in range(early_decay, self.rir_length):
            # Exponential decay with realistic RT60 characteristics
            decay_rate = 50  # samples (faster decay = shorter RT60)
            amplitude = 0.1 * np.exp(-i / decay_rate)
            rir[i] = amplitude * (0.8 + 0.4 * np.random.random())  # Add some variation
        
        # Normalize to ensure direct sound is prominent
        if np.max(np.abs(rir)) > 0:
            rir = rir / np.max(np.abs(rir))
        
        return rir
    
    def _compute_quality_metrics(self, rir_estimate: np.ndarray) -> Dict[str, float]:
        """Compute RIR quality metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['peak_amplitude'] = float(np.max(np.abs(rir_estimate)))
        metrics['energy'] = float(np.sum(rir_estimate ** 2))
        metrics['sparsity'] = float(np.sum(np.abs(rir_estimate) > 0.01) / len(rir_estimate))
        
        # If ground truth is available
        if hasattr(self, 'true_rir') and self.true_rir is not None:
            min_len = min(len(rir_estimate), len(self.true_rir))
            correlation = np.corrcoef(
                rir_estimate[:min_len], self.true_rir[:min_len]
            )[0, 1]
            metrics['correlation'] = float(correlation if not np.isnan(correlation) else 0.0)
            
            mse = np.mean((rir_estimate[:min_len] - self.true_rir[:min_len]) ** 2)
            metrics['mse'] = float(mse)
        
        return metrics
    
    def _parse_action(self, action: np.ndarray) -> Tuple[Dict, Dict]:
        """Parse action into algorithm parameters."""
        dereverb_params = {
            'alpha': float(2.0 + action[0]),  # Spectral subtraction factor
            'beta': float(0.01 + 0.09 * (action[1] + 1) / 2),  # Spectral floor
        }
        
        rir_params = {
            'regularization': float(1e-4 + 1e-2 * (action[2] + 1) / 2),
            'step_size': float(1e-4 + 1e-2 * (action[3] + 1) / 2),
            'method_weights': {
                'wiener': float((action[4] + 1) / 2),
                'lms': float((action[5] + 1) / 2)
            }
        }
        
        return dereverb_params, rir_params
    
    def _apply_dereverberation(self, audio: np.ndarray, 
                             params: Dict) -> np.ndarray:
        """Apply dereverberation with given parameters."""
        # Placeholder implementation
        return audio * (1.0 - params['alpha'] * 0.1)
    
    def _estimate_rir(self, reverb_audio: np.ndarray, 
                     clean_audio: np.ndarray, params: Dict) -> np.ndarray:
        """Estimate RIR with given parameters."""
        # Placeholder: simple correlation-based estimate
        correlation = np.correlate(reverb_audio[:self.rir_length], 
                                 clean_audio[:self.rir_length], mode='full')
        center = len(correlation) // 2
        rir = correlation[center:center + self.rir_length]
        
        # Normalize
        rir = rir / (np.max(np.abs(rir)) + 1e-10)
        
        # Optional temporal smoothing with previous/global estimate to support streaming refinement
        prev = getattr(self, 'global_rir_estimate', None)
        alpha = float(params.get('blend', 0.3)) if isinstance(params, dict) else 0.3
        if prev is not None and len(prev) == len(rir):
            rir = (1 - alpha) * prev + alpha * rir

        # Update global estimate for use in subsequent steps/segments
        self.global_rir_estimate = rir.copy()

        return rir
    
    def _compute_reward(self, metrics: Dict[str, float]) -> float:
        """Compute reward based on quality metrics."""
        reward = 0.0
        
        # Reward for high correlation with ground truth
        if 'correlation' in metrics:
            reward += 10.0 * max(0, metrics['correlation'])
        
        # Penalty for high MSE
        if 'mse' in metrics:
            reward -= metrics['mse']
        
        # Reward for reasonable peak amplitude
        peak_reward = 1.0 - abs(metrics['peak_amplitude'] - 1.0)
        reward += peak_reward
        
        # Reward for improvement over previous iteration
        if len(self.episode_history) > 0:
            prev_correlation = self.episode_history[-1].get('correlation', 0.0)
            current_correlation = metrics.get('correlation', 0.0)
            improvement = current_correlation - prev_correlation
            reward += 5.0 * improvement
        
        return float(reward)
    
    def _is_terminated(self, metrics: Dict[str, float]) -> bool:
        """Check if episode should terminate early."""
        # Terminate if correlation is very high
        if 'correlation' in metrics and metrics['correlation'] > 0.95:
            return True
        
        # Terminate if no improvement for several iterations
        if len(self.episode_history) >= 3:
            recent_correlations = [
                ep.get('correlation', 0.0) for ep in self.episode_history[-3:]
            ]
            if max(recent_correlations) - min(recent_correlations) < 0.01:
                return True
        
        return False
    
    def _compute_improvement(self) -> Dict[str, float]:
        """Compute improvement metrics."""
        if len(self.episode_history) < 2:
            return {}
        
        improvement = {}
        current = self.episode_history[-1]
        initial = self.episode_history[0]
        
        for key in current:
            if key in initial:
                if key == 'mse':  # Lower is better
                    improvement[f'{key}_improvement'] = initial[key] - current[key]
                else:  # Higher is better
                    improvement[f'{key}_improvement'] = current[key] - initial[key]
        
        return improvement
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        state = self.current_state
        
        # Concatenate all state components
        obs_components = [
            state.reverb_features,
            state.dereverberated_features,
            state.current_rir_estimate,
            np.array([
                state.iteration / self.max_iterations,
                state.quality_metrics.get('peak_amplitude', 0.0),
                state.quality_metrics.get('energy', 0.0),
                state.quality_metrics.get('sparsity', 0.0),
                state.quality_metrics.get('correlation', 0.0),
                state.quality_metrics.get('mse', 0.0),
                len(state.estimation_history) / self.max_iterations,
            ])
        ]
        
        observation = np.concatenate(obs_components).astype(np.float32)
        
        # Pad to fixed size if necessary
        target_size = self.observation_space.shape[0]
        if len(observation) < target_size:
            observation = np.pad(observation, (0, target_size - len(observation)))
        elif len(observation) > target_size:
            observation = observation[:target_size]
        
        return observation


class StreamingRIREstimationEnv(RIREstimationEnv):
    """Environment extension that supports streaming new speech segments.

    - push_segment(): enqueue new (reverb, clean, true_rir) tuples
    - start_next_segment(): begin processing next segment while retaining
      the global RIR estimate to refine iteratively across segments
    """

    def push_segment(self, reverb_audio: np.ndarray,
                     clean_audio: Optional[np.ndarray] = None,
                     true_rir: Optional[np.ndarray] = None) -> None:
        self._stream_queue.append((reverb_audio, clean_audio, true_rir))

    def start_next_segment(self) -> Tuple[np.ndarray, Dict]:
        if not self._stream_queue:
            raise RuntimeError("No streaming segments available. Call push_segment() first.")
        return self.reset()


class DQNAgent:
    """Deep Q-Network agent for RIR estimation."""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 1e-3, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, memory_size: int = 10000):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension  
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon
            memory_size: Replay buffer size
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Neural networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Update target network
        self.update_target_network()
        
    def _build_network(self) -> nn.Module:
        """Build Q-network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return int(q_values.argmax().item())
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size: int = 32):
        """Train the network on a batch of experiences."""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network with current network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())


def _build_action_codebook(num_actions: int, action_dim: int) -> np.ndarray:
    """Deterministic codebook for mapping discrete actions to continuous vectors."""
    rng = np.random.RandomState(0)
    codebook = rng.uniform(low=-1.0, high=1.0, size=(num_actions, action_dim)).astype(np.float32)
    # Ensure first few are meaningful anchors
    anchors = [
        np.zeros(action_dim, dtype=np.float32),
        np.full(action_dim, -0.5, dtype=np.float32),
        np.full(action_dim, 0.5, dtype=np.float32),
    ]
    for i, v in enumerate(anchors):
        if i < num_actions:
            codebook[i] = v
    return codebook


def train_rir_agent(env: RIREstimationEnv, agent: DQNAgent, 
                   episodes: int = 1000, max_steps: int = 200) -> List[Dict]:
    """
    Train RL agent for RIR estimation.
    
    Args:
        env: RIR estimation environment
        agent: DQN agent
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        
    Returns:
        List of training statistics
    """
    training_stats = []
    
    # If env is continuous, prepare a codebook to map discrete actions
    is_continuous = hasattr(env.action_space, 'shape')
    if is_continuous:
        action_size = getattr(agent, 'action_dim', 8)
        act_dim = int(np.prod(env.action_space.shape))
        codebook = _build_action_codebook(action_size, act_dim)

    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0.0
        steps = 0
        
        for step in range(max_steps):
            action = agent.act(state, training=True)
            if is_continuous and not isinstance(action, np.ndarray):
                idx = int(action) % codebook.shape[0]
                action_vec = codebook[idx]
            else:
                action_vec = action
            next_state, reward, terminated, truncated, info = env.step(action_vec)
            
            agent.remember(state, action, reward, next_state, terminated or truncated)
            state = next_state
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        # Train agent
        agent.replay()
        
        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Log statistics
        episode_stats = {
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'epsilon': agent.epsilon,
            'final_metrics': info.get('metrics', {}),
            'improvement': info.get('improvement', {})
        }
        training_stats.append(episode_stats)
        
        if episode % 100 == 0:
            logger.info(f"Episode {episode}, Reward: {total_reward:.2f}, "
                       f"Steps: {steps}, Epsilon: {agent.epsilon:.3f}")
    
    return training_stats