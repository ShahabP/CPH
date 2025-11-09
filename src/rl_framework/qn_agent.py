"""
Q-Learning (Tabular) Agent for RIR Estimation

This implements classic Q-learning with discrete state-action spaces for
RIR estimation optimization.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from collections import defaultdict
import pickle


class QNAgent:
    """
    Tabular Q-Learning agent for RIR estimation.
    
    Uses discretized state space and discrete action space for classic
    reinforcement learning without neural networks.
    """
    
    def __init__(self, 
                 state_bins: Dict[str, int],
                 action_space_size: int = 8,
                 learning_rate: float = 0.1,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize Q-Learning agent.
        
        Args:
            state_bins: Dictionary mapping state features to number of bins
            action_space_size: Number of discrete actions
            learning_rate: Learning rate (alpha)
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate per episode
            epsilon_min: Minimum epsilon value
        """
        self.state_bins = state_bins
        self.action_space_size = action_space_size
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: defaultdict for efficient sparse storage
        # Keys are (discretized_state, action) tuples
        self.q_table = defaultdict(float)
        
        # Statistics
        self.episodes_trained = 0
        self.total_updates = 0
        
        # Action codebook (maps discrete actions to continuous parameter vectors)
        self.action_codebook = self._build_action_codebook()
        
    def _build_action_codebook(self) -> np.ndarray:
        """
        Build action codebook mapping discrete actions to continuous parameters.
        
        Returns:
            Array of shape (action_space_size, 6) containing parameter vectors
        """
        # Define meaningful parameter combinations
        # Each action maps to [alpha, beta, reg, step_size, w1, w2]
        
        codebook = np.array([
            # Conservative approaches
            [0.5, 0.05, 0.001, 0.001, 0.8, 0.2],   # Action 0: conservative spectral
            [1.0, 0.02, 0.005, 0.005, 0.6, 0.4],   # Action 1: balanced
            
            # Aggressive approaches  
            [2.0, 0.01, 0.01, 0.01, 0.5, 0.5],     # Action 2: aggressive spectral
            [3.0, 0.001, 0.05, 0.02, 0.3, 0.7],    # Action 3: very aggressive
            
            # Regularization focused
            [1.5, 0.03, 0.1, 0.001, 0.9, 0.1],     # Action 4: high regularization
            [1.0, 0.05, 0.0001, 0.05, 0.1, 0.9],   # Action 5: low regularization
            
            # Step size variations
            [1.2, 0.04, 0.01, 0.001, 0.7, 0.3],    # Action 6: small steps
            [1.8, 0.02, 0.01, 0.1, 0.4, 0.6],      # Action 7: large steps
        ], dtype=np.float32)
        
        # Ensure we have the right number of actions
        if len(codebook) < self.action_space_size:
            # Fill with random variations
            rng = np.random.RandomState(42)
            while len(codebook) < self.action_space_size:
                action = rng.uniform(
                    low=[0.5, 0.001, 0.0001, 0.0001, 0.0, 0.0],
                    high=[3.0, 0.1, 0.1, 0.1, 1.0, 1.0],
                    size=6
                ).astype(np.float32)
                codebook = np.vstack([codebook, action[np.newaxis, :]])
        
        # Convert to [-1, 1] range expected by environment
        normalized = np.zeros_like(codebook)
        normalized[:, 0] = (codebook[:, 0] - 1.75) / 1.25  # alpha: [0.5, 3.0] -> [-1, 1]
        normalized[:, 1] = (codebook[:, 1] - 0.05) / 0.05  # beta: [0, 0.1] -> [-1, 1]
        normalized[:, 2] = (codebook[:, 2] - 0.05) / 0.05  # reg: [0, 0.1] -> [-1, 1]
        normalized[:, 3] = (codebook[:, 3] - 0.05) / 0.05  # step: [0, 0.1] -> [-1, 1]
        normalized[:, 4] = codebook[:, 4] * 2 - 1  # w1: [0, 1] -> [-1, 1]
        normalized[:, 5] = codebook[:, 5] * 2 - 1  # w2: [0, 1] -> [-1, 1]
        
        return np.clip(normalized, -1.0, 1.0)
    
    def discretize_state(self, state: np.ndarray) -> Tuple[int, ...]:
        """
        Discretize continuous state into bins.
        
        Args:
            state: Continuous state vector
            
        Returns:
            Tuple of bin indices representing discretized state
        """
        # Extract key features from state
        # State structure: [reverb_features, dereverb_features, rir, metrics]
        
        # Simplified: use aggregated statistics
        state_len = len(state)
        
        # Extract approximate regions (adjust based on actual state structure)
        # These indices are approximate - adjust if needed
        feature_dim = 513  # from environment
        rir_length = 1024
        
        if state_len >= 2 * feature_dim + rir_length + 7:
            reverb_feat = state[:feature_dim]
            dereverb_feat = state[feature_dim:2*feature_dim]
            rir_est = state[2*feature_dim:2*feature_dim+rir_length]
            metrics = state[2*feature_dim+rir_length:]
            
            # Compute aggregated features
            reverb_energy = float(np.mean(reverb_feat ** 2))
            dereverb_energy = float(np.mean(dereverb_feat ** 2))
            rir_peak = float(np.max(np.abs(rir_est)))
            rir_energy = float(np.sum(rir_est ** 2))
            rir_sparsity = float(np.sum(np.abs(rir_est) > 0.01) / len(rir_est))
            
            # Get metrics if available
            correlation = float(metrics[4]) if len(metrics) > 4 else 0.0
            iteration_progress = float(metrics[0]) if len(metrics) > 0 else 0.0
        else:
            # Fallback: use simple statistics
            reverb_energy = float(np.mean(state ** 2))
            dereverb_energy = reverb_energy * 0.8
            rir_peak = 0.5
            rir_energy = 1.0
            rir_sparsity = 0.1
            correlation = 0.0
            iteration_progress = 0.0
        
        # Discretize each feature into bins
        discretized = []
        
        # Reverb energy: [0, 10] -> bins
        bins_reverb = self.state_bins.get('reverb_energy', 5)
        discretized.append(min(bins_reverb - 1, int(np.clip(reverb_energy * bins_reverb / 10, 0, bins_reverb - 1))))
        
        # Dereverb energy: [0, 10] -> bins
        bins_dereverb = self.state_bins.get('dereverb_energy', 5)
        discretized.append(min(bins_dereverb - 1, int(np.clip(dereverb_energy * bins_dereverb / 10, 0, bins_dereverb - 1))))
        
        # RIR peak: [0, 1] -> bins
        bins_peak = self.state_bins.get('rir_peak', 5)
        discretized.append(min(bins_peak - 1, int(np.clip(rir_peak * bins_peak, 0, bins_peak - 1))))
        
        # RIR energy: [0, 100] -> bins
        bins_energy = self.state_bins.get('rir_energy', 5)
        discretized.append(min(bins_energy - 1, int(np.clip(rir_energy * bins_energy / 100, 0, bins_energy - 1))))
        
        # RIR sparsity: [0, 1] -> bins
        bins_sparsity = self.state_bins.get('rir_sparsity', 5)
        discretized.append(min(bins_sparsity - 1, int(np.clip(rir_sparsity * bins_sparsity, 0, bins_sparsity - 1))))
        
        # Correlation: [0, 1] -> bins
        bins_corr = self.state_bins.get('correlation', 5)
        discretized.append(min(bins_corr - 1, int(np.clip(correlation * bins_corr, 0, bins_corr - 1))))
        
        # Iteration progress: [0, 1] -> bins
        bins_iter = self.state_bins.get('iteration', 3)
        discretized.append(min(bins_iter - 1, int(np.clip(iteration_progress * bins_iter, 0, bins_iter - 1))))
        
        return tuple(discretized)
    
    def get_q_value(self, state: Tuple[int, ...], action: int) -> float:
        """Get Q-value for state-action pair."""
        return self.q_table[(state, action)]
    
    def set_q_value(self, state: Tuple[int, ...], action: int, value: float):
        """Set Q-value for state-action pair."""
        self.q_table[(state, action)] = value
        self.total_updates += 1
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Continuous state vector
            training: Whether in training mode (use exploration)
            
        Returns:
            Selected action index
        """
        discrete_state = self.discretize_state(state)
        
        # Epsilon-greedy
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_space_size)
        
        # Greedy: select action with highest Q-value
        q_values = [self.get_q_value(discrete_state, a) for a in range(self.action_space_size)]
        return int(np.argmax(q_values))
    
    def get_action_vector(self, action: int) -> np.ndarray:
        """
        Get continuous parameter vector for discrete action.
        
        Args:
            action: Discrete action index
            
        Returns:
            Continuous parameter vector
        """
        return self.action_codebook[action % len(self.action_codebook)]
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Current Q-value
        current_q = self.get_q_value(discrete_state, action)
        
        # Max Q-value for next state
        if done:
            max_next_q = 0.0
        else:
            next_q_values = [self.get_q_value(discrete_next_state, a) 
                           for a in range(self.action_space_size)]
            max_next_q = max(next_q_values)
        
        # Q-learning update
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.set_q_value(discrete_state, action, new_q)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.episodes_trained += 1
    
    def save(self, filepath: str):
        """Save Q-table and agent state."""
        state = {
            'q_table': dict(self.q_table),
            'state_bins': self.state_bins,
            'action_space_size': self.action_space_size,
            'lr': self.lr,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'episodes_trained': self.episodes_trained,
            'total_updates': self.total_updates,
            'action_codebook': self.action_codebook
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, filepath: str):
        """Load Q-table and agent state."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.q_table = defaultdict(float, state['q_table'])
        self.state_bins = state['state_bins']
        self.action_space_size = state['action_space_size']
        self.lr = state['lr']
        self.gamma = state['gamma']
        self.epsilon = state['epsilon']
        self.epsilon_decay = state['epsilon_decay']
        self.epsilon_min = state['epsilon_min']
        self.episodes_trained = state.get('episodes_trained', 0)
        self.total_updates = state.get('total_updates', 0)
        self.action_codebook = state.get('action_codebook', self._build_action_codebook())
    
    def get_stats(self) -> Dict:
        """Get agent statistics."""
        return {
            'q_table_size': len(self.q_table),
            'episodes_trained': self.episodes_trained,
            'total_updates': self.total_updates,
            'epsilon': self.epsilon,
            'state_space_size': np.prod(list(self.state_bins.values())),
            'action_space_size': self.action_space_size
        }


def train_qn_agent(env, agent: QNAgent, episodes: int = 1000, 
                   max_steps: int = 200, verbose: bool = True) -> List[Dict]:
    """
    Train Q-Learning agent on RIR estimation environment.
    
    Args:
        env: RIR estimation environment
        agent: QN agent to train
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        verbose: Whether to print progress
        
    Returns:
        List of training statistics per episode
    """
    training_stats = []
    
    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0.0
        steps = 0
        
        for step in range(max_steps):
            # Select and execute action
            action_idx = agent.act(state, training=True)
            action_vec = agent.get_action_vector(action_idx)
            
            next_state, reward, terminated, truncated, info = env.step(action_vec)
            done = terminated or truncated
            
            # Update Q-table
            agent.update(state, action_idx, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Decay epsilon after episode
        agent.decay_epsilon()
        
        # Record statistics
        episode_stats = {
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'epsilon': agent.epsilon,
            'final_metrics': info.get('metrics', {}),
            'improvement': info.get('improvement', {}),
            'q_table_size': len(agent.q_table)
        }
        training_stats.append(episode_stats)
        
        if verbose and episode % 100 == 0:
            print(f"Episode {episode}/{episodes}, "
                  f"Reward: {total_reward:.2f}, "
                  f"Steps: {steps}, "
                  f"Epsilon: {agent.epsilon:.3f}, "
                  f"Q-table size: {len(agent.q_table)}")
    
    return training_stats
