"""
Deep Q-Network (DQN) Agent for RIR Estimation

This implements DQN with experience replay and target networks for
RIR estimation optimization.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):
    """Deep Q-Network architecture."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        """
        Initialize DQN network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation - raw Q-values)
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute Q-values."""
        return self.network(state)


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.
    
    This is a standard DQN implementation with:
    - Experience replay buffer for stable learning
    - Target network for stable Q-value targets
    - Epsilon-greedy exploration
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 10,
                 hidden_dims: List[int] = None):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension (number of discrete actions)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate per episode
            epsilon_min: Minimum epsilon value
            memory_size: Replay buffer size
            batch_size: Batch size for training
            target_update_freq: Episodes between target network updates
            hidden_dims: Hidden layer dimensions for networks
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Neural networks
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dims)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dims)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize target network
        self.update_target_network()
        
        # Statistics
        self.episodes_trained = 0
        self.total_updates = 0
        self.training_losses = []
        
        # Action codebook (maps discrete actions to continuous parameters)
        self.action_codebook = self._build_action_codebook()
        
    def _build_action_codebook(self) -> np.ndarray:
        """
        Build action codebook mapping discrete actions to continuous parameters.
        
        Returns:
            Array of shape (action_dim, 6) containing parameter vectors
        """
        # Define meaningful parameter combinations
        # Each action maps to [alpha, beta, reg, step_size, w1, w2]
        
        if self.action_dim == 8:
            codebook = np.array([
                # Conservative approaches
                [0.5, 0.05, 0.001, 0.001, 0.8, 0.2],   # Action 0
                [1.0, 0.02, 0.005, 0.005, 0.6, 0.4],   # Action 1
                
                # Aggressive approaches
                [2.0, 0.01, 0.01, 0.01, 0.5, 0.5],     # Action 2
                [3.0, 0.001, 0.05, 0.02, 0.3, 0.7],    # Action 3
                
                # Regularization focused
                [1.5, 0.03, 0.1, 0.001, 0.9, 0.1],     # Action 4
                [1.0, 0.05, 0.0001, 0.05, 0.1, 0.9],   # Action 5
                
                # Step size variations
                [1.2, 0.04, 0.01, 0.001, 0.7, 0.3],    # Action 6
                [1.8, 0.02, 0.01, 0.1, 0.4, 0.6],      # Action 7
            ], dtype=np.float32)
        else:
            # Generate random actions for different action_dim
            rng = np.random.RandomState(42)
            codebook = rng.uniform(
                low=[0.5, 0.001, 0.0001, 0.0001, 0.0, 0.0],
                high=[3.0, 0.1, 0.1, 0.1, 1.0, 1.0],
                size=(self.action_dim, 6)
            ).astype(np.float32)
        
        # Convert to [-1, 1] range expected by environment
        normalized = np.zeros_like(codebook)
        normalized[:, 0] = (codebook[:, 0] - 1.75) / 1.25  # alpha
        normalized[:, 1] = (codebook[:, 1] - 0.05) / 0.05  # beta
        normalized[:, 2] = (codebook[:, 2] - 0.05) / 0.05  # reg
        normalized[:, 3] = (codebook[:, 3] - 0.05) / 0.05  # step
        normalized[:, 4] = codebook[:, 4] * 2 - 1  # w1
        normalized[:, 5] = codebook[:, 5] * 2 - 1  # w2
        
        return np.clip(normalized, -1.0, 1.0)
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (use exploration)
            
        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return int(q_values.argmax().item())
    
    def get_action_vector(self, action: int) -> np.ndarray:
        """
        Get continuous parameter vector for discrete action.
        
        Args:
            action: Discrete action index
            
        Returns:
            Continuous parameter vector
        """
        return self.action_codebook[action % len(self.action_codebook)]
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self) -> Optional[float]:
        """
        Train the network on a batch of experiences.
        
        Returns:
            Training loss if update was performed, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        # Current Q-values for taken actions
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.total_updates += 1
        loss_value = float(loss.item())
        self.training_losses.append(loss_value)
        
        return loss_value
    
    def update_target_network(self):
        """Update target network with current network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.episodes_trained += 1
    
    def save(self, filepath: str):
        """Save agent state."""
        state = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episodes_trained': self.episodes_trained,
            'total_updates': self.total_updates,
            'action_codebook': self.action_codebook,
            'hyperparams': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'lr': self.lr,
                'gamma': self.gamma,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq
            }
        }
        torch.save(state, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        state = torch.load(filepath)
        self.q_network.load_state_dict(state['q_network'])
        self.target_network.load_state_dict(state['target_network'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.epsilon = state['epsilon']
        self.episodes_trained = state.get('episodes_trained', 0)
        self.total_updates = state.get('total_updates', 0)
        self.action_codebook = state.get('action_codebook', self._build_action_codebook())
    
    def get_stats(self) -> Dict:
        """Get agent statistics."""
        return {
            'episodes_trained': self.episodes_trained,
            'total_updates': self.total_updates,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'avg_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0.0,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }


def train_dqn_agent(env, agent: DQNAgent, episodes: int = 1000,
                    max_steps: int = 200, verbose: bool = True) -> List[Dict]:
    """
    Train DQN agent on RIR estimation environment.
    
    Args:
        env: RIR estimation environment
        agent: DQN agent to train
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
        episode_losses = []
        
        for step in range(max_steps):
            # Select and execute action
            action_idx = agent.act(state, training=True)
            action_vec = agent.get_action_vector(action_idx)
            
            next_state, reward, terminated, truncated, info = env.step(action_vec)
            done = terminated or truncated
            
            # Store experience
            agent.remember(state, action_idx, reward, next_state, done)
            
            # Train on batch
            loss = agent.replay()
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Decay epsilon after episode
        agent.decay_epsilon()
        
        # Update target network periodically
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()
        
        # Record statistics
        episode_stats = {
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'epsilon': agent.epsilon,
            'avg_loss': np.mean(episode_losses) if episode_losses else 0.0,
            'final_metrics': info.get('metrics', {}),
            'improvement': info.get('improvement', {}),
            'memory_size': len(agent.memory)
        }
        training_stats.append(episode_stats)
        
        if verbose and episode % 100 == 0:
            print(f"Episode {episode}/{episodes}, "
                  f"Reward: {total_reward:.2f}, "
                  f"Steps: {steps}, "
                  f"Epsilon: {agent.epsilon:.3f}, "
                  f"Avg Loss: {episode_stats['avg_loss']:.4f}")
    
    return training_stats
