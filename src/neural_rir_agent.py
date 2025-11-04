"""
Neural RIR Policy Agent - Direct RIR estimation using deep networks.

This agent takes the current RIR estimate as state and outputs RIR updates
directly, optimized via DRR (Direct-to-Reverberant Ratio) based rewards.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy import signal
from collections import deque
import random


def compute_drr(rir: np.ndarray, sample_rate: int = 16000, 
                direct_window_ms: float = 2.5) -> float:
    """
    Compute Direct-to-Reverberant Ratio (DRR) from RIR.
    
    Args:
        rir: Room impulse response
        sample_rate: Sampling rate
        direct_window_ms: Time window for direct sound in milliseconds
        
    Returns:
        DRR in dB
    """
    if len(rir) == 0:
        return -np.inf
    
    # Find direct sound peak
    peak_idx = np.argmax(np.abs(rir))
    
    # Define direct sound window
    direct_samples = int(direct_window_ms * sample_rate / 1000)
    direct_start = max(0, peak_idx - direct_samples // 4)
    direct_end = min(len(rir), peak_idx + direct_samples)
    
    # Direct energy
    direct_energy = np.sum(rir[direct_start:direct_end] ** 2)
    
    # Reverberant energy (everything else)
    reverb_mask = np.ones(len(rir), dtype=bool)
    reverb_mask[direct_start:direct_end] = False
    reverb_energy = np.sum(rir[reverb_mask] ** 2)
    
    if reverb_energy <= 0:
        return np.inf
    
    drr_db = 10 * np.log10(direct_energy / reverb_energy + 1e-12)
    return float(drr_db)


class RIRPolicyNetwork(nn.Module):
    """Neural network that takes RIR as input and outputs RIR updates."""
    
    def __init__(self, rir_length: int = 1024, hidden_dim: int = 512):
        super().__init__()
        self.rir_length = rir_length
        
        # RIR encoder - compress RIR to latent representation
        self.encoder = nn.Sequential(
            nn.Linear(rir_length, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Policy head - outputs RIR update
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rir_length),
            nn.Tanh()  # Bounded updates
        )
        
        # Value head - estimates state value
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, rir_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            rir_state: Current RIR estimate [batch, rir_length]
            
        Returns:
            Tuple of (rir_update, state_value)
        """
        # Encode RIR
        features = self.encoder(rir_state)
        
        # Get policy (RIR update) and value
        rir_update = self.policy_head(features)
        state_value = self.value_head(features)
        
        return rir_update, state_value


class NeuralRIRAgent:
    """Agent that directly updates RIR using neural policy with DRR rewards."""
    
    def __init__(self, rir_length: int = 1024, learning_rate: float = 1e-4,
                 gamma: float = 0.99, update_scale: float = 0.1):
        self.rir_length = rir_length
        self.gamma = gamma
        self.update_scale = update_scale
        
        # Neural network
        self.policy_net = RIRPolicyNetwork(rir_length)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Current episode data
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_values = []
        
    def act(self, rir_state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action (RIR update) given current RIR state.
        
        Args:
            rir_state: Current RIR estimate
            training: Whether in training mode
            
        Returns:
            RIR update vector
        """
        with torch.no_grad():
            rir_tensor = torch.FloatTensor(rir_state).unsqueeze(0)
            rir_update, state_value = self.policy_net(rir_tensor)
            
            if training:
                # Add exploration noise during training
                noise = torch.randn_like(rir_update) * 0.01
                rir_update = rir_update + noise
            
            # Store for episode
            if training:
                self.episode_states.append(rir_state.copy())
                self.episode_actions.append(rir_update.squeeze(0).numpy())
                self.episode_values.append(state_value.item())
            
            return self.update_scale * rir_update.squeeze(0).numpy()
    
    def update_rir(self, current_rir: np.ndarray, rir_update: np.ndarray) -> np.ndarray:
        """
        Apply RIR update to current estimate.
        
        Args:
            current_rir: Current RIR estimate
            rir_update: Update vector from policy
            
        Returns:
            Updated RIR
        """
        # Direct additive update with normalization
        new_rir = current_rir + rir_update
        
        # Normalize to prevent explosion
        max_amp = np.max(np.abs(new_rir))
        if max_amp > 1.0:
            new_rir = new_rir / max_amp
            
        return new_rir
    
    def compute_reward(self, rir: np.ndarray, reverb_audio: np.ndarray, 
                      clean_audio: Optional[np.ndarray] = None,
                      sample_rate: int = 16000) -> float:
        """
        Compute DRR-based reward for RIR estimate.
        
        Args:
            rir: Current RIR estimate
            reverb_audio: Reverberant audio
            clean_audio: Clean audio (optional, for oracle metrics)
            sample_rate: Sampling rate
            
        Returns:
            Reward value
        """
        # Primary reward: DRR (higher is better for most rooms)
        drr = compute_drr(rir, sample_rate)
        
        # Normalize DRR to reasonable range (typical DRR: -10 to +10 dB)
        drr_reward = np.tanh(drr / 10.0)  # Maps [-inf, +inf] to [-1, +1]
        
        # Secondary rewards
        rewards = {'drr': drr_reward}
        
        # Sparsity reward (encourage sparse RIR)
        sparsity = 1.0 - (np.count_nonzero(np.abs(rir) > 0.01) / len(rir))
        rewards['sparsity'] = 0.2 * sparsity
        
        # Stability reward (penalize extreme values)
        max_amp = np.max(np.abs(rir))
        stability = 1.0 - min(max_amp, 2.0) / 2.0
        rewards['stability'] = 0.1 * stability
        
        # Energy decay reward (encourage exponential decay shape)
        try:
            peak_idx = np.argmax(np.abs(rir))
            tail = rir[peak_idx + 50:]  # Skip initial reflections
            if len(tail) > 100:
                # Fit exponential decay
                t = np.arange(len(tail))
                log_env = np.log(np.abs(tail) + 1e-12)
                # Simple linear fit to log envelope
                decay_slope = np.polyfit(t, log_env, 1)[0]
                decay_reward = max(0, -decay_slope)  # Reward negative slopes
                rewards['decay'] = 0.1 * min(decay_reward, 1.0)
            else:
                rewards['decay'] = 0.0
        except Exception:
            rewards['decay'] = 0.0
        
        # Optional: Oracle reward if clean audio available
        if clean_audio is not None:
            try:
                # Deconvolve with estimated RIR
                estimated_clean = signal.wiener(reverb_audio, noise=0.1)
                # Simple correlation with true clean
                if len(estimated_clean) > 0 and len(clean_audio) > 0:
                    min_len = min(len(estimated_clean), len(clean_audio))
                    corr = np.corrcoef(estimated_clean[:min_len], clean_audio[:min_len])[0, 1]
                    if not np.isnan(corr):
                        rewards['oracle'] = 0.2 * corr
                    else:
                        rewards['oracle'] = 0.0
                else:
                    rewards['oracle'] = 0.0
            except Exception:
                rewards['oracle'] = 0.0
        
        total_reward = sum(rewards.values())
        return float(total_reward)
    
    def store_reward(self, reward: float):
        """Store reward for current step."""
        self.episode_rewards.append(reward)
    
    def end_episode(self):
        """End episode and perform learning update."""
        if len(self.episode_rewards) == 0:
            return
        
        # Compute discounted returns
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Convert episode data to tensors
        states = torch.FloatTensor(self.episode_states)
        actions = torch.FloatTensor(self.episode_actions)
        values = torch.FloatTensor(self.episode_values)
        
        # Compute advantages
        advantages = returns - values
        
        # Policy gradient loss
        policy_updates, policy_values = self.policy_net(states)
        
        # Actor loss (policy gradient)
        action_log_probs = -0.5 * torch.sum((policy_updates - actions) ** 2, dim=1)
        actor_loss = -(action_log_probs * advantages.detach()).mean()
        
        # Critic loss (value function)
        critic_loss = F.mse_loss(policy_values.squeeze(), returns)
        
        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()
        
        # Clear episode data
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_values.clear()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'total_loss': total_loss.item(),
            'avg_return': returns.mean().item()
        }


class NeuralRIREnvironment:
    """Environment for neural RIR policy training."""
    
    def __init__(self, max_iterations: int = 20, rir_length: int = 1024,
                 sample_rate: int = 16000):
        self.max_iterations = max_iterations
        self.rir_length = rir_length
        self.sample_rate = sample_rate
        
        self.current_rir = None
        self.reverb_audio = None
        self.clean_audio = None
        self.step_count = 0
        
    def reset(self, reverb_audio: np.ndarray, clean_audio: Optional[np.ndarray] = None,
              initial_rir: Optional[np.ndarray] = None):
        """Reset environment with new audio."""
        self.reverb_audio = reverb_audio
        self.clean_audio = clean_audio
        self.step_count = 0
        
        # Initialize RIR estimate
        if initial_rir is not None:
            self.current_rir = initial_rir.copy()
        else:
            # Simple initialization: impulse + small random tail
            self.current_rir = np.zeros(self.rir_length)
            self.current_rir[0] = 1.0
            self.current_rir[1:100] = np.random.randn(99) * 0.01
        
        return self.current_rir.copy()
    
    def step(self, agent: NeuralRIRAgent) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step with the neural agent."""
        # Agent selects RIR update
        rir_update = agent.act(self.current_rir, training=True)
        
        # Apply update to RIR
        self.current_rir = agent.update_rir(self.current_rir, rir_update)
        
        # Compute reward
        reward = agent.compute_reward(
            self.current_rir, self.reverb_audio, 
            self.clean_audio, self.sample_rate
        )
        
        # Store reward in agent
        agent.store_reward(reward)
        
        # Check termination
        self.step_count += 1
        terminated = self.step_count >= self.max_iterations
        
        info = {
            'step': self.step_count,
            'drr': compute_drr(self.current_rir, self.sample_rate),
            'rir_energy': np.sum(self.current_rir ** 2),
            'rir_peak': np.max(np.abs(self.current_rir))
        }
        
        return self.current_rir.copy(), reward, terminated, info


def compare_agents_demo():
    """Demo function to compare DQN vs Neural RIR agents."""
    print("=== Neural RIR Agent Demo ===")
    
    # Create synthetic data
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(duration * sample_rate))
    clean = 0.2 * np.sin(2 * np.pi * 440 * t) * np.exp(-t)  # Decaying sine
    
    # True RIR (exponential decay)
    true_rir = np.zeros(1024)
    true_rir[0] = 1.0
    for i in range(1, 200):
        true_rir[i] = 0.3 * np.exp(-i / 50) * (1 + 0.2 * np.random.randn())
    
    # Create reverberant audio
    reverb = np.convolve(clean, true_rir, mode='same')
    
    # Initialize environment and agent
    env = NeuralRIREnvironment(max_iterations=20)
    agent = NeuralRIRAgent(rir_length=1024, learning_rate=1e-3)
    
    print(f"True RIR DRR: {compute_drr(true_rir, sample_rate):.2f} dB")
    
    # Run training episode
    rir_estimates = []
    rewards = []
    
    state = env.reset(reverb, clean)
    rir_estimates.append(state.copy())
    
    for step in range(env.max_iterations):
        next_state, reward, terminated, info = env.step(agent)
        rir_estimates.append(next_state.copy())
        rewards.append(reward)
        
        print(f"Step {step}: Reward={reward:.3f}, DRR={info['drr']:.2f} dB, "
              f"Peak={info['rir_peak']:.3f}")
        
        if terminated:
            break
    
    # End episode and learn
    train_info = agent.end_episode()
    if train_info:
        print(f"Training: Actor Loss={train_info['actor_loss']:.4f}, "
              f"Critic Loss={train_info['critic_loss']:.4f}")
    
    # Final comparison
    final_rir = rir_estimates[-1]
    final_drr = compute_drr(final_rir, sample_rate)
    
    # Simple correlation with true RIR
    min_len = min(len(final_rir), len(true_rir))
    correlation = np.corrcoef(final_rir[:min_len], true_rir[:min_len])[0, 1]
    
    print(f"\n=== Results ===")
    print(f"Final DRR: {final_drr:.2f} dB (true: {compute_drr(true_rir, sample_rate):.2f} dB)")
    print(f"RIR Correlation: {correlation:.3f}")
    print(f"Total Reward: {sum(rewards):.3f}")
    
    return {
        'rir_estimates': rir_estimates,
        'rewards': rewards,
        'final_drr': final_drr,
        'correlation': correlation,
        'true_rir': true_rir
    }


if __name__ == "__main__":
    results = compare_agents_demo()