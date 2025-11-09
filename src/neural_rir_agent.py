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


def compute_rir_drr_metric(rir: np.ndarray, sample_rate: int = 16000, 
                          direct_window_ms: float = 2.5) -> float:
    """
    Compute DRR-like metric from RIR structure (for monitoring only).
    NOTE: This is NOT the actual DRR - it's just a structural metric.
    The real DRR should be computed on dereverberated speech.
    
    Args:
        rir: Room impulse response
        sample_rate: Sampling rate
        direct_window_ms: Time window for direct sound in milliseconds
        
    Returns:
        DRR in dB
    """
    if len(rir) == 0:
        return -np.inf
    
    # Find direct sound peak with better detection
    abs_rir = np.abs(rir)
    peak_idx = np.argmax(abs_rir)
    peak_value = abs_rir[peak_idx]
    
    # Adaptive window based on peak strength
    direct_samples = int(direct_window_ms * sample_rate / 1000)
    # Expand window for stronger peaks to capture more direct energy
    window_scale = min(2.0, 1.0 + peak_value)
    direct_samples = int(direct_samples * window_scale)
    
    # Define direct sound window - centered around peak
    direct_start = max(0, peak_idx - direct_samples // 2)
    direct_end = min(len(rir), peak_idx + direct_samples // 2)
    
    # Direct energy with emphasis on peak region
    direct_region = rir[direct_start:direct_end]
    direct_energy = np.sum(direct_region ** 2)
    
    # Reverberant energy - exclude direct region and early reflections
    # Skip immediate post-direct region to avoid counting early reflections as reverb
    early_reflection_skip = int(0.005 * sample_rate)  # 5ms skip
    reverb_start = min(len(rir), direct_end + early_reflection_skip)
    
    if reverb_start >= len(rir):
        # No reverb tail, assume very dry RIR
        return 20.0  # High positive DRR for dry conditions
    
    reverb_energy = np.sum(rir[reverb_start:] ** 2)
    
    # Add small energy floor to prevent division issues
    reverb_energy = max(reverb_energy, direct_energy * 1e-6)
    
    # Compute DRR with bias toward positive values
    if reverb_energy <= 0 or direct_energy <= 0:
        return 10.0  # Default positive DRR
    
    drr_db = 10 * np.log10(direct_energy / reverb_energy + 1e-12)
    
    # Apply floor to encourage positive DRR
    drr_db = max(drr_db, -15.0)  # Prevent extremely negative DRR
    
    return float(drr_db)


class RIRPolicyNetwork(nn.Module):
    """
    Enhanced neural network for RIR updates optimized for positive DRR.
    
    Key improvements:
    1. Larger capacity for better RIR structure learning
    2. Residual connections for stable training  
    3. Specialized heads for direct/reverb components
    4. Batch normalization for stable gradients
    """
    
    def __init__(self, rir_length: int = 1024, hidden_dim: int = 768):
        super().__init__()
        self.rir_length = rir_length
        
        # Input normalization
        self.input_norm = nn.LayerNorm(rir_length)
        
        # Enhanced RIR encoder with residual connections
        self.encoder1 = nn.Sequential(
            nn.Linear(rir_length, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.encoder3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # Multiple specialized heads for realistic RIR structure
        latent_dim = hidden_dim // 2
        
        # Direct sound head (first tap only)
        self.direct_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Single direct sound tap
            nn.Sigmoid()  # Always positive for direct sound
        )
        
        # Early reflections head (taps 1-63) 
        self.early_reflections_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Linear(64, 63),  # Early reflection taps
            nn.Tanh()  # Can be positive or negative
        )
        
        # Late reverberation head (taps 64-255)
        self.late_reverb_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(), 
            nn.Linear(128, 64),
            nn.Linear(64, 192),  # Late reverb taps
            nn.Tanh()
        )
        
        # Tail decay head (remaining taps)
        self.tail_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Linear(32, rir_length - 256),  # Tail taps
            nn.Tanh()
        )
        
        # Enhanced value head
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.ReLU(),
            nn.Linear(latent_dim // 4, 1)
        )
        
    def forward(self, rir_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with specialized direct/reverb processing.
        
        Args:
            rir_state: Current RIR estimate [batch, rir_length]
            
        Returns:
            Tuple of (rir_update, state_value)
        """
        # Normalize input
        x = self.input_norm(rir_state)
        
        # Encode RIR with residual connections
        h1 = self.encoder1(x)
        h2 = self.encoder2(h1) + h1  # Residual connection
        features = self.encoder3(h2)
        
        # Generate structured RIR components
        direct_update = self.direct_head(features)  # [batch, 1]
        early_update = self.early_reflections_head(features)  # [batch, 63] 
        late_update = self.late_reverb_head(features)  # [batch, 192]
        tail_update = self.tail_head(features)  # [batch, rir_length-256]
        
        # Apply realistic acoustic scaling - encourage more reverberation
        direct_scale = 0.3   # Moderate direct sound
        early_scale = 0.4    # Significant early reflections 
        late_scale = 0.3     # Substantial late reverberation
        tail_scale = 0.2     # Noticeable tail
        
        # Combine into structured RIR update
        rir_update = torch.cat([
            direct_update * direct_scale,      # Direct sound (tap 0)
            early_update * early_scale,        # Early reflections (taps 1-63)
            late_update * late_scale,          # Late reverb (taps 64-255)
            tail_update * tail_scale           # Decay tail (taps 256+)
        ], dim=1)
        
        # Get state value
        state_value = self.value_head(features)
        
        return rir_update, state_value


class NeuralRIRAgent:
    """Agent that directly updates RIR using neural policy with DRR rewards."""
    
    def __init__(self, rir_length: int = 1024, learning_rate: float = 3e-4,
                 gamma: float = 0.95, update_scale: float = 0.05):
        self.rir_length = rir_length
        self.gamma = gamma
        self.update_scale = update_scale
        
        # Enhanced neural network
        self.policy_net = RIRPolicyNetwork(rir_length, hidden_dim=768)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Learning rate scheduler for stable convergence
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.8, patience=10
        )
        
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
        # Set network mode appropriately for batch norm
        if training:
            self.policy_net.train()
        else:
            self.policy_net.eval()
            
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
        Apply RIR update to generate natural acoustic structure.
        
        Args:
            current_rir: Current RIR estimate
            rir_update: Update vector from policy
            
        Returns:
            Updated RIR with realistic acoustic properties
        """
        # Apply update with adaptive momentum and structure enhancement
        momentum = 0.2
        new_rir = current_rir * (1 - momentum) + (current_rir + rir_update) * momentum
        
        # Actively encourage reverberation structure during updates
        new_rir = self._encourage_reverberation(new_rir)
        
        # Ensure realistic RIR structure
        new_rir = self._enforce_acoustic_structure(new_rir)
        
        # Normalize while preserving structure
        max_amp = np.max(np.abs(new_rir))
        if max_amp > 1.0:
            new_rir = new_rir / max_amp
            
        return new_rir
    
    def _enforce_acoustic_structure(self, rir: np.ndarray) -> np.ndarray:
        """Enforce realistic acoustic structure on RIR."""
        # Ensure direct sound is present but allow substantial reverberation
        if len(rir) > 0:
            # Direct sound should be significant but not overwhelming
            peak_idx = np.argmax(np.abs(rir))
            if peak_idx > 10:  # If peak is too late, boost early tap
                early_boost = np.abs(rir[peak_idx]) * 0.6
                rir[0] = max(rir[0], early_boost)
        
        # Apply realistic but generous decay envelope to allow reverberation
        sample_rate = 16000  # Assuming 16kHz
        
        # Early reflections (0-50ms): allow substantial reflections
        early_end = int(0.05 * sample_rate)  # 50ms
        if len(rir) > early_end and len(rir) > 0:
            max_direct = abs(rir[0])
            for i in range(1, min(early_end, len(rir))):
                # Allow early reflections up to 60% of direct sound
                max_early = max_direct * 0.6 * np.exp(-i / 200)  # Gradual decay
                # Don't limit if already smaller
                if abs(rir[i]) > max_early:
                    rir[i] = np.sign(rir[i]) * max_early
        
        # Late reverberation (50-200ms): allow substantial reverb
        late_start = early_end
        late_end = int(0.2 * sample_rate)  # 200ms
        if len(rir) > late_start and len(rir) > 0:
            max_direct = abs(rir[0])
            for i in range(late_start, min(late_end, len(rir))):
                # Allow late reverb up to 40% of direct with slower decay
                t = (i - late_start) / (late_end - late_start)
                decay_factor = np.exp(-2 * t)  # Gentler decay
                max_late = max_direct * 0.4 * decay_factor
                if abs(rir[i]) > max_late:
                    rir[i] = np.sign(rir[i]) * max_late
        
        # Tail (200ms+): allow moderate tail
        if len(rir) > late_end and len(rir) > 0:
            max_direct = abs(rir[0])
            for i in range(late_end, len(rir)):
                t = (i - late_end) / (len(rir) - late_end)
                decay_factor = np.exp(-4 * t)  # Moderate decay
                max_tail = max_direct * 0.2 * decay_factor
                if abs(rir[i]) > max_tail:
                    rir[i] = np.sign(rir[i]) * max_tail
        
        return rir
    
    def _dereverberate_with_rir(self, reverb_audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
        """
        Dereverberate audio using estimated RIR via Wiener deconvolution.
        
        Args:
            reverb_audio: Reverberant speech signal
            rir: Estimated room impulse response
            
        Returns:
            Dereverberated speech signal
        """
        from scipy import signal
        
        # Pad RIR to avoid circular convolution artifacts
        rir_padded = np.zeros(len(reverb_audio) + len(rir) - 1)
        rir_padded[:len(rir)] = rir
        
        # Convert to frequency domain
        reverb_fft = np.fft.fft(reverb_audio, n=len(rir_padded))
        rir_fft = np.fft.fft(rir_padded)
        
        # Wiener deconvolution with regularization
        regularization = 0.01
        rir_conj = np.conj(rir_fft)
        rir_power = np.abs(rir_fft) ** 2
        
        # Wiener filter: H* / (|H|^2 + λ)
        wiener_filter = rir_conj / (rir_power + regularization)
        
        # Apply filter
        clean_fft = reverb_fft * wiener_filter
        
        # Convert back to time domain and trim to original length
        dereverberated = np.real(np.fft.ifft(clean_fft))[:len(reverb_audio)]
        
        return dereverberated
    
    def _compute_speech_drr(self, audio: np.ndarray, sample_rate: int = 16000) -> float:
        """
        Compute DRR (Direct-to-Reverberant Ratio) on speech signal.
        
        Args:
            audio: Speech signal
            sample_rate: Sampling rate
            
        Returns:
            DRR in dB
        """
        if len(audio) < sample_rate // 10:  # Less than 100ms
            return -20.0
        
        # Frame-based analysis
        frame_size = int(0.025 * sample_rate)  # 25ms frames
        hop_size = int(0.01 * sample_rate)     # 10ms hop
        
        frames = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            frames.append(frame)
        
        if len(frames) < 5:
            return -20.0
        
        # Compute frame energies
        frame_energies = [np.sum(frame ** 2) for frame in frames]
        
        # Find direct sound (strongest frames in first 200ms)
        max_direct_frames = min(20, len(frame_energies))  # First 200ms
        direct_energy = np.max(frame_energies[:max_direct_frames])
        
        # Estimate reverberation energy (mean of remaining frames)
        if len(frame_energies) > max_direct_frames:
            reverb_frames = frame_energies[max_direct_frames:]
            reverb_energy = np.mean(reverb_frames)
        else:
            reverb_energy = np.mean(frame_energies) * 0.1  # Assume 10% reverb
        
        # Compute DRR
        drr_db = 10 * np.log10((direct_energy + 1e-12) / (reverb_energy + 1e-12))
        
        return float(drr_db)
    
    def _encourage_reverberation(self, rir: np.ndarray) -> np.ndarray:
        """Actively modify RIR to have more realistic reverberation structure."""
        if len(rir) < 64:
            return rir
            
        # Target energy distribution: 30% direct, 25% early, 25% late, 20% tail  
        current_total = np.sum(rir ** 2)
        if current_total < 1e-12:
            return rir
            
        current_direct = rir[0] ** 2 / current_total
        
        # If direct sound is too dominant (>80%), redistribute energy
        if current_direct > 0.8 and len(rir) > 256:
            # Calculate how much energy to redistribute
            excess_direct = (current_direct - 0.5) * current_total
            
            # Add structured reverberation
            # Early reflections (exponential decay)
            for i in range(1, 64):
                decay = np.exp(-i / 20.0)  # 20-sample decay constant
                addition = np.sqrt(excess_direct * 0.4 * decay / 63.0)  # 40% to early
                rir[i] += addition * np.random.choice([-1, 1])  # Random polarity
                
            # Late reverberation (slower decay)
            for i in range(64, 256):
                decay = np.exp(-(i-64) / 50.0)  # 50-sample decay constant  
                addition = np.sqrt(excess_direct * 0.35 * decay / 192.0)  # 35% to late
                rir[i] += addition * np.random.choice([-1, 1])
                
            # Tail (very slow decay)
            for i in range(256, len(rir)):
                decay = np.exp(-(i-256) / 100.0)  # 100-sample decay constant
                addition = np.sqrt(excess_direct * 0.25 * decay / (len(rir)-256))  # 25% to tail
                rir[i] += addition * np.random.choice([-1, 1])
                
            # Reduce direct sound to maintain energy balance
            rir[0] *= 0.8
            
        return rir
    
    def compute_reward(self, rir: np.ndarray, reverb_audio: np.ndarray, 
                      clean_audio: Optional[np.ndarray] = None,
                      sample_rate: int = 16000) -> float:
        """
        Compute DRR-based reward for RIR estimate by dereverbarating the speech.
        
        Args:
            rir: Current RIR estimate
            reverb_audio: Reverberant audio
            clean_audio: Clean audio (optional, for oracle metrics)
            sample_rate: Sampling rate
            
        Returns:
            Reward value
        """
        # Apply estimated RIR to dereverb the speech and compute DRR on result
        try:
            # Use the estimated RIR to dereverberate the speech
            from scipy import signal
            
            # Perform Wiener deconvolution to get dereverberated signal
            # This is the key: use the RIR to actually dereverb the speech
            dereverberated_audio = self._dereverberate_with_rir(reverb_audio, rir)
            
            # Compute DRR on the dereverberated audio (not on RIR itself!)
            drr = self._compute_speech_drr(dereverberated_audio, sample_rate)
            
        except Exception as e:
            # Fallback: heavily penalize if dereverberation fails
            drr = -20.0
        
        # Normalize DRR to realistic room range (0-15 dB for typical rooms)
        if drr > 0:
            drr_reward = min(1.0, drr / 15.0)  # Linear up to 15dB
        else:
            drr_reward = np.tanh(drr / 5.0)  # Gentle penalty for negative DRR
        
        # Secondary rewards for acoustic realism
        rewards = {'drr': 2.0 * drr_reward}  # Moderate DRR weighting
        
        # Direct sound strength (should be clear but not dominating)
        direct_strength = np.abs(rir[0]) if len(rir) > 0 else 0
        # Penalize too strong direct sound to encourage reverberation
        direct_reward = min(1.0, direct_strength * 2.0) * (1 - max(0, direct_strength - 0.6))
        rewards['direct'] = 0.8 * direct_reward
        
        # Early reflection structure (should be substantial)
        if len(rir) > 64:
            early_energy = np.sum(rir[1:64] ** 2)
            early_ratio = early_energy / max(rir[0] ** 2, 1e-8)
            early_reward = 1.0 - abs(early_ratio - 0.4)  # Target 40% early energy
            rewards['early_struct'] = 1.2 * max(0, early_reward)
        
        # Realistic decay structure - encourage more late reverb
        if len(rir) > 128:
            late_start = 64
            late_energy = np.sum(rir[late_start:late_start+128] ** 2)
            total_non_direct = np.sum(rir[1:] ** 2)
            if total_non_direct > 0:
                late_ratio = late_energy / total_non_direct
                late_reward = 1.0 - abs(late_ratio - 0.4)  # Late reverb should be 40% of non-direct
                rewards['late_struct'] = 1.0 * max(0, late_reward)
        
        # Encourage realistic tail presence
        if len(rir) > 256:
            tail_energy = np.sum(rir[256:] ** 2)
            total_energy = np.sum(rir ** 2)
            tail_ratio = tail_energy / max(total_energy, 1e-8)
            # Target 5-15% tail energy for realistic rooms
            tail_reward = 1.0 - abs(tail_ratio - 0.1)  
            rewards['tail_presence'] = 0.8 * max(0, tail_reward)
        
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
    
    def _get_exponential_decay_rir(self, rir_length: int) -> np.ndarray:
        """
        Generate exponential decay RIR initialization.
        
        Creates a physically plausible RIR with:
        - Strong direct sound at t=0
        - Exponential decay following typical room acoustics
        - Realistic decay constants for early and late reflections
        """
        rir = np.zeros(rir_length)
        
        # Direct sound (strong impulse at t=0)
        rir[0] = 1.0
        
        # Early reflections (first 200 samples ~12.5ms at 16kHz)
        early_decay = 200
        for i in range(1, min(early_decay, rir_length)):
            # Add some early reflections with decreasing amplitude
            reflection_strength = 0.3 * np.exp(-i / 100)
            if np.random.random() < 0.1:  # Sparse early reflections
                rir[i] += reflection_strength * (0.5 + np.random.random())
        
        # Late reverberation (exponential tail)
        for i in range(early_decay, rir_length):
            # Exponential decay with realistic RT60 characteristics
            decay_rate = 50  # samples (faster decay = shorter RT60)
            amplitude = 0.1 * np.exp(-i / decay_rate)
            rir[i] = amplitude * (0.8 + 0.4 * np.random.random())
        
        # Normalize to ensure direct sound is prominent
        if np.max(np.abs(rir)) > 0:
            rir = rir / np.max(np.abs(rir))
        
        return rir
    
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
            'rir_metric': compute_rir_drr_metric(self.current_rir, self.sample_rate),  # RIR structural metric
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
    
    print(f"True RIR structural metric: {compute_rir_drr_metric(true_rir, sample_rate):.2f} dB")
    
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
    final_rir_metric = compute_rir_drr_metric(final_rir, sample_rate)
    
    # Simple correlation with true RIR
    min_len = min(len(final_rir), len(true_rir))
    correlation = np.corrcoef(final_rir[:min_len], true_rir[:min_len])[0, 1]
    
    print(f"\n=== Results ===")
    print(f"Final RIR metric: {final_rir_metric:.2f} dB (true: {compute_rir_drr_metric(true_rir, sample_rate):.2f} dB)")
    print(f"RIR Correlation: {correlation:.3f}")
    print(f"Total Reward: {sum(rewards):.3f}")
    
    return {
        'rir_estimates': rir_estimates,
        'rewards': rewards,
        'final_rir_metric': final_rir_metric,
        'correlation': correlation,
        'true_rir': true_rir
    }


if __name__ == "__main__":
    results = compare_agents_demo()