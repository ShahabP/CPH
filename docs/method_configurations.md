# Method Configurations and Hyperparameters

## Comprehensive Comparison: Q-Learning (QN), Deep Q-Network (DQN), and Neural RIR Agent

---

## Table 1: Agent Architecture & Core Parameters

| **Parameter** | **Q-Learning (QN)** | **Deep Q-Network (DQN)** | **Neural RIR Agent** |
|---------------|---------------------|--------------------------|----------------------|
| **Agent Type** | Tabular Q-Learning | Deep Q-Network | Policy Gradient (Actor-Critic) |
| **State Representation** | Discretized (binned) | Continuous (high-dimensional) | Direct RIR vector |
| **Action Space** | Discrete (12 actions) | Discrete (12 actions) | Continuous RIR updates |
| **Function Approximation** | Q-table (lookup) | Deep Neural Network | Policy Network + Value Network |
| **Network Architecture** | N/A | 4-layer MLP [768→512→256→128] | Multi-head architecture (1024 hidden dim) |

---

## Table 2: Learning Parameters

| **Parameter** | **Q-Learning (QN)** | **Deep Q-Network (DQN)** | **Neural RIR Agent** |
|---------------|---------------------|--------------------------|----------------------|
| **Learning Rate (α)** | 0.15 | 5×10⁻⁴ | 2×10⁻⁴ |
| **Discount Factor (γ)** | 0.99 | 0.99 | 0.99 |
| **Initial Epsilon (ε₀)** | 1.0 | 1.0 | N/A (exploration via noise) |
| **Epsilon Decay** | 0.9975 per episode | 0.9975 per episode | N/A |
| **Minimum Epsilon (ε_min)** | 0.01 | 0.01 | N/A |
| **Exploration Strategy** | ε-greedy | ε-greedy | Gaussian noise (σ=0.01) |
| **Weight Decay** | N/A | N/A | 1×10⁻⁵ |

---

## Table 3: Training Configuration

| **Parameter** | **Q-Learning (QN)** | **Deep Q-Network (DQN)** | **Neural RIR Agent** |
|---------------|---------------------|--------------------------|----------------------|
| **Episodes (Standard)** | 1200 | 1200 | 1200 |
| **Max Steps per Episode** | 15 | 15 | 15 |
| **Replay Buffer Size** | N/A | 50,000 | 10,000 (episode buffer) |
| **Batch Size** | N/A | 64 | 32 |
| **Target Network** | N/A | Yes (update every 10 episodes) | No |
| **Learning Rate Scheduler** | No | No | Yes (ReduceLROnPlateau) |
| **Optimizer** | N/A (direct update) | Adam | Adam |

---

## Table 4: State Space Configuration

| **Feature** | **Q-Learning (QN)** | **Deep Q-Network (DQN)** | **Neural RIR Agent** |
|-------------|---------------------|--------------------------|----------------------|
| **State Dimension** | 7 features × bins | ~530 dimensions | RIR length (256-4096) |
| **State Features** | reverb_energy (8 bins)<br>dereverb_energy (8 bins)<br>rir_peak (10 bins)<br>rir_energy (8 bins)<br>rir_sparsity (7 bins)<br>correlation (10 bins)<br>iteration (5 bins) | Raw feature vector:<br>- Spectral features<br>- Energy ratios<br>- Correlation<br>- Iteration count | Direct RIR vector:<br>- Time-domain samples<br>- Normalized to [-1, 1] |
| **State Discretization** | Manual binning | None (continuous) | None (continuous) |
| **Total States** | ~10⁷ (sparse) | Continuous | Continuous |

---

## Table 5: Action Space Configuration

| **Parameter** | **Q-Learning (QN)** | **Deep Q-Network (DQN)** | **Neural RIR Agent** |
|---------------|---------------------|--------------------------|----------------------|
| **Action Type** | Discrete parameters | Discrete parameters | Continuous RIR updates |
| **Number of Actions** | 12 | 12 | RIR length (continuous) |
| **Action Parameters** | 6D vector per action:<br>- α (spectral): [0.5, 3.0]<br>- β (noise floor): [0.001, 0.1]<br>- regularization: [0.0001, 0.1]<br>- step_size: [0.0001, 0.1]<br>- w₁, w₂ (weights): [0, 1] | Same as QN | Direct RIR updates:<br>- Direct sound (1 tap)<br>- Early reflections (63 taps)<br>- Late reverb (192 taps)<br>- Tail decay (remaining) |
| **Action Mapping** | Codebook lookup | Codebook lookup | Neural network output |
| **Update Scale** | As specified | As specified | 0.05 (configurable) |

---

## Table 6: Network Architecture Details (Neural RIR Agent Only)

| **Component** | **Architecture** | **Output Dimension** | **Activation** |
|---------------|------------------|----------------------|----------------|
| **Input Normalization** | LayerNorm | RIR length | - |
| **Encoder 1** | Linear + LayerNorm + ReLU + Dropout(0.1) | 1024 | ReLU |
| **Encoder 2** | Linear + LayerNorm + ReLU + Dropout(0.1) | 1024 | ReLU |
| **Encoder 3** | Linear + LayerNorm + ReLU | 512 | ReLU |
| **Direct Sound Head** | Linear(512→32) + ReLU + Linear(32→1) | 1 | Sigmoid |
| **Early Reflections Head** | Linear(512→128) + ReLU + Linear(128→64) + Linear(64→63) | 63 | Tanh |
| **Late Reverb Head** | Linear(512→128) + ReLU + Linear(128→64) + Linear(64→192) | 192 | Tanh |
| **Tail Decay Head** | Linear(512→128) + ReLU + Linear(128→64) + Linear(64→RIR_length-256) | RIR_length-256 | Tanh |
| **Value Head** | Linear(512→128) + ReLU + Linear(128→1) | 1 | Linear |

---

## Table 7: Memory & Computational Requirements

| **Metric** | **Q-Learning (QN)** | **Deep Q-Network (DQN)** | **Neural RIR Agent** |
|------------|---------------------|--------------------------|----------------------|
| **Model Size** | ~1-10 MB (Q-table) | ~5-10 MB (network weights) | ~15-20 MB (network weights) |
| **Memory Usage** | Low (sparse Q-table) | Medium (replay buffer) | Medium (episode buffer) |
| **Training Time (1200 eps)** | ~10-15 minutes | ~20-30 minutes | ~30-45 minutes |
| **GPU Required** | No | Optional (faster) | Recommended |
| **Inference Speed** | Very fast (lookup) | Fast (forward pass) | Fast (forward pass) |

---

## Table 8: Performance Characteristics

| **Metric** | **Q-Learning (QN)** | **Deep Q-Network (DQN)** | **Neural RIR Agent** |
|------------|---------------------|--------------------------|----------------------|
| **Average DRR (256 samples)** | ~7-10 dB | ~12-15 dB | ~27 dB |
| **Average DRR (2048 samples)** | ~5-8 dB | ~10-13 dB | ~23 dB |
| **Success Rate (≥7 dB)** | 60-80% | 80-90% | 100% |
| **Convergence Speed** | Slow (500-800 eps) | Medium (300-500 eps) | Fast (100-200 eps) |
| **Generalization** | Poor (discrete states) | Good | Excellent |
| **Stability** | High (stable updates) | Medium (replay helps) | High (policy gradient) |

---

## Table 9: Initialization Methods

| **Method** | **Description** | **All Agents** |
|------------|-----------------|----------------|
| **Random** | `np.random.normal(0, 0.1, rir_length)` | Poor performance (0.6-2.9 dB) |
| **Exponential Decay** | Realistic RIR with exponential decay:<br>- Direct sound at t=0<br>- Exponential decay based on RT60<br>- `rir[n] = δ[n] + noise × exp(-n/τ)`<br>- τ = -RT60 / (3 × log10(e)) | Excellent performance (20-28 dB) |

---

## Table 10: Environment Configuration (Common for All Agents)

| **Parameter** | **Value** | **Description** |
|---------------|-----------|-----------------|
| **Max Iterations** | 15 | Maximum dereverberation steps per episode |
| **RIR Length** | 256, 512, 1024, 2048 samples | 16, 32, 64, 128 ms @ 16 kHz |
| **Sample Rate** | 16,000 Hz | Audio sampling rate |
| **RT60 Range** | 100-800 ms | Reverberation time range tested |
| **Feature Dimension** | 513 | STFT feature size (for QN/DQN) |
| **Direct Window** | 2.5 ms | Window for direct sound in DRR computation |
| **Reward Function** | DRR-based | Higher DRR = higher reward |

---

## Table 11: Key Improvements (Enhanced vs Baseline)

| **Improvement** | **Q-Learning (QN)** | **Deep Q-Network (DQN)** | **Neural RIR Agent** |
|-----------------|---------------------|--------------------------|----------------------|
| **Episodes** | 600 → 1200 | 600 → 1200 | 1000 → 1200 |
| **State Bins** | 5 → 8-10 per feature | N/A | N/A |
| **Action Space** | 8 → 12 actions | 8 → 12 actions | N/A |
| **Network Size** | N/A | [512, 256, 128] → [768, 512, 256, 128] | 768 → 1024 hidden dim |
| **Learning Rate** | 0.1 → 0.15 | 1×10⁻³ → 5×10⁻⁴ | 1×10⁻⁴ → 2×10⁻⁴ |
| **Replay Buffer** | N/A | 10k → 50k | N/A |
| **Batch Size** | N/A | 32 → 64 | 32 (unchanged) |
| **Epsilon Decay** | 0.995 → 0.9975 | 0.995 → 0.9975 | N/A |

---

## Summary

### Q-Learning (QN)
- **Best for**: Simple problems, interpretable policy
- **Strengths**: Stable, no neural network complexity
- **Weaknesses**: Poor generalization, limited by discretization
- **Performance**: Moderate (7-10 dB DRR)

### Deep Q-Network (DQN)
- **Best for**: Moderate complexity, balance of performance and speed
- **Strengths**: Better generalization than QN, continuous state space
- **Weaknesses**: Requires tuning, replay buffer overhead
- **Performance**: Good (12-15 dB DRR)

### Neural RIR Agent
- **Best for**: Maximum performance, complex acoustic modeling
- **Strengths**: Direct RIR manipulation, excellent generalization, highest DRR
- **Weaknesses**: Most computationally intensive, requires careful initialization
- **Performance**: Excellent (23-27 dB DRR)

---

## Recommended Usage

1. **For research/experiments**: Neural RIR Agent (best performance)
2. **For production/real-time**: DQN (good balance)
3. **For debugging/baseline**: Q-Learning (simplest)

## Citation

If you use these configurations, please cite:
```
Reinforcement Learning for Room Impulse Response Estimation
https://github.com/ShahabP/Copenhagen
```
