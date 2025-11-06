# Reinforcement Learning for Room Impulse Response Estimation

This repository implements **two reinforcement learning approaches** for estimating room impulse responses from reverberant speech through iterative blind dereverberation.

## Problem Statement

Given reverberant speech $y[n] = x[n] * h[n] + \eta[n]$, estimate the room impulse response $h[n]$ to maximize the Direct-to-Reverberant Ratio (DRR) of the dereverberated speech.

## Algorithm Overview

### Method 1: Neural RIR Policy Network

**Direct end-to-end RIR estimation using deep reinforcement learning.**

#### Step-by-Step Algorithm:

1. **Initialize**: Random RIR estimate $h_0$ (1024 samples, 64ms at 16kHz)

2. **For each episode**:
   - Input: Reverberant speech $y[n]$ (2-3 seconds)
   - Current RIR estimate $h_t$ → Neural network
   - Output: RIR update $\Delta h$ via 4 specialized heads:
     - **Direct Head**: 1 tap (direct sound)
     - **Early Head**: 63 taps (early reflections, 0-4ms) 
     - **Late Head**: 192 taps (late reverberation, 4-12ms)
     - **Tail Head**: 768 taps (exponential decay, >12ms)

3. **Apply Update**: $h_{t+1} = h_t + \alpha \cdot \Delta h$

4. **Dereverberate**: $\hat{s}[n] = \text{Wiener}(y[n], h_{t+1})$
   $$\hat{S}(\omega) = \frac{H^*(\omega)}{|H(\omega)|^2 + \lambda} Y(\omega)$$

5. **Compute Reward**: DRR on dereverberated speech
   $$\text{DRR} = 10\log_{10}\left(\frac{E_{\text{direct}}}{E_{\text{reverb}}}\right)$$

6. **Learn**: Actor-critic update using policy gradients

### Method 2: DQN Parameter Optimization

**Discrete action space for hyperparameter tuning.**

#### Step-by-Step Algorithm:

1. **State**: Audio features (MFCC, spectral, temporal) → 21 dimensions

2. **Actions**: Discrete choices for algorithm parameters:
   - Dereverberation: α (over-subtraction), β (spectral floor)
   - RIR estimation: regularization λ, adaptation μ

3. **Q-Network**: Maps states to Q-values for each parameter choice

4. **Episode Loop**:
   - Extract features from reverberant speech
   - Select parameters via ε-greedy policy
   - Apply spectral subtraction + Wiener deconvolution
   - Compute reward based on dereverberation quality
   - Update Q-network using experience replay

## Performance Results

| Method | DRR Improvement | Training Episodes | Convergence |
|--------|----------------|-------------------|-------------|
| **Neural RIR Agent** | **+7.67 dB** | 500-1000 | 15-50 steps |
| **DQN Optimization** | **+3.24 dB** | 1000-2000 | 200 steps |

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/ShahabP/Copenhagen.git
cd Copenhagen

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### Method Comparison
Run both Neural RIR Agent and DQN Parameter Optimization:
```bash
python train_full.py --compare
```

#### Individual Training
```bash
# Neural RIR Agent (recommended)
python train_full.py --neural --episodes 500

# DQN Parameter Optimization  
python train_full.py --dqn --episodes 1000
```

#### Quick Test
```bash
# Verify installation
python scripts/smoke_test.py

# Generate visualizations  
python scripts/create_rir_visualizations.py
```

## Project Structure

```
src/
├── audio_processing/     # STFT, features, RT60 estimation
├── dereverberation/      # Spectral subtraction, Wiener filtering
├── rir_estimation/       # Deconvolution algorithms
├── rl_framework/         # DQN agent and environments
└── neural_rir_agent.py   # Neural policy network

scripts/
├── train_full.py         # Main training script
├── smoke_test.py         # Installation verification
└── create_*.py           # Visualization generators

configs/
└── default_config.yaml   # Training parameters
```

## Configuration

Key parameters in `configs/default_config.yaml`:

```yaml
environment:
  max_iterations: 15        # Steps per episode
  rir_length: 1024         # RIR samples (64ms at 16kHz)

training:
  episodes: 1000           # Neural agent episodes
  max_steps: 200          # DQN episodes

agent:
  learning_rate: 0.001    # Learning rate
  gamma: 0.99            # Discount factor
```

## Key Features

- **Multi-zone RIR synthesis**: Direct sound, early reflections, late reverberation, exponential tail
- **Wiener deconvolution**: Frequency domain dereverberation with regularization  
- **Frame-based DRR**: Perceptually relevant reward computed on actual speech output
- **Actor-critic learning**: Stable policy gradients for continuous RIR space
- **Experience replay**: Efficient Q-learning with target network stabilization

## Technical Details

- **RIR Length**: 1024 samples (64ms) captures essential room acoustics
- **Speech Length**: 2-3 seconds provides robust training signal  
- **Sample Rate**: 16kHz standard for speech processing
- **Dereverberation**: Wiener filtering with adaptive regularization:
  $$\hat{S}(\omega) = \frac{H^*(\omega)}{|H(\omega)|^2 + \lambda} Y(\omega)$$

## Dependencies

Core requirements:
- `numpy>=1.26.4`
- `scipy>=1.11.4`  
- `librosa>=0.10.2`
- `torch>=2.0.0` (for Neural RIR Agent)
- `gymnasium>=0.29.1` (for RL environments)

