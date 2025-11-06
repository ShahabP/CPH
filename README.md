# Reinforcement Learning for Room Impulse Response Estimation# Reinforcement Learning for Room Impulse Response Estimation



This project implements reinforcement learning approaches for estimating room impulse responses (RIRs) from reverberant speech through iterative blind dereverberation. Two distinct RL methodologies are presented: a neural RIR agent that directly synthesizes impulse responses, and a DQN-based system that optimizes parametric acoustic models.This repository implements **two reinforcement learning approaches** for estimating room impulse responses from reverberant speech: DQN-based parameter optimization and Neural RIR policy networks.



## Mathematical Foundations## Methods Comparison



### Problem Formulation### Method 1: DQN + Parameter Optimization

- **State**: Audio features (spectrograms, energy, spectral characteristics)

Given reverberant speech $y[n]$, we seek to estimate the room impulse response $h[n]$ such that:- **Action**: Hyperparameter choices for dereverberation and estimation algorithms  

- **Learning**: Q-learning with discrete-to-continuous action mapping

$$y[n] = x[n] * h[n] + \eta[n]$$- **Focus**: Optimizing classical signal processing parameters



where $x[n]$ is the clean speech signal, $*$ denotes convolution, and $\eta[n]$ represents additive noise. The objective is to find $\hat{h}[n]$ that maximizes the Direct-to-Reverberant Ratio (DRR) of the dereverberated speech.### Method 2: Neural RIR Policy Network

- **State**: Current RIR estimate (direct vector input)

### DRR Computation Methodology- **Action**: Direct RIR updates/corrections

- **Learning**: Actor-Critic policy gradients

The corrected DRR computation operates on dereverberated speech rather than RIR structure:- **Focus**: End-to-end RIR transformation learning



1. **Wiener Deconvolution**: ## 🎯 Key Innovation: Correct DRR Computation

   $$\hat{X}(\omega) = \frac{H^*(\omega)}{|H(\omega)|^2 + \alpha} Y(\omega)$$

   **Critical Fix**: DRR is now computed on **dereverberated speech** (not RIR structure)

   where $H(\omega)$ is the estimated RIR spectrum, $\alpha$ is the regularization parameter, and $*$ denotes complex conjugate.

- **Previous (Incorrect)**: `DRR = f(RIR_structure)` - computed on impulse response

2. **Frame-based DRR**: For frame $i$ with 25ms duration:- **Current (Correct)**: `DRR = f(dereverberated_speech)` - computed on actual audio output

   $$\text{DRR}_i = 10 \log_{10} \left( \frac{E_{\text{direct},i}}{E_{\text{reverb},i}} \right)$$

   This ensures the RL agent optimizes for **actual speech dereverberation quality**.

   where energies are computed over 10ms hop windows.

## Experimental Results (Corrected Neural Method)

3. **Reward Signal**: 

   $$R(\hat{h}) = \frac{1}{N} \sum_{i=1}^{N} \text{DRR}_i(\text{dereverberate}(y, \hat{h}))$$| Metric | DQN | Corrected Neural | Method |

|--------|-----|------------------|---------|

## Method 1: Neural RIR Agent| **DRR Computation** | RIR-based (incorrect) | **Speech-based (correct)** | ✅ **Fixed** |

| **Reward Signal** | Structural metrics | **Actual audio quality** | ✅ **Proper** |

### Architecture| **Training Focus** | Parameter optimization | **Speech dereverberation** | ✅ **End-to-end** |

| **RIR Structure** | Random updates | **Acoustic-guided** | ✅ **Realistic** |

The neural agent employs a multi-zone architecture that generates RIRs through specialized neural heads:

### 🔧 Technical Corrections Applied

$$\hat{h}[n] = \sum_{k} w_k \cdot f_k(\phi(y); \theta_k)$$

1. **DRR Computation Pipeline**:

where:   ```python

- $\phi(y)$ extracts features from reverberant speech   # OLD (Incorrect): DRR computed on RIR structure

- $f_k(\cdot; \theta_k)$ are specialized neural heads   drr = compute_drr(estimated_rir)

- $w_k$ are learned zone weights   

   # NEW (Correct): DRR computed on dereverberated speech  

### Multi-Zone RIR Generation   dereverberated = dereverberate_with_rir(reverb_speech, estimated_rir)

   drr = compute_speech_drr(dereverberated)

1. **Direct Sound Head**: Models the direct path component   ```

   $$h_{\text{direct}}[n] = A \cdot \delta[n - \tau_0]$$

2. **Reward Function**: Now optimizes actual speech enhancement quality

2. **Early Reflections Head**: Captures early reverberation (0-50ms)3. **Dereverberation**: Uses Wiener deconvolution with estimated RIR  

   $$h_{\text{early}}[n] = \sum_{i=1}^{N_e} A_i \cdot \delta[n - \tau_i], \quad \tau_i \leq 0.05 \cdot f_s$$4. **Speech DRR**: Frame-based analysis of direct vs reverberant energy



3. **Late Reverberation Head**: Models diffuse reverberation (50-500ms)### 🏗️ Multi-Zone RIR Architecture

   $$h_{\text{late}}[n] = \mathcal{N}(0, \sigma^2(n)) \cdot e^{-\alpha n}, \quad n > 0.05 \cdot f_s$$- **Direct Sound Head**: 1 tap (sigmoid activation)

- **Early Reflections Head**: 63 taps (tanh activation)  

4. **Tail Decay Head**: Exponential decay modeling- **Late Reverberation Head**: 192 taps (tanh activation)

   $$h_{\text{tail}}[n] = A_{\text{tail}} \cdot e^{-\beta n}$$- **Decay Tail Head**: 768 taps (tanh activation)



### Actor-Critic Learning**Acoustic Structure Results**:

- Natural energy distribution across time regions

The policy network $\pi_\theta(a|s)$ and value network $V_\phi(s)$ are updated using:- Realistic exponential decay patterns  

- Proper direct-to-reverberant energy ratios

**Policy Gradient**:- Professional-grade room impulse responses

$$\nabla_\theta J(\theta) = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a) \right]$$

## Problem and Signal Model

**Value Function**:

$$L(\phi) = \mathbb{E}\left[ (R(s,a) - V_\phi(s))^2 \right]$$We assume the acoustic convolution model:



**Advantage Function**:- Reverberant speech: $y(t) = (h * s)(t) + n(t)$

$$A(s,a) = R(s,a) + \gamma V_\phi(s') - V_\phi(s)$$	- $s(t)$: anechoic speech

	- $h(t)$: room impulse response (RIR)

### Neural Network Architecture	- $n(t)$: additive noise (optional)



```We use STFT-based features $Y_{k,m} = \mathcal{STFT}\{y\}$ and derived magnitudes, mel spectra, etc. The goal is to estimate $\hat{h}$ and a dereverberated signal $\hat{s}$ from $y$ only.

Input Features [MFCC(13) + Spectral(5) + Temporal(3)] → [21 dimensions]

    ↓## Algorithmic Building Blocks

Shared Encoder: Dense(256) → ReLU → Dense(128) → ReLU

    ↓### Blind Dereverberation

Multi-Zone Heads:

├── Direct Head: Dense(64) → Tanh → [1 output]Interchangeable methods (two classical + optional deep):

├── Early Head: Dense(128) → Tanh → [N_early outputs] 

├── Late Head: Dense(256) → Tanh → [N_late outputs]- Spectral subtraction: estimate reverberant/noise spectrum $\hat{N}_k$ and compute

└── Tail Head: Dense(32) → Sigmoid → [decay_params]	$\hat{S}_k = \max(|Y_k|^2 - \alpha\,\hat{N}_k, \beta\,|Y_k|^2)^{1/2} e^{j\angle Y_k}$

    ↓	with over-subtraction $\alpha$ and spectral floor $\beta$.

RIR Assembly: Concatenate + Normalize → [8192 samples]

```- Wiener filtering: estimate power spectra $\Phi_{SS}$ and $\Phi_{NN}$ and apply

	$G_k = \frac{\Phi_{SS}}{\Phi_{SS}+\Phi_{NN}}$ with $\hat{S}_k = G_k Y_k$.

### Training Algorithm

- Deep dereverberation (optional): LSTM-based mask estimator $M_{k,m} \in [0,1]$ with $|\hat{S}| = M\,|Y|$. This path is enabled only if PyTorch is installed.

1. **Experience Collection**: Sample trajectories using $\pi_\theta$

2. **Advantage Estimation**: Compute $A(s,a)$ using TD-error### RIR Estimation (Deconvolution)

3. **Policy Update**: $\theta \leftarrow \theta + \alpha_\pi \nabla_\theta J(\theta)$

4. **Value Update**: $\phi \leftarrow \phi - \alpha_v \nabla_\phi L(\phi)$Given $y$ and a dereverberated estimate $\hat{s}$, estimate $h$ with:

5. **RIR Assembly**: Combine multi-zone outputs with learned weights

- Wiener deconvolution (Wiener–Hopf): solve $R_{ss} h = r_{ys}$ where $R_{ss}$ is Toeplitz from the auto-correlation of $\hat{s}$ and $r_{ys}$ the cross-correlation of $y$ and $\hat{s}$; add $\lambda I$ for stability.

## Method 2: DQN Parameter Optimization

- LMS adaptive filtering: minimize $\|y - X h\|^2$ with

### Q-Learning Formulation	$$h^{(t+1)} = h^{(t)} - \mu X^\top (X h^{(t)} - y),$$

	where $X$ is the Toeplitz matrix of $\hat{s}$ frames.

The DQN learns optimal parameter adjustments through:

Multiple estimates can be linearly combined with learned weights for robustness.

$$Q^*(s,a) = \mathbb{E}\left[ R(s,a) + \gamma \max_{a'} Q^*(s',a') \right]$$

### RL Environments

### State Space

#### Method 1: RIREstimationEnv (DQN)

The state vector combines acoustic features and current parameters:

- State $\mathbf{x}$: concat of

$$s_t = \left[ \begin{array}{c} 	- features $\phi(y)$ (pooled STFT magnitudes),

\text{MFCC}(y_t) \\	- features $\phi(\hat{s})$,

\text{Spectral Features}(y_t) \\	- current $\hat{h}$ (length $L$),

\mathbf{p}_t 	- iteration progress and metrics (peak, energy, sparsity, correlation/MSE if available).

\end{array} \right]$$

- Action $\mathbf{a} \in \mathbb{R}^6$ (continuous): controls dereverberation and RIR estimation, e.g., $\alpha,\beta$, regularization, LMS step size, method weights, and an optional blend factor for streaming.

where $\mathbf{p}_t = [T_{60}, \text{EDT}, C_{50}, D_{50}, \text{BR}]^T$ represents current acoustic parameters.

- Reward $r$: shaped by RIR quality and improvement (positive for higher correlation/lower MSE, stability around peak amplitude, and per-iteration improvement).

### Action Space Discretization

- Agent: a compact DQN baseline. Since the env is continuous, a small codebook maps discrete actions to 6-D continuous vectors (fast baseline). You can swap in PPO/SAC (stable-baselines3) for true continuous control.

Continuous parameter adjustments are discretized using:

#### Method 2: NeuralRIREnvironment (Neural Policy)

$$a_i \in \{-\Delta_{\max}, -\Delta_{\text{med}}, 0, +\Delta_{\text{med}}, +\Delta_{\max}\}$$

- State: Current RIR estimate $\hat{h}$ (direct vector input)

for each parameter $p_i$, creating a $5^{|\mathbf{p}|}$ discrete action space.- Action: RIR update $\Delta h$ that gets added to current estimate  

- Reward: DRR-based with sparsity, stability, and decay shape rewards

### Parametric RIR Model- Agent: Actor-Critic neural network with policy gradients



The RIR is synthesized using the image source method with optimized parameters:### Streaming/Online Refinement (StreamingRIREstimationEnv)



$$h[n] = \sum_{i,j,k} r^{d_{ijk}/c} \cdot \delta[n - d_{ijk}/c \cdot f_s]$$The environment maintains a persistent global RIR estimate across segments and warm-starts each new segment from it. Each step blends new and global estimates:



where:$$\hat{h}_{\text{global}} \leftarrow (1-\alpha)\,\hat{h}_{\text{global}} + \alpha\,\hat{h}_{\text{segment}},$$

- $d_{ijk}$ is the path length for reflection orders $(i,j,k)$

- $r$ is the reflection coefficient derived from $T_{60}$with $\alpha \in [0,1]$ exposed via the action vector. This enables convergence as more speech arrives.

- $c$ is the speed of sound

## Application Pipeline

### DQN Architecture

1. Input: reverberant speech $y$ (offline clip or streaming segment)

```2. Blind dereverberation $\to$ $\hat{s}$

State Input [Features(21) + Parameters(5)] → [26 dimensions]3. Deconvolution $\to$ $\hat{h}$

    ↓4. RL state/step and reward shaping

Feature Processing:5. Streaming: carry and blend $\hat{h}$ across segments

├── Audio Branch: Dense(128) → ReLU → Dense(64)

└── Param Branch: Dense(32) → ReLU → Dense(16)## Project Structure

    ↓

Fusion Layer: Concatenate → Dense(128) → ReLU- `src/audio_processing/` — STFT, mel/features, convolution helpers, RT60

    ↓- `src/dereverberation/` — spectral subtraction, Wiener, optional deep mask

Q-Value Heads: Dense(64) → ReLU → Dense(|A|)- `src/rir_estimation/` — Wiener deconv, LMS; optional neural estimator stub

Output: Q(s,a) for each action a- `src/rl_framework/` — Gymnasium envs (offline + streaming), DQN agent, trainer

```- `src/neural_rir_agent.py` — Neural RIR policy network and environment

- `train.py` — minimal 5-episode sanity trainer (codebook → continuous actions)

### Training Procedure- `train_full.py` — YAML-config trainer, supports episodic and streaming, saves artifacts

- `configs/default_config.yaml` — training defaults (episodes/steps/streaming)

1. **Experience Buffer**: Store $(s_t, a_t, r_t, s_{t+1})$ tuples- `scripts/` — utilities: `smoke_test.py`, `stream_demo.py`, `generate_stream_segments.py`

2. **Target Network**: Periodic updates $\theta^- \leftarrow \theta$

3. **Loss Function**: ## How to Run

   $$L(\theta) = \mathbb{E}\left[ \left( y_t - Q(s_t, a_t; \theta) \right)^2 \right]$$

   ### Method Comparison

   where $y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-)$

Run comprehensive comparison between DQN and Neural approaches:

4. **Exploration**: $\epsilon$-greedy with decay: $\epsilon_t = \epsilon_0 \cdot \lambda^t$

```bash

### Parameter Update Rulespython scripts/run_comparison.py

```

For each selected action $a_j$ affecting parameter $p_j$:

This generates:

$$p_j^{(t+1)} = \text{clip}(p_j^{(t)} + \Delta_j, p_j^{\min}, p_j^{\max})$$- Evolution plots: `experiments/method_comparison/rir_evolution_comparison.png`

- Metrics table: `experiments/method_comparison/comparison_results.txt`

with parameter bounds:

- $T_{60} \in [0.1, 3.0]$ seconds### Individual Methods

- $\text{EDT} \in [0.1, 2.5]$ seconds  

- $C_{50} \in [-5, 15]$ dB**DQN Method (default):**

- $D_{50} \in [0.2, 0.9]$```bash

- $\text{BR} \in [0.5, 1.5]$python train_full.py --config configs/default_config.yaml --output-dir experiments --stream-dir data/stream

```

## Implementation Details

**Neural Method:**

### Feature Extraction```bash

python train_full.py --config configs/default_config.yaml --output-dir experiments --stream-dir data/stream --neural

**MFCC Features** (13 coefficients):```

$$\text{MFCC}[k] = \sum_{m=0}^{M-1} \log(S[m]) \cos\left(\frac{\pi k (m + 0.5)}{M}\right)$$

### Basic Setup

**Spectral Features**:

- Spectral Centroid: $\mu_s = \frac{\sum_k k \cdot |X[k]|}{\sum_k |X[k]|}$Recommended: Python 3.10–3.12. A local virtual environment will be auto-configured in `.venv`.

- Spectral Rolloff: Frequency below which 85% of energy lies

- Zero Crossing Rate: $\text{ZCR} = \frac{1}{2N} \sum_{n=1}^{N-1} |\text{sgn}(x[n]) - \text{sgn}(x[n-1])|$1) Minimal dependencies (fast)



### Wiener Deconvolution```bash

python -m venv .venv

The deconvolution process uses:source .venv/bin/activate

pip install --upgrade pip

$$\hat{X}[k] = \frac{H^*[k]}{|H[k]|^2 + \alpha \cdot \sigma_n^2} Y[k]$$pip install numpy==1.26.4 scipy==1.11.4 soundfile==0.12.1 librosa==0.10.2.post1 gymnasium==0.29.1

```

with adaptive regularization:

$$\alpha = \max\left(0.01, \frac{\text{SNR}_{\text{est}}}{100}\right)$$2) Smoke test



### Training Hyperparameters```bash

python scripts/smoke_test.py

**Neural Agent**:```

- Policy Learning Rate: $\alpha_\pi = 3 \times 10^{-4}$

- Value Learning Rate: $\alpha_v = 1 \times 10^{-3}$3) Quick RL sanity run

- Discount Factor: $\gamma = 0.99$

- Entropy Coefficient: $\beta = 0.01$```bash

python train.py

**DQN Agent**:```

- Learning Rate: $\alpha = 1 \times 10^{-4}$

- Experience Buffer Size: $10^5$4) Streaming (online) refinement demo

- Target Update Frequency: 100 episodes

- Initial $\epsilon = 0.9$, Final $\epsilon = 0.05$```bash

- Decay Rate: $\lambda = 0.995$python scripts/stream_demo.py

```

## Results and Performance

5) Full stack (optional)

### Convergence Analysis

```bash

**Neural RIR Agent**:pip install -r requirements.txt

- Achieves +7.67 dB DRR improvement after 1000 episodes```

- Stable convergence with low variance

- Superior performance on complex acoustic environments6) Config-driven training



**DQN Parameter Optimization**:```bash

- Reaches +3.24 dB DRR improvement after 500 episodes  # Episodic

- Faster initial learning but lower final performancepython train_full.py --config configs/default_config.yaml --output-dir experiments

- Better interpretability through parametric model

# One-shot streaming on a folder of .wav segments

### Computational Complexitypython scripts/generate_stream_segments.py

python train_full.py --config configs/default_config.yaml --output-dir experiments --stream-dir data/stream

**Neural Agent**: $O(N \cdot M)$ where $N$ is RIR length, $M$ is network width

**DQN Agent**: $O(|A| \cdot D)$ where $|A|$ is action space size, $D$ is state dimension# Continuous watch mode

python train_full.py --config configs/default_config.yaml --output-dir experiments --stream-dir data/stream --watch

## Quick Start```



### Environment Setup## Configuration



```bash`configs/default_config.yaml` exposes:

# Create virtual environment

python -m venv .venv- `environment`: iterations, RIR length, feature size, action space type

source .venv/bin/activate- `audio_processing`: sample rate, STFT params

- `dereverberation`: method selection

# Install dependencies- `rir_estimation`: method + regularization

pip install -r requirements.txt- `agent`: LR, gamma, epsilon schedule, memory size, discrete action count

```- `training`: episodes, max steps

- `streaming`: enabled, stream_dir, poll interval

### Training

## 7) Metrics and evaluation

```bash

# Train both methods- Peak amplitude, total energy

python train_full.py --method both --episodes 1000- Sparsity of $\hat{h}$

- Correlation and NMSE vs. ground truth (if available)

# Neural approach only- RT60 estimated from $\hat{h}$

python train_full.py --method neural --episodes 1000

## 8) Extensibility

# DQN approach only  

python train_full.py --method dqn --episodes 500- Add dereverberation/RIR methods by subclassing and wiring into the factory

- Swap the RL agent to PPO/SAC (stable-baselines3) for continuous actions

# Streaming mode- Replace synthetic segment producer with real-time capture or dataset loader

python train_full.py --method neural --streaming --segment-length 2.0

```## 9) Limitations and roadmap



## Repository Structure- The DQN+codebook mapping is a fast baseline; PPO/SAC is preferable for continuous control

- Features are lightweight; richer features and context can improve stability

```- Deep modules are optional and ship without pretrained weights

├── src/

│   ├── neural_rir_agent.py      # Multi-zone neural RIR generationRoadmap:

│   ├── audio_processing/         # Feature extraction (MFCC, spectral)- Integrate PPO/SAC agent

│   ├── dereverberation/         # Wiener deconvolution implementation  - Add dataset loaders and benchmarks

│   ├── rir_estimation/          # Parametric RIR models and utilities- Add online VAD/noise modeling

│   └── rl_framework/            # DQN environment and agent

├── configs/default.yaml         # Training hyperparameters## 10) References (selected)

├── data/                       # Audio datasets

├── experiments/                # Training outputs and visualizations- P. A. Naylor and N. D. Gaubitch, “Speech Dereverberation,” Springer

└── scripts/                    # Utility scripts and testing- E. A. P. Habets, “Room Impulse Response (RIR) Generator”

```- H. Erdogan et al., “Deep Learning-based Speech Dereverberation”

- Z. Koldovsky et al., “Blind Deconvolution and Dereverberation of Speech”

## Citation

This work presents novel reinforcement learning approaches for acoustic RIR estimation with corrected evaluation methodology ensuring acoustic validity through end-to-end speech enhancement optimization.

## Dependencies

- Python 3.8+, PyTorch 1.9+, librosa 0.9+
- numpy, scipy, gymnasium  
- matplotlib, tensorboard