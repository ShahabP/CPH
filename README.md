# Technical README: RL for Room Impulse Response (RIR) Estimation

This repository implements a reinforcement learning (RL) system that estimates room impulse responses from reverberant speech through iterative blind dereverberation, deconvolution, and online/streaming refinement.

## 1) Problem and signal model

We assume the acoustic convolution model:

- Reverberant speech: $y(t) = (h * s)(t) + n(t)$
	- $s(t)$: anechoic speech
	- $h(t)$: room impulse response (RIR)
	- $n(t)$: additive noise (optional)

We use STFT-based features $Y_{k,m} = \mathcal{STFT}\{y\}$ and derived magnitudes, mel spectra, etc. The goal is to estimate $\hat{h}$ and a dereverberated signal $\hat{s}$ from $y$ only.

## 2) Algorithmic building blocks

### 2.1 Blind dereverberation

Interchangeable methods (two classical + optional deep):

- Spectral subtraction: estimate reverberant/noise spectrum $\hat{N}_k$ and compute
	$\hat{S}_k = \max(|Y_k|^2 - \alpha\,\hat{N}_k, \beta\,|Y_k|^2)^{1/2} e^{j\angle Y_k}$
	with over-subtraction $\alpha$ and spectral floor $\beta$.

- Wiener filtering: estimate power spectra $\Phi_{SS}$ and $\Phi_{NN}$ and apply
	$G_k = \frac{\Phi_{SS}}{\Phi_{SS}+\Phi_{NN}}$ with $\hat{S}_k = G_k Y_k$.

- Deep dereverberation (optional): LSTM-based mask estimator $M_{k,m} \in [0,1]$ with $|\hat{S}| = M\,|Y|$. This path is enabled only if PyTorch is installed.

### 2.2 RIR estimation (deconvolution)

Given $y$ and a dereverberated estimate $\hat{s}$, estimate $h$ with:

- Wiener deconvolution (Wiener–Hopf): solve $R_{ss} h = r_{ys}$ where $R_{ss}$ is Toeplitz from the auto-correlation of $\hat{s}$ and $r_{ys}$ the cross-correlation of $y$ and $\hat{s}$; add $\lambda I$ for stability.

- LMS adaptive filtering: minimize $\|y - X h\|^2$ with
	$$h^{(t+1)} = h^{(t)} - \mu X^\top (X h^{(t)} - y),$$
	where $X$ is the Toeplitz matrix of $\hat{s}$ frames.

Multiple estimates can be linearly combined with learned weights for robustness.

### 2.3 RL loop (RIREstimationEnv)

- State $\mathbf{x}$: concat of
	- features $\phi(y)$ (pooled STFT magnitudes),
	- features $\phi(\hat{s})$,
	- current $\hat{h}$ (length $L$),
	- iteration progress and metrics (peak, energy, sparsity, correlation/MSE if available).

- Action $\mathbf{a} \in \mathbb{R}^6$ (continuous): controls dereverberation and RIR estimation, e.g., $\alpha,\beta$, regularization, LMS step size, method weights, and an optional blend factor for streaming.

- Reward $r$: shaped by RIR quality and improvement (positive for higher correlation/lower MSE, stability around peak amplitude, and per-iteration improvement).

- Agent: a compact DQN baseline. Since the env is continuous, a small codebook maps discrete actions to 6-D continuous vectors (fast baseline). You can swap in PPO/SAC (stable-baselines3) for true continuous control.

### 2.4 Streaming/online refinement (StreamingRIREstimationEnv)

The environment maintains a persistent global RIR estimate across segments and warm-starts each new segment from it. Each step blends new and global estimates:

$$\hat{h}_{\text{global}} \leftarrow (1-\alpha)\,\hat{h}_{\text{global}} + \alpha\,\hat{h}_{\text{segment}},$$

with $\alpha \in [0,1]$ exposed via the action vector. This enables convergence as more speech arrives.

## 3) Application pipeline

1. Input: reverberant speech $y$ (offline clip or streaming segment)
2. Blind dereverberation $\to$ $\hat{s}$
3. Deconvolution $\to$ $\hat{h}$
4. RL state/step and reward shaping
5. Streaming: carry and blend $\hat{h}$ across segments

## 4) Project structure

- `src/audio_processing/` — STFT, mel/features, convolution helpers, RT60
- `src/dereverberation/` — spectral subtraction, Wiener, optional deep mask
- `src/rir_estimation/` — Wiener deconv, LMS; optional neural estimator stub
- `src/rl_framework/` — Gymnasium envs (offline + streaming), DQN agent, trainer
- `train.py` — minimal 5-episode sanity trainer (codebook → continuous actions)
- `train_full.py` — YAML-config trainer, supports episodic and streaming, saves artifacts
- `configs/default_config.yaml` — training defaults (episodes/steps/streaming)
- `scripts/` — utilities: `smoke_test.py`, `stream_demo.py`, `generate_stream_segments.py`

## 5) How to run

Recommended: Python 3.10–3.12. A local virtual environment will be auto-configured in `.venv`.

1) Minimal dependencies (fast)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy==1.26.4 scipy==1.11.4 soundfile==0.12.1 librosa==0.10.2.post1 gymnasium==0.29.1
```

2) Smoke test

```bash
python scripts/smoke_test.py
```

3) Quick RL sanity run

```bash
python train.py
```

4) Streaming (online) refinement demo

```bash
python scripts/stream_demo.py
```

5) Full stack (optional)

```bash
pip install -r requirements.txt
```

6) Config-driven training

```bash
# Episodic
python train_full.py --config configs/default_config.yaml --output-dir experiments

# One-shot streaming on a folder of .wav segments
python scripts/generate_stream_segments.py
python train_full.py --config configs/default_config.yaml --output-dir experiments --stream-dir data/stream

# Continuous watch mode
python train_full.py --config configs/default_config.yaml --output-dir experiments --stream-dir data/stream --watch
```

## 6) Configuration

`configs/default_config.yaml` exposes:

- `environment`: iterations, RIR length, feature size, action space type
- `audio_processing`: sample rate, STFT params
- `dereverberation`: method selection
- `rir_estimation`: method + regularization
- `agent`: LR, gamma, epsilon schedule, memory size, discrete action count
- `training`: episodes, max steps
- `streaming`: enabled, stream_dir, poll interval

## 7) Metrics and evaluation

- Peak amplitude, total energy
- Sparsity of $\hat{h}$
- Correlation and NMSE vs. ground truth (if available)
- RT60 estimated from $\hat{h}$

## 8) Extensibility

- Add dereverberation/RIR methods by subclassing and wiring into the factory
- Swap the RL agent to PPO/SAC (stable-baselines3) for continuous actions
- Replace synthetic segment producer with real-time capture or dataset loader

## 9) Limitations and roadmap

- The DQN+codebook mapping is a fast baseline; PPO/SAC is preferable for continuous control
- Features are lightweight; richer features and context can improve stability
- Deep modules are optional and ship without pretrained weights

Roadmap:
- Integrate PPO/SAC agent
- Add dataset loaders and benchmarks
- Add online VAD/noise modeling

## 10) References (selected)

- P. A. Naylor and N. D. Gaubitch, “Speech Dereverberation,” Springer
- E. A. P. Habets, “Room Impulse Response (RIR) Generator”
- H. Erdogan et al., “Deep Learning-based Speech Dereverberation”
- Z. Koldovsky et al., “Blind Deconvolution and Dereverberation of Speech”
