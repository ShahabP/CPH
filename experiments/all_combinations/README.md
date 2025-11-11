# All combinations — overview

This folder documents the six experimental configurations (QN / DQN / Neural agents × Random / Exponential-decay initialization). Aggregate result files and visualizations were intentionally removed to keep the repository lightweight; they can be restored from git history if needed.

Purpose: explain the approaches and initialization choices so experiments can be reproduced with `scripts/train_all_combinations.py`.

## Methods (short)

- QN (tabular Q-Learning): discrete-state/action baseline that updates a Q-table via temporal-difference learning. Good for low-dimensional, discretized problems and very fast iteration.

- DQN (Deep Q-Network): neural-network function approximator for Q-values. Uses replay buffers and a target network to stabilize learning; appropriate for larger observation vectors (features + RIR estimate).

- Neural RIR Agent: a learned continuous agent (policy/regressor) that predicts direct updates to the RIR estimate. Trained end-to-end to minimize quality losses (e.g., correlation or MSE with a reference) rather than selecting discrete algorithmic actions.

## Initialization strategies

- Random: initialize the RIR estimate with small Gaussian noise. Minimal prior, slower to converge.

- Exponential-decay: seed the RIR with a direct impulse plus an exponentially decaying tail and sparse early reflections. Provides a strong acoustic prior (direct path + decay) and typically helps convergence and final quality.

## Reproducing experiments

Run the orchestrator (virtualenv example):

```bash
/Users/shahabpasha/Sonnet/Copenhagen/.venv/bin/python scripts/train_all_combinations.py
```

Tips:
- Start with a single RT60 value to smoke-test and confirm outputs before running the full sweep.
- The training code now saves final RIR estimates — make sure you are on the updated branch.

If you'd like, I can add a short "quick test" command that runs one agent for a few episodes and demonstrates where the final RIR is saved.

## Agent, reward & policy (detailed)

Below are concise, implementation-aligned descriptions of each method: the agent, its action/policy, and the reward signal that drives learning.

### QN (Tabular Q-Learning)
- Agent: `QNAgent` — tabular Q-learning with a discretized state representation (aggregated features: reverb/dereverb energies, RIR peak/energy/sparsity, correlation, iteration progress).
- Action: discrete index (default 8). Each index maps to a continuous 6-d parameter vector via an action codebook (alpha, beta, regularization, step_size, w1, w2) normalized to [-1,1].
- Policy: epsilon-greedy over the Q-table; epsilon decays each episode.
- Reward: the environment reward (correlation/MSE/improvement/peak-driven) — positive correlation and improvement increase reward; MSE and poor RIR structure reduce it.

### DQN (Deep Q-Network)
- Agent: `DQNAgent` — neural-network approximator (`DQNNetwork`) that outputs Q-values for each discrete action.
- Action: discrete index mapped through the same 6-d action codebook as QN.
- Policy: epsilon-greedy with greedy selection via q_network(state). Epsilon decays during training.
- Reward: identical environment reward as QN (driven by RIR quality, correlation, MSE, and improvement signals).

### Neural RIR Agent (continuous policy)
- Agent: `NeuralRIRAgent` — policy + value network (`RIRPolicyNetwork`) that directly outputs a structured RIR update vector (separate heads for direct tap, early reflections, late reverb, tail) and a scalar value estimate.
- Action: continuous full-length RIR update (applied as update_scale * network_output). Small Gaussian noise is added during training for exploration.
- Policy: actor-critic / policy-gradient style; actions are deterministic network outputs with exploratory noise during training.
- Reward: DRR-focused reward computed by dereverberating speech with the estimated RIR and scoring the result (strong positive reward for positive DRR, large penalties for negative DRR), plus structural terms that reward strong direct tap and penalize excessive early/late/tail energy and slow decay.

## Compact comparison table

| Method | Agent type | Action space | Policy | Reward signal | Learning algorithm |
|---|---:|---|---|---|---|
| QN | Tabular Q-Learning (`QNAgent`) | Discrete index → 6‑D parameter codebook | Epsilon-greedy over Q-table | Env reward (correlation/MSE/improvement/peak) | Q-learning TD updates (alpha, gamma); epsilon decay |
| DQN | Deep Q-Network (`DQNAgent`) | Discrete index → 6‑D parameter codebook | Epsilon-greedy; NN argmax for greedy action | Env reward (same as QN) | DQN with replay, target network, Adam optimizer, gradient clipping |
| Neural | Neural policy (`NeuralRIRAgent`) | Continuous RIR update vector (full length) | Deterministic NN output + exploration noise; actor-critic updates | DRR-based reward on dereverberated speech + structure penalties (direct/early/late/tail) | Policy-gradient style: actor loss (advantage-weighted), critic MSE; Adam, scheduler |

If you'd like, I can integrate this table and the per-method descriptions into the top-level `README.md` as well, or add a one-line quick-test example below to show the exact command for a single-agent smoke test.
