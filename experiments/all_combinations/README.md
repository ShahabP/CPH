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

## RIR length (note)

The codebase uses a unified RIR length of **4096 samples (256 ms at 16 kHz)** across all components:

- Environment: `RIREstimationEnv` default is `rir_length=4096`
- Neural policy: `NeuralRIRAgent` / `RIRPolicyNetwork` default is `rir_length=4096`
- Training orchestrator: uses `rir_length=4096` for all experiments

This length balances realism (captures typical room tail characteristics) with computational efficiency (manageable network sizes and training times). If you need longer or shorter RIRs for specific acoustic scenarios, update the defaults in `src/rl_framework/__init__.py`, `src/neural_rir_agent.py`, and `scripts/train_all_combinations.py` consistently.

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

## Episode flow, counts, and metrics

What happens during one episode
- Environment reset: each episode begins with `env.reset()` which either generates or loads a short synthetic reverberant speech segment together with a "true" RIR (used for metrics). The environment also initializes the current RIR estimate according to the chosen initialization method (random noise or exponential-decay prior).

- Per-step loop (QN / DQN): for up to `max_steps` (15 in the orchestrator) the agent:
	1. Observes the environment state (audio features + current RIR estimate + small metrics vector).
	2. Selects an action: QN/DQN choose a discrete action index; that index is mapped to a 6‑D continuous parameter vector (alpha, beta, regularization, step_size, w1, w2) which the environment uses to run its dereverberation / RIR estimation step.
	3. The environment applies the action, returns the next observation, a scalar reward, a termination flag, and info metrics.
	4. The agent stores the transition and (DQN) may sample from its replay buffer to update the Q-network; QN updates its Q-table immediately.
	5. If terminated (or max steps), the episode ends; epsilon is decayed for exploration.

- Per-step loop (Neural RIR Agent): for up to `max_iterations` (15 in the orchestrator) the neural agent:
	1. Receives the current RIR estimate as state and predicts a full-length RIR update vector (structured via direct/early/late/tail heads).
	2. The environment applies the update (with momentum and enforcement of a dry-structure), computes a DRR-based reward by dereverberating the speech with the new RIR, and returns observation/reward/info.
	3. The agent stores (state, action, value, reward) pairs and, at episode end, computes discounted returns and performs an actor-critic style update (actor loss + critic MSE).

How many episodes are run
- Per the orchestrator (`scripts/train_all_combinations.py`) the default counts used in the sweep are:
	- QN: 300 episodes per QN combination
	- DQN: 300 episodes per DQN combination
	- Neural: 200 episodes per Neural combination

- There are 6 method combinations (QN/DQN/Neural × Random/Exponential-decay). The script also sweeps 6 RT60 values (100, 300, 500, 700, 900, 1000 ms). For a single RT60 the total episodes are:
	- QN: 2 combinations × 300 = 600 episodes
	- DQN: 2 combinations × 300 = 600 episodes
	- Neural: 2 combinations × 200 = 400 episodes
	- Total per RT60 = 1,600 episodes

- Full sweep across the six RT60 values therefore runs 1,600 × 6 = 9,600 episodes (this is the full-batch run used to produce the earlier aggregated results; run times will scale accordingly).

What "correlation" means
- When the code reports `correlation` it is the Pearson correlation coefficient between the estimated RIR and the ground-truth RIR (computed over the overlapping length). It ranges from -1 (perfect inverse) to +1 (perfect match), with 0 meaning no linear correlation. The code computes it with NumPy's `np.corrcoef` on the two vectors and stores the [0,1] element.

What "average reward" means
- The `avg_reward` shown in the summaries is the mean of the episode total rewards taken over the final window of training (the code uses the last 50 episodes when available). The per-episode total reward equals the sum of the per-step rewards emitted by the environment during that episode.

- Environment reward (QN/DQN): the environment's reward function combines structural RIR metrics (correlation with ground truth when available, MSE penalty), peak/energy based rewards, and an improvement bonus across iterations. See `RIREstimationEnv._compute_reward` in the code for the exact composition.

- Neural agent reward: the neural agent uses a DRR-focused reward computed by dereverberating the speech using the current RIR estimate and computing a speech-based DRR metric (`_compute_speech_drr`). Positive (dryer) DRR yields strong positive reward; additional terms reward a strong direct tap and penalize early/late/tail energy and slow decay (see `NeuralRIRAgent.compute_reward`).

Notes on interpretation
- Correlation is a direct structural match between estimated and true RIRs — it is useful for RIR recovery evaluation but does not directly reflect perceptual dereverberation quality.
- Average reward is the algorithm's internal performance signal (aggregated across episodes) and combines many signals; improvements in `avg_reward` usually indicate the agent is learning policies that produce cleaner/drier dereverberation outputs under the chosen metric.
