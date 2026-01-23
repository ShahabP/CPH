# Algorithm Section - Key Components and Flow

## Document Overview

The LaTeX algorithm section (`docs/algorithm_section.tex`) provides a complete, publication-ready description of the neural reinforcement learning method for blind RIR estimation. The document includes 40 numbered equations that are systematically referenced throughout the text.

## Structure and Flow

### 1. Problem Formulation (Section 3.1)
- **Equation 1**: Convolution model $y(t) = x(t) * h(t) + n(t)$
- Establishes the blind estimation problem
- Defines discrete RIR representation and sampling parameters

### 2. RL Framework (Section 3.2)
- Formalizes as MDP: state (RIR), action (update), reward (acoustic quality)
- Sets up iterative refinement over K=15 steps
- Discount factor γ=0.99

### 3. Neural Network Architecture (Section 3.3)

#### Encoder (Equations 2-4)
- **Equation 2**: First encoder layer with LayerNorm and ReLU
- **Equation 3**: Second layer with residual connection
- **Equation 4**: Feature extraction layer
- Hidden dimension d=1024 for high capacity

#### Structured Update Generation (Equations 5-9)
- **Equation 5**: Direct sound head (1 sample, sigmoid activation)
- **Equation 6**: Early reflections head (63 samples, tanh)
- **Equation 7**: Late reverberation head (192 samples, tanh)
- **Equation 8**: Tail decay head (remaining samples, tanh)
- **Equation 9**: Combined structured update with acoustic scaling

#### Value Function (Equation 10)
- Critic network for state value estimation

### 4. RIR Update Mechanism (Section 3.4)

#### Momentum-Based Update (Equations 11-13)
- **Equation 11**: Momentum update rule (β=0.2)
- **Equation 12**: Direct sound constraint (must be dominant)
- **Equation 13**: Temporal envelope constraints
- **Equation 14**: Multi-region envelope function (early/late/tail)

### 5. Reward Function Design (Section 3.5)

This is the core of the learning signal. **Equation 15** defines the total reward as a weighted sum of 6 components:

#### DRR Reward (Equations 16-20)
- **Equation 16**: Wiener deconvolution for dereverberation
- **Equation 17**: Direct energy from strongest frames (first 200ms)
- **Equation 18**: Reverberant energy from later frames
- **Equation 19**: DRR computation in dB
- **Equation 20**: Non-linear DRR reward (exponential for positive, penalty for negative)

#### Structural Rewards (Equations 21-29)
- **Equation 21**: Direct sound strength reward
- **Equations 22-23**: Early reflection suppression (target <10% of direct)
- **Equations 24-25**: Late reverberation suppression (target <5% of total)
- **Equations 26-27**: Tail suppression (target <2% of total)
- **Equations 28-29**: Exponential decay fitting and reward

### 6. Actor-Critic Learning (Section 3.6)

#### Return and Advantage Computation (Equations 30-33)
- **Equation 30**: Discounted returns with γ=0.99
- **Equation 31**: Return normalization for stable learning
- **Equation 32**: Advantage function (how much better than expected)

#### Loss Functions (Equations 33-37)
- **Equation 33**: Advantage function
- **Equation 34**: Actor loss (policy gradient)
- **Equation 35**: Log-probability under Gaussian policy
- **Equation 36**: Critic loss (MSE for value prediction)
- **Equation 37**: Combined total loss
- **Equation 38**: Parameter update with Adam and gradient clipping

### 7. Initialization Strategy (Section 3.7)

- **Equations 39-42**: Exponential decay initialization
  - Strong direct impulse (h₀=1.0)
  - Sparse early reflections (3% probability)
  - Fast exponential decay (3× faster than natural RT60)
  - Normalization to unit amplitude

### 8. Complete Algorithm (Algorithm 1)

The algorithm pseudocode ties everything together:
1. Initialize network and optimizer
2. For each episode:
   - Generate reverberant speech
   - Initialize RIR estimate (Eq. 39-42)
   - Iterative refinement loop (K=15 steps):
     - Forward pass through network
     - Apply RIR update (Eq. 11)
     - Apply constraints (Eq. 12-14)
     - Wiener deconvolution (Eq. 16)
     - Compute reward (Eq. 15)
   - Compute returns and advantages (Eq. 30-32)
   - Compute losses (Eq. 34, 36, 37)
   - Update parameters (Eq. 38)
3. Return trained policy

### 9. Inference (Section 3.8)

Simple 3-step procedure for test time:
1. Initialize with exponential decay
2. Iteratively refine (no exploration noise)
3. Return final estimate

### 10. Complexity Analysis (Section 3.9)

- Per-episode: O(K(Ld + d²)) for network, O(KL log L) for FFT
- Training time: 2-3 minutes for 300 episodes
- Practical for real-world applications

## Equation Reference Flow

The equations are designed to be read in order, with clear dependencies:

**Network Architecture Flow:**
Eq. 1 (problem) → Eq. 2-4 (encoder) → Eq. 5-8 (heads) → Eq. 9 (structured update) → Eq. 10 (value)

**Update Mechanism Flow:**
Eq. 9 (update vector) → Eq. 11 (momentum) → Eq. 12-14 (constraints)

**Reward Computation Flow:**
Eq. 11 (updated RIR) → Eq. 16 (deconvolution) → Eq. 17-19 (DRR) → Eq. 20 (DRR reward) → Eq. 21-29 (structural rewards) → Eq. 15 (total)

**Learning Flow:**
Eq. 15 (rewards) → Eq. 30 (returns) → Eq. 31 (normalization) → Eq. 32 (advantage) → Eq. 34 (actor loss) → Eq. 36 (critic loss) → Eq. 37 (total loss) → Eq. 38 (update)

**Initialization Flow:**
Eq. 39 (direct) → Eq. 40 (reflections) → Eq. 41 (normalization)

## Key Innovations Highlighted

1. **Structured multi-head architecture** (Eq. 5-9): Physically-informed design
2. **Momentum-based updates** (Eq. 11): Stable convergence
3. **Temporal constraints** (Eq. 12-14): Enforces realistic acoustics
4. **Multi-component reward** (Eq. 15): Comprehensive acoustic quality
5. **DRR-based learning** (Eq. 16-20): Direct optimization of perceptual quality
6. **Exponential initialization** (Eq. 39-41): Accelerated convergence

## Publication Quality

The document is ready for submission with:
- Clear mathematical notation
- Systematic equation numbering
- Comprehensive cross-referencing
- Balanced technical depth
- Complete algorithm pseudocode
- Complexity analysis
- Practical implementation details

All hyperparameters are explicitly stated (d=1024, K=15, β=0.2, η=2×10⁻⁴, etc.) and all design choices are justified with clear acoustic or learning-theoretic rationale.
