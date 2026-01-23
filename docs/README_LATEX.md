# LaTeX Algorithm Documentation

## Files

### Main Documents

1. **`algorithm_section.tex`** - Complete algorithm section with equations, tables, and pseudocode
2. **`algorithm_main.tex`** - Standalone compilable document (includes algorithm_section.tex)
3. **`algorithm_section_guide.md`** - Guide explaining the structure and equation flow

## Tables

### Table 1: Network Architecture Details (`Table~\ref{tab:network_architecture}`)

Provides a comprehensive breakdown of the neural policy network including:
- **Input normalization layer**
- **Encoder**: 3-layer residual architecture with LayerNorm and dropout
- **Actor heads**: 4 specialized heads for different RIR temporal regions
  - Direct sound head (1 sample, sigmoid)
  - Early reflections head (63 samples, tanh)
  - Late reverberation head (192 samples, tanh)
  - Tail decay head (L-256 samples, tanh)
- **Critic head**: Value function network
- **Total parameters**: ~3.2M for L=2048, d=1024

**Columns:**
- Component: Major network section
- Layer: Specific layer name
- Input Dim: Input dimensionality
- Output Dim: Output dimensionality
- Operations: Mathematical operations and weights

### Table 2: Hyperparameters and Coefficients (`Table~\ref{tab:hyperparameters}`)

Comprehensive listing of all parameters organized by category:

1. **RIR Configuration** (4 parameters)
   - RIR lengths: 256, 512, 1024, 2048 samples
   - Sampling rate: 16 kHz
   - RT60 range: 100-1200 ms

2. **Network Architecture** (5 parameters)
   - Hidden dimension: 1024
   - Dropout rate: 0.1
   - Latent dimension: 512
   - Total parameters: ~3.2M

3. **Acoustic Scaling Factors** (6 parameters)
   - Direct sound: 0.8
   - Early reflections: 0.15
   - Late reverberation: 0.08
   - Tail decay: 0.03
   - Momentum: 0.2
   - Wiener regularization: 0.01

4. **Envelope Constraints** (9 parameters)
   - Thresholds for direct, early, late, tail regions
   - Decay constants
   - Time boundaries

5. **Reward Weights** (12 parameters)
   - Component weights (DRR: 5.0, Direct: 2.0, etc.)
   - Target ratios for each temporal region
   - Normalization scales

6. **DRR Computation** (4 parameters)
   - Frame size: 400 samples (25 ms)
   - Hop size: 160 samples (10 ms)
   - Direct frames: 20 (200 ms)
   - Numerical epsilon: 1e-12

7. **Training Configuration** (9 parameters)
   - Episodes: 300-1200
   - Iterations per episode: 15
   - Learning rate: 2e-4
   - Discount factor: 0.99
   - Gradient clipping: 0.5
   - LR scheduling parameters

8. **Initialization** (4 parameters)
   - Direct impulse: 1.0
   - Early reflection probability: 3%
   - Amplitude scales
   - RT60 acceleration factor: 1/3

**Columns:**
- Category: Parameter grouping
- Parameter: Variable name/symbol
- Description: Explanation
- Value: Numerical value or range

## Compilation

To compile the standalone document:

```bash
cd docs/
pdflatex algorithm_main.tex
pdflatex algorithm_main.tex  # Run twice for references
```

Or to include in your main paper:

```latex
\input{algorithm_section.tex}
```

## Required LaTeX Packages

```latex
\usepackage{amsmath,amssymb,amsfonts}  % Math symbols
\usepackage{algorithmic}                % Algorithm environment
\usepackage{algorithm}                  % Algorithm floating environment
\usepackage{booktabs}                   % Professional tables
\usepackage{multirow}                   % Multi-row table cells
```

## Table Placement

Both tables are positioned with `[htbp]` placement specifier:
- `h`: Here (preferred)
- `t`: Top of page
- `b`: Bottom of page
- `p`: Separate page

LaTeX will choose the best placement automatically.

## Referencing in Text

```latex
% Reference Table 1
As shown in Table~\ref{tab:network_architecture}, the encoder...

% Reference Table 2
All hyperparameters are listed in Table~\ref{tab:hyperparameters}.
```

## Cross-References with Equations

The tables work seamlessly with the 42 numbered equations in the document:

- Network architecture (Table 1) ↔ Equations 2-10
- Hyperparameters (Table 2) ↔ All equations (1-42)
- Algorithm pseudocode ↔ Both tables

## Notes

- Tables use `\small` font for better fit
- `@{}` in tabular removes side padding for compact layout
- `\toprule`, `\midrule`, `\bottomrule` from booktabs for professional appearance
- `\cmidrule(lr)` for partial rules with trimming
- Multi-row cells group related parameters
- Values aligned right, text aligned left for readability

## Total Document Statistics

- **Sections**: 10
- **Equations**: 42 (all numbered and cross-referenced)
- **Tables**: 2 (detailed architecture and parameters)
- **Algorithm**: 1 (25-step pseudocode)
- **References**: 100+ cross-references
- **Pages**: Approximately 12-15 (single column)
