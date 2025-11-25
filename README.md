# Reinforcement Learning for Room Impulse Response Estimation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reinforcement learning system for extracting room impulse responses (RIRs) from reverberant speech through iterative blind dereverberation.

## 🎯 Key Achievement

**Neural-Exponential_Decay method achieves 24.6-27.4 dB DRR** across all RIR lengths, significantly exceeding the 7 dB target by 3.5-4x.

## 📊 Results Overview

| Method | Avg DRR | Best DRR | Success Rate (≥7 dB) |
|--------|---------|----------|---------------------|
| **Neural-Exp_Decay** | **26.39 dB** | **27.41 dB** | **100% (4/4)** ✓ |
| Neural-Random | 1.64 dB | 2.85 dB | 0% |
| DQN-Enhanced | -3.47 dB | 6.14 dB | 0% |
| QN-Enhanced | -3.30 dB | 6.65 dB | 0% |

**Overall Performance:**
- Average DRR improvement: 2.42 dB
- Best single result: 27.41 dB (Neural-exp_decay @ 16ms RIR)
- Success rate: 16.7% (4/24 method-length combinations ≥7 dB)

See [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) for detailed analysis.

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/ShahabP/Copenhagen.git
cd Copenhagen

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### Run Training

```bash
# Train all enhanced methods (6-8 hours)
python scripts/train_enhanced.py

# Monitor training progress
python scripts/monitor_training.py

# Generate result plots
python scripts/plot_enhanced_results.py
```

## 📁 Project Structure

```
Copenhagen/
├── src/
│   ├── audio_processing/         # Audio feature extraction, DRR computation
│   ├── rl_framework/             # RL environment, QN/DQN agents
│   └── neural_rir_agent.py       # Neural policy agent (BEST)
├── scripts/
│   ├── train_enhanced.py         # Main training script
│   ├── plot_enhanced_results.py  # Generate plots
│   └── monitor_training.py       # Track progress
├── experiments/
│   ├── rir_length_enhanced/      # Enhanced training results
│   └── *.png                     # Performance plots
└── RESULTS_SUMMARY.md            # Detailed analysis
```

## 🔬 Key Findings

### ✅ What Works

1. **Neural-Exponential_Decay:** Consistently achieves 24-27 dB
2. **Exponential initialization:** Provides crucial physical prior
3. **Enhanced capacity:** 1024 hidden dim enables better learning

### ❌ What Doesn't Work

1. **Tabular Q-learning (QN):** Unsuitable for continuous RIR outputs
2. **Deep Q-Networks (DQN):** Struggle despite deeper architecture
3. **Random initialization:** Poor results even with neural agent

## 📝 License

This project is licensed under the MIT License.

---

**Last Updated:** November 26, 2025  
**Best Result:** 27.41 dB DRR (Neural-Exponential_Decay @ 16ms RIR)
