# All Combinations Results

This directory contains comprehensive training results and visualizations for all 6 combinations of RL agents and initialization methods.

## Files

### Training Results
- **`all_results.pkl`**: Complete training statistics for all 6 combinations (Python pickle format)
- **`summary.json`**: Summary metrics in JSON format

### Visualizations

1. **`drr_comparison_all_methods.png`**: Learning curves showing reward/DRR over episodes for all 6 methods
2. **`final_rirs_all_methods.png`**: Final RIR waveforms for all 6 methods with correlation and DRR metrics
3. **`summary_table.png`**: Performance comparison table
4. **`all_combinations_comparison.png`**: Comprehensive 6-panel comparison (rewards, correlation, steps, etc.)

## Combinations Tested

1. Q-Learning (QN) + Random Initialization
2. Q-Learning (QN) + Exponential Decay Initialization
3. Deep Q-Network (DQN) + Random Initialization
4. Deep Q-Network (DQN) + Exponential Decay Initialization
5. Neural RIR Agent + Random Initialization
6. Neural RIR Agent + Exponential Decay Initialization ⭐ **BEST**

## Key Results

| Agent | Initialization | Avg Reward | Avg Correlation | Avg Steps |
|-------|---------------|------------|-----------------|-----------|
| QN | Random | 6.38 | 0.118 | 3.0 |
| QN | Exp Decay | 6.38 | 0.118 | 3.0 |
| DQN | Random | 6.38 | 0.118 | 3.0 |
| DQN | Exp Decay | 6.38 | 0.118 | 3.0 |
| Neural | Random | 38.49 | 0.131 | 15.0 |
| **Neural** | **Exp Decay** | **61.40** | **0.678** | **15.0** |

## Reproducing Results

To reproduce these results, run:

```bash
python scripts/train_all_combinations.py
```

Training parameters:
- QN: 300 episodes
- DQN: 300 episodes  
- Neural: 200 episodes
- Max steps per episode: 15
