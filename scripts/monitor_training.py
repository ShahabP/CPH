#!/usr/bin/env python3
"""
Monitor enhanced training progress.

Checks completed runs and estimates remaining time.
"""

import json
from pathlib import Path
from datetime import datetime


def check_training_progress():
    """Check which training runs have completed."""
    
    base_dir = Path("experiments/rir_length_enhanced")
    
    rir_lengths = [256, 512, 1024, 2048]
    agent_types = ['qn', 'dqn', 'neural']
    init_methods = ['random', 'exponential_decay']
    
    total_runs = len(rir_lengths) * len(agent_types) * len(init_methods)
    completed = 0
    
    print("\n" + "=" * 80)
    print("ENHANCED TRAINING PROGRESS MONITOR")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for rir_len in rir_lengths:
        print(f"\nRIR Length: {rir_len} samples ({rir_len/16:.1f} ms)")
        
        for agent in agent_types:
            for init in init_methods:
                key = f"{agent}_{init}"
                model_dir = base_dir / f"rir_{rir_len}" / key
                
                # Check for completion markers
                is_complete = False
                
                if agent == 'qn':
                    is_complete = (model_dir / "q_table.pkl").exists()
                elif agent == 'dqn':
                    is_complete = (model_dir / "model.pt").exists()
                elif agent == 'neural':
                    is_complete = (model_dir / "best_model.pt").exists()
                
                status = "✓ DONE" if is_complete else "⏳ Running..."
                print(f"  {key:25s} {status}")
                
                if is_complete:
                    completed += 1
    
    print("\n" + "=" * 80)
    print(f"Progress: {completed}/{total_runs} runs completed ({100*completed/total_runs:.1f}%)")
    
    if completed < total_runs:
        remaining = total_runs - completed
        # Estimate: ~15 min per run average
        est_minutes = remaining * 15
        est_hours = est_minutes / 60
        print(f"Estimated remaining time: {est_hours:.1f} hours")
    else:
        print("\n🎉 ALL TRAINING COMPLETE! Ready to generate plots.")
        print(f"\nRun: python3 scripts/plot_enhanced_drr.py")
    
    print("=" * 80 + "\n")
    
    return completed, total_runs


if __name__ == '__main__':
    check_training_progress()
