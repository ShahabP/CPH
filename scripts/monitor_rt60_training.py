#!/usr/bin/env python3
"""
Monitor the RT60 sweep training progress.
"""

import json
from pathlib import Path
import time

def monitor_progress():
    """Monitor training progress."""
    output_dir = Path('experiments/neural_rt60_sweep')
    
    rt60_values = [100, 200, 300, 400, 500, 600, 700, 800]
    
    print("\n" + "="*70)
    print("MONITORING RT60 SWEEP TRAINING PROGRESS")
    print("="*70)
    
    completed = []
    for rt60 in rt60_values:
        result_path = output_dir / f'rt60_{rt60}ms.pkl'
        if result_path.exists():
            completed.append(rt60)
    
    print(f"\nCompleted: {len(completed)}/{len(rt60_values)} RT60 values")
    print(f"Progress: {len(completed)/len(rt60_values)*100:.1f}%")
    
    if completed:
        print(f"\nCompleted RT60 values: {', '.join(str(x) for x in completed)} ms")
    
    if len(completed) < len(rt60_values):
        remaining = [x for x in rt60_values if x not in completed]
        print(f"Remaining: {', '.join(str(x) for x in remaining)} ms")
        
        # Estimate time remaining (approx 3-5 min per RT60 value)
        avg_time_per_rt60 = 4  # minutes
        time_remaining = (len(rt60_values) - len(completed)) * avg_time_per_rt60
        print(f"\nEstimated time remaining: ~{time_remaining} minutes")
    else:
        print("\n✓ All training complete!")
        
        # Check if summary exists
        summary_path = output_dir / 'summary.json'
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            print("\n" + "="*70)
            print("QUICK RESULTS SUMMARY")
            print("="*70)
            
            for rt60, drr in zip(summary['rt60_values_ms'], summary['drr_values_db']):
                success = '✓' if drr >= 7 else ' '
                print(f"RT60 = {rt60:3d} ms: DRR = {drr:6.2f} dB  {success}")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    monitor_progress()
