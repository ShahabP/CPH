"""
Comprehensive comparison runner for DQN vs Neural RIR methods.

Runs both approaches with streaming segments and generates:
- Side-by-side evolution plots
- Performance metrics table
- Training curves comparison
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import time
import yaml

# Add src to path
THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from neural_rir_agent import compute_rir_drr_metric


def run_method(method: str, output_suffix: str = ""):
    """Run training with specified method."""
    cmd = [
        f"{ROOT_DIR}/.venv/bin/python",
        f"{ROOT_DIR}/train_full.py",
        "--config", f"{ROOT_DIR}/configs/default_config.yaml",
        "--output-dir", f"{ROOT_DIR}/experiments",
        "--stream-dir", f"{ROOT_DIR}/data/stream"
    ]
    
    if method == "neural":
        cmd.append("--neural")
    
    print(f"Running {method} method...")
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    print(f"Method: {method}")
    print(f"Exit code: {result.returncode}")
    print(f"Time: {elapsed:.2f}s")
    if result.stdout:
        print("STDOUT:", result.stdout[-500:])  # Last 500 chars
    if result.stderr:
        print("STDERR:", result.stderr[-300:])  # Last 300 chars
    
    return {
        'method': method,
        'exit_code': result.returncode,
        'time': elapsed,
        'stdout': result.stdout,
        'stderr': result.stderr
    }


def collect_rir_files(experiments_dir: Path, method_suffix: str = ""):
    """Collect RIR files for a method."""
    pattern = f"*_rir{method_suffix}.npy"
    files = list(experiments_dir.glob(pattern))
    
    # Sort by segment and step
    def sort_key(p):
        name = p.stem
        if "_step_" in name:
            parts = name.split("_")
            seg_idx = next((i for i, part in enumerate(parts) if part.isdigit()), 0)
            step_idx = next((i for i, part in enumerate(parts) if part == "step"), -1)
            if seg_idx < len(parts) and step_idx > 0 and step_idx + 1 < len(parts):
                try:
                    return (int(parts[seg_idx]), int(parts[step_idx + 1]))
                except ValueError:
                    pass
        return (0, 0)
    
    return sorted(files, key=sort_key)


def load_rirs(files):
    """Load RIR arrays from files."""
    rirs = []
    for f in files:
        try:
            arr = np.load(f)
            rirs.append(arr)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return rirs


def analyze_rirs(rirs, method_name: str):
    """Analyze RIR quality metrics."""
    if not rirs:
        return {}
    
    drrs = []
    energies = []
    peaks = []
    sparsities = []
    
    for rir in rirs:
        # DRR
        drr = compute_rir_drr_metric(rir)
        drrs.append(drr if not np.isinf(drr) else 0)
        
        # Energy
        energy = np.sum(rir ** 2)
        energies.append(energy)
        
        # Peak amplitude
        peak = np.max(np.abs(rir))
        peaks.append(peak)
        
        # Sparsity (fraction of near-zero elements)
        sparsity = np.sum(np.abs(rir) < 0.01) / len(rir)
        sparsities.append(sparsity)
    
    return {
        'method': method_name,
        'num_estimates': len(rirs),
        'drr_final': drrs[-1] if drrs else 0,
        'drr_mean': np.mean(drrs),
        'drr_std': np.std(drrs),
        'energy_final': energies[-1] if energies else 0,
        'energy_mean': np.mean(energies),
        'peak_final': peaks[-1] if peaks else 0,
        'sparsity_final': sparsities[-1] if sparsities else 0,
        'sparsity_mean': np.mean(sparsities),
        'drr_evolution': drrs,
        'energy_evolution': energies
    }


def create_evolution_comparison(dqn_rirs, neural_rirs, output_path: Path):
    """Create side-by-side evolution comparison plot."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # DQN RIR Evolution
    if dqn_rirs:
        L = min(len(r) for r in dqn_rirs)
        R_dqn = np.stack([r[:L] for r in dqn_rirs], axis=0)
        
        # Overlay plot
        axes[0, 0].set_title("DQN: RIR Evolution")
        x = np.arange(L)
        for i in range(min(len(R_dqn), 10)):
            alpha = max(0.2, 0.8 / (i + 1))
            axes[0, 0].plot(x, R_dqn[i], alpha=alpha, label=f't{i}' if i < 5 else "")
        axes[0, 0].set_xlabel("Tap Index")
        axes[0, 0].set_ylabel("Amplitude")
        if len(R_dqn) <= 5:
            axes[0, 0].legend()
        
        # Heatmap
        axes[0, 1].set_title("DQN: Evolution Heatmap")
        vmin, vmax = np.percentile(R_dqn, [5, 95])
        im1 = axes[0, 1].imshow(R_dqn, aspect='auto', origin='lower', 
                               cmap='viridis', vmin=vmin, vmax=vmax,
                               extent=(0, L-1, 0, len(R_dqn)-1))
        axes[0, 1].set_xlabel("Tap Index")
        axes[0, 1].set_ylabel("Iteration")
        plt.colorbar(im1, ax=axes[0, 1])
    else:
        axes[0, 0].text(0.5, 0.5, "No DQN RIRs", transform=axes[0, 0].transAxes, ha='center')
        axes[0, 1].text(0.5, 0.5, "No DQN RIRs", transform=axes[0, 1].transAxes, ha='center')
    
    # Neural RIR Evolution
    if neural_rirs:
        L = min(len(r) for r in neural_rirs)
        R_neural = np.stack([r[:L] for r in neural_rirs], axis=0)
        
        # Overlay plot
        axes[1, 0].set_title("Neural: RIR Evolution")
        x = np.arange(L)
        for i in range(min(len(R_neural), 10)):
            alpha = max(0.2, 0.8 / (i + 1))
            axes[1, 0].plot(x, R_neural[i], alpha=alpha, label=f't{i}' if i < 5 else "")
        axes[1, 0].set_xlabel("Tap Index")
        axes[1, 0].set_ylabel("Amplitude")
        if len(R_neural) <= 5:
            axes[1, 0].legend()
        
        # Heatmap
        axes[1, 1].set_title("Neural: Evolution Heatmap")
        vmin, vmax = np.percentile(R_neural, [5, 95])
        im2 = axes[1, 1].imshow(R_neural, aspect='auto', origin='lower', 
                               cmap='plasma', vmin=vmin, vmax=vmax,
                               extent=(0, L-1, 0, len(R_neural)-1))
        axes[1, 1].set_xlabel("Tap Index")
        axes[1, 1].set_ylabel("Iteration")
        plt.colorbar(im2, ax=axes[1, 1])
    else:
        axes[1, 0].text(0.5, 0.5, "No Neural RIRs", transform=axes[1, 0].transAxes, ha='center')
        axes[1, 1].text(0.5, 0.5, "No Neural RIRs", transform=axes[1, 1].transAxes, ha='center')
    
    # Metrics comparison
    axes[0, 2].set_title("DRR Evolution Comparison")
    if dqn_rirs:
        dqn_drrs = [compute_rir_drr_metric(r) for r in dqn_rirs]
        dqn_drrs = [d if not np.isinf(d) else 0 for d in dqn_drrs]
        axes[0, 2].plot(dqn_drrs, 'b-', label='DQN', alpha=0.7)
    
    if neural_rirs:
        neural_drrs = [compute_rir_drr_metric(r) for r in neural_rirs]
        neural_drrs = [d if not np.isinf(d) else 0 for d in neural_drrs]
        axes[0, 2].plot(neural_drrs, 'r-', label='Neural', alpha=0.7)
    
    axes[0, 2].set_xlabel("Iteration")
    axes[0, 2].set_ylabel("DRR (dB)")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Final RIR comparison
    axes[1, 2].set_title("Final RIR Comparison")
    if dqn_rirs and neural_rirs:
        final_dqn = dqn_rirs[-1]
        final_neural = neural_rirs[-1]
        L_comp = min(len(final_dqn), len(final_neural))
        x = np.arange(L_comp)
        axes[1, 2].plot(x, final_dqn[:L_comp], 'b-', label='DQN Final', alpha=0.7)
        axes[1, 2].plot(x, final_neural[:L_comp], 'r-', label='Neural Final', alpha=0.7)
        axes[1, 2].legend()
    elif dqn_rirs:
        axes[1, 2].plot(dqn_rirs[-1], 'b-', label='DQN Final')
        axes[1, 2].legend()
    elif neural_rirs:
        axes[1, 2].plot(neural_rirs[-1], 'r-', label='Neural Final')
        axes[1, 2].legend()
    
    axes[1, 2].set_xlabel("Tap Index")
    axes[1, 2].set_ylabel("Amplitude")
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved evolution comparison: {output_path}")


def create_metrics_table(dqn_metrics, neural_metrics, run_results):
    """Create comparison table as text and dict."""
    
    table_data = {
        'Method': ['DQN', 'Neural'],
        'Training Time (s)': [
            run_results[0]['time'] if run_results[0]['exit_code'] == 0 else 'FAILED',
            run_results[1]['time'] if run_results[1]['exit_code'] == 0 else 'FAILED'
        ],
        'RIR Estimates': [
            dqn_metrics.get('num_estimates', 0),
            neural_metrics.get('num_estimates', 0)
        ],
        'Final DRR (dB)': [
            f"{dqn_metrics.get('drr_final', 0):.2f}",
            f"{neural_metrics.get('drr_final', 0):.2f}"
        ],
        'Mean DRR (dB)': [
            f"{dqn_metrics.get('drr_mean', 0):.2f}",
            f"{neural_metrics.get('drr_mean', 0):.2f}"
        ],
        'Final Energy': [
            f"{dqn_metrics.get('energy_final', 0):.3f}",
            f"{neural_metrics.get('energy_final', 0):.3f}"
        ],
        'Final Sparsity': [
            f"{dqn_metrics.get('sparsity_final', 0):.3f}",
            f"{neural_metrics.get('sparsity_final', 0):.3f}"
        ]
    }
    
    # Format as table text
    table_text = "| Metric | DQN | Neural |\n"
    table_text += "|--------|-----|--------|\n"
    
    for metric in ['Training Time (s)', 'RIR Estimates', 'Final DRR (dB)', 
                   'Mean DRR (dB)', 'Final Energy', 'Final Sparsity']:
        dqn_val = table_data[metric][0]
        neural_val = table_data[metric][1]
        table_text += f"| {metric} | {dqn_val} | {neural_val} |\n"
    
    return table_data, table_text


def main():
    """Run comprehensive comparison."""
    print("=== DQN vs Neural RIR Methods Comparison ===\n")
    
    root_dir = Path(ROOT_DIR)
    experiments_dir = root_dir / 'experiments'
    comparison_dir = experiments_dir / 'method_comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean old RIR files
    old_files = list(experiments_dir.glob("*_rir*.npy"))
    for f in old_files:
        f.unlink()
    
    # Run both methods
    run_results = []
    
    # DQN method
    dqn_result = run_method("dqn")
    run_results.append(dqn_result)
    
    # Neural method  
    neural_result = run_method("neural")
    run_results.append(neural_result)
    
    # Collect RIR files
    dqn_files = collect_rir_files(experiments_dir, "")
    neural_files = collect_rir_files(experiments_dir, "_neural")
    
    print(f"\nFound {len(dqn_files)} DQN RIR files")
    print(f"Found {len(neural_files)} Neural RIR files")
    
    # Load RIRs
    dqn_rirs = load_rirs(dqn_files)
    neural_rirs = load_rirs(neural_files)
    
    # Analyze metrics
    dqn_metrics = analyze_rirs(dqn_rirs, "DQN")
    neural_metrics = analyze_rirs(neural_rirs, "Neural")
    
    # Create comparison visualization
    evolution_plot = comparison_dir / 'rir_evolution_comparison.png'
    create_evolution_comparison(dqn_rirs, neural_rirs, evolution_plot)
    
    # Create metrics table
    table_data, table_text = create_metrics_table(dqn_metrics, neural_metrics, run_results)
    
    # Save results
    results_file = comparison_dir / 'comparison_results.txt'
    with open(results_file, 'w') as f:
        f.write("=== DQN vs Neural RIR Methods Comparison ===\n\n")
        f.write("## Performance Metrics\n\n")
        f.write(table_text)
        f.write(f"\n## Detailed Metrics\n\n")
        f.write(f"DQN Metrics:\n{dqn_metrics}\n\n")
        f.write(f"Neural Metrics:\n{neural_metrics}\n\n")
    
    # Print summary
    print(f"\n=== Results Summary ===")
    print(table_text)
    
    print(f"\nFiles generated:")
    print(f"  - Evolution plot: {evolution_plot}")
    print(f"  - Results summary: {results_file}")
    
    return {
        'table_data': table_data,
        'table_text': table_text,
        'dqn_metrics': dqn_metrics,
        'neural_metrics': neural_metrics,
        'run_results': run_results
    }


if __name__ == "__main__":
    results = main()