import os
import sys
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt


def find_rir_files(root: Path):
    # Find files like segment_XX_rir.npy under root
    files = list(root.glob("*_rir.npy"))
    # Try to sort by embedded index if present
    def key_fn(p: Path):
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else -1
    files_sorted = sorted(files, key=lambda p: (key_fn(p), p.stat().st_mtime))
    return files_sorted


def load_rirs(files):
    rirs = []
    for f in files:
        try:
            arr = np.load(f)
            rirs.append(arr.astype(np.float32))
        except Exception as e:
            print(f"Skip {f}: {e}")
    return rirs


def plot_evolution(rirs, out_path: Path, max_plot: int = 12):
    if not rirs:
        print("No RIR files found to plot.")
        return

    # Align lengths
    L = min(len(r) for r in rirs)
    R = np.stack([r[:L] for r in rirs], axis=0)  # [T, L]
    T = R.shape[0]

    # Running average envelope
    env = np.cumsum(np.abs(R), axis=0) / (np.arange(len(R))[:, None] + 1)

    # Figure
    plt.figure(figsize=(12, 8))

    # 1) Overlay of first N RIRs
    plt.subplot(2, 2, 1)
    n = min(len(R), max_plot)
    x = np.arange(L)
    for i in range(n):
        plt.plot(x, R[i], alpha=max(0.15, 0.8 / (i + 1)), label=f"t{i}")
    plt.title("Global RIR evolution (overlay)")
    plt.xlabel("Tap index")
    plt.ylabel("Amplitude")
    if n <= 10:
        plt.legend(loc='upper right', fontsize=8)

    # 2) Heatmap over time
    plt.subplot(2, 2, 2)
    vmin, vmax = np.percentile(R, [1, 99])
    # Set extent to align pixel centers to integer indices (0..T-1, 0..L-1)
    extent = (0, L-1, 0, T-1)
    plt.imshow(R, aspect='auto', origin='lower', cmap='magma',
               vmin=vmin, vmax=vmax, extent=extent, interpolation='nearest')
    plt.colorbar(label='Amplitude')
    plt.title("RIR over time (heatmap)")
    plt.ylabel("Iteration/segment")
    plt.xlabel("Tap index")
    # Optional tidy integer ticks on y if small enough
    if T <= 30:
        import numpy as _np
        plt.yticks(_np.arange(T))

    # 3) Running average envelope
    plt.subplot(2, 1, 2)
    plt.plot(x, env[-1], label='Final running |avg|', color='C1')
    plt.plot(x, np.abs(R[-1]), label='|Final RIR|', color='C0', alpha=0.7)
    plt.title("Final |RIR| vs running average envelope")
    plt.xlabel("Tap index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot: {out_path}")


def main():
    # Default to experiments directory
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('experiments')
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else root / 'rir_evolution.png'

    # Search direct children first, then recurse one level for convenience
    files = find_rir_files(root)
    if not files:
        files = []
        for sub in root.glob('*'):
            if sub.is_dir():
                files.extend(find_rir_files(sub))
    files = sorted(files)

    print(f"Found {len(files)} RIR snapshots under {root}")
    rirs = load_rirs(files)
    plot_evolution(rirs, out)


if __name__ == '__main__':
    main()
