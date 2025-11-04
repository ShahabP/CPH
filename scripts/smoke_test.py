import os
import sys
import numpy as np
from scipy import signal

# Add src to path for local development
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from audio_processing import AudioProcessor, convolve_with_rir, estimate_rt60
from rir_estimation import DeconvolutionRIREstimator


def main():
    # Generate synthetic clean signal
    sr = 16000
    duration = 1.5
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    clean = 0.2 * (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t))

    # Synthetic simple RIR
    rir_len = 512
    rir = np.zeros(rir_len)
    rir[0] = 1.0
    if rir_len > 120:
        rir[120] = 0.25
    if rir_len > 240:
        rir[240] = 0.12

    # Make reverberant
    reverb = convolve_with_rir(clean, rir)

    # Feature extraction
    ap = AudioProcessor(sample_rate=sr)
    feats = ap.extract_features(reverb)
    print("Features extracted:", {k: v.shape if hasattr(v, 'shape') else len(v) for k, v in feats.items()})

    # Estimate RT60 from ground-truth RIR
    rt60 = estimate_rt60(rir, sr)
    print(f"Estimated RT60 (from true RIR): {rt60:.3f}s")

    # RIR estimation via Wiener deconvolution
    estimator = DeconvolutionRIREstimator(method='wiener_deconv')
    est_rir = estimator.estimate_rir(reverb, clean, rir_length=rir_len)
    print("Estimated RIR length:", len(est_rir))

    # Basic sanity: correlation with true RIR
    min_len = min(len(est_rir), len(rir))
    corr = np.corrcoef(est_rir[:min_len], rir[:min_len])[0, 1]
    print(f"Correlation with true RIR (rough): {corr:.3f}")


if __name__ == "__main__":
    main()
