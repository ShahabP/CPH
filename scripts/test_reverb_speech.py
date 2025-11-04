import os
import sys
import numpy as np
import soundfile as sf
from pathlib import Path

# Make src importable
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from audio_processing import AudioProcessor, convolve_with_rir, estimate_rt60  # type: ignore
from dereverberation import BlindDereverberation  # type: ignore
from rir_estimation import DeconvolutionRIREstimator  # type: ignore

SR = 16000
AP = AudioProcessor(sample_rate=SR)


def synth_vowel_segment(vowel: str, duration: float = 0.6) -> np.ndarray:
    # Approximate formant frequencies for some vowels (Hz)
    formants = {
        'a': [800, 1150, 2900],
        'e': [500, 1900, 2500],
        'i': [350, 2200, 3000],
        'o': [450, 800, 2830],
        'u': [325, 700, 2530],
    }
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    env = 0.5 * (1 - np.cos(2 * np.pi * np.clip(t / duration, 0, 1)))  # raised-cosine envelope
    amps = [0.8, 0.5, 0.3]
    sig = np.zeros_like(t)
    for a, f in zip(amps, formants.get(vowel, formants['a'])):
        sig += a * np.sin(2 * np.pi * f * t)
    # Add a gentle F0 amplitude modulation to mimic voicing
    f0 = 140.0
    sig *= (0.75 + 0.25 * np.sin(2 * np.pi * f0 * t))
    return 0.2 * env * sig


def synth_clean_speech() -> np.ndarray:
    seq = ['a', 'e', 'i', 'o', 'u']
    segs = [synth_vowel_segment(v, duration=0.5) for v in seq]
    pause = np.zeros(int(0.05 * SR))
    clean = np.concatenate([np.concatenate([s, pause]) for s in segs])
    return clean.astype(np.float32)


def synth_rir(rt60: float = 0.4, length_s: float = 0.6) -> np.ndarray:
    """Simple exponentially decaying RIR (minimum phase-like) for stable deconvolution."""
    n = int(length_s * SR)
    # Per-sample decay to reach -60 dB at RT60
    a = 10 ** (-3.0 / (rt60 * SR))
    idx = np.arange(n)
    rir = (a ** idx).astype(np.float32)
    # Ensure unit peak at t=0
    rir /= np.max(np.abs(rir)) + 1e-9
    return rir


def main(out_dir='experiments/sample_test'):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Generate clean speech-like signal
    clean = synth_clean_speech()
    sf.write(str(out / 'clean.wav'), clean, SR)

    # 2) Generate synthetic RIR and reverberate
    target_rt60 = 0.45
    rir = synth_rir(rt60=target_rt60)
    sf.write(str(out / 'rir.wav'), rir, SR)
    reverb = convolve_with_rir(clean, rir)
    reverb = np.clip(reverb, -1.0, 1.0).astype(np.float32)
    sf.write(str(out / 'reverberant.wav'), reverb, SR)

    # 3) Blind dereverberation (classical)
    derev = BlindDereverberation(method='wiener', sample_rate=SR)
    enhanced = derev.dereverberate(reverb)
    enhanced = enhanced.astype(np.float32)
    sf.write(str(out / 'dereverb.wav'), enhanced, SR)

    # 4) RIR estimation via Wiener deconvolution
    est = DeconvolutionRIREstimator(method='wiener_deconv', regularization=1e-2)
    # Blind-ish: use dereverberated signal as a proxy for clean
    est_rir_blind = est.estimate_rir(reverb, enhanced, rir_length=min(len(rir), 2048))
    # Oracle upper bound: use ground-truth clean
    est_rir_oracle = est.estimate_rir(reverb, clean, rir_length=min(len(rir), 2048))

    # 5) Metrics
    Lb = min(len(est_rir_blind), len(rir))
    Lo = min(len(est_rir_oracle), len(rir))
    corr_blind = float(np.corrcoef(est_rir_blind[:Lb], rir[:Lb])[0, 1])
    nmse_blind = float(np.mean((est_rir_blind[:Lb] - rir[:Lb])**2) / (np.mean(rir[:Lb]**2) + 1e-12))
    corr_oracle = float(np.corrcoef(est_rir_oracle[:Lo], rir[:Lo])[0, 1])
    nmse_oracle = float(np.mean((est_rir_oracle[:Lo] - rir[:Lo])**2) / (np.mean(rir[:Lo]**2) + 1e-12))
    rt60_true = estimate_rt60(rir, SR)
    rt60_est_blind = estimate_rt60(est_rir_blind[:Lb], SR)
    rt60_est_oracle = estimate_rt60(est_rir_oracle[:Lo], SR)

    # Save estimated RIR
    np.save(out / 'estimated_rir_blind.npy', est_rir_blind.astype(np.float32))
    np.save(out / 'estimated_rir_oracle.npy', est_rir_oracle.astype(np.float32))

    # Optional: LMS oracle estimation for robustness
    est_lms = DeconvolutionRIREstimator(method='lms')
    est_rir_lms, lms_errors = est_lms.estimate_rir_lms(reverb, clean, rir_length=min(len(rir), 2048), step_size=0.01, num_iterations=200)
    Ll = min(len(est_rir_lms), len(rir))
    corr_lms = float(np.corrcoef(est_rir_lms[:Ll], rir[:Ll])[0, 1])
    nmse_lms = float(np.mean((est_rir_lms[:Ll] - rir[:Ll])**2) / (np.mean(rir[:Ll]**2) + 1e-12))
    rt60_est_lms = estimate_rt60(est_rir_lms[:Ll], SR)
    np.save(out / 'estimated_rir_lms_oracle.npy', est_rir_lms.astype(np.float32))

    print('=== Sample Reverberant Speech Test ===')
    print(f'Clean length: {len(clean)} samples, Reverb length: {len(reverb)}')
    print(f'RIR length: {len(rir)}')
    print(f'— Blind estimate —')
    print(f'  Est RIR length: {len(est_rir_blind)}  | Corr: {corr_blind:.3f} | NMSE: {nmse_blind:.4f} | RT60: {rt60_est_blind:.3f}s')
    print(f'— Oracle (using clean) —')
    print(f'  Est RIR length: {len(est_rir_oracle)} | Corr: {corr_oracle:.3f} | NMSE: {nmse_oracle:.4f} | RT60: {rt60_est_oracle:.3f}s')
    print(f'— Oracle LMS —')
    print(f'  Est RIR length: {len(est_rir_lms)}  | Corr: {corr_lms:.3f} | NMSE: {nmse_lms:.4f} | RT60: {rt60_est_lms:.3f}s')
    print(f'(Target RT60: {target_rt60:.3f}s, Measured true RIR RT60: {rt60_true:.3f}s)')
    print(f'Wrote WAVs and RIR to: {out.resolve()}')


if __name__ == '__main__':
    main()
