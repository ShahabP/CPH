import numpy as np
import librosa


def compute_srmr(audio: np.ndarray, sr: int = 16000,
                 n_fft: int = 512, hop_length: int = 256,
                 n_mels: int = 8, low_mod_hz: float = 0.5,
                 high_mod_hz: float = 64.0) -> float:
    """
    Compute a lightweight SRMR approximation using mel-band envelopes' modulation
    spectra. This is a practical proxy of the full SRMR algorithm sufficient for
    test-time reward use and reproducible experiments.

    Returns a scalar roughly proportional to SRMR, scaled into [-1, 1].
    """
    if audio is None or len(audio) == 0:
        return 0.0

    # compute mel spectrogram (power)
    S = librosa.feature.melspectrogram(y=audio.astype(float), sr=sr,
                                       n_fft=n_fft, hop_length=hop_length,
                                       win_length=n_fft, window='hann',
                                       n_mels=n_mels, power=2.0)

    # log-compress envelopes per mel band
    eps = 1e-8
    env = np.log(S + eps)

    # modulation spectrum per band
    n_frames = env.shape[1]
    if n_frames < 4:
        return 0.0

    # FFT of temporal envelopes
    mod_freqs = np.fft.rfftfreq(n_frames, d=hop_length / float(sr))
    mod_spec = np.abs(np.fft.rfft(env, axis=1))  # shape: (n_mels, n_mod_bins)

    # find modulation band indices
    low_idx = np.where(mod_freqs <= 4.0)[0]
    high_idx = np.where((mod_freqs > 4.0) & (mod_freqs <= high_mod_hz))[0]

    if low_idx.size == 0 or high_idx.size == 0:
        return 0.0

    low_energy = np.sum(mod_spec[:, low_idx] ** 2)
    high_energy = np.sum(mod_spec[:, high_idx] ** 2) + eps

    ratio = low_energy / high_energy

    # map ratio to bounded [-1,1] using a smooth transform
    srmr_raw = np.log10(ratio + 1e-12)
    srmr_scaled = np.tanh(srmr_raw / 2.0)

    return float(srmr_scaled)
