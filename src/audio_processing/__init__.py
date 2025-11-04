"""
Audio Processing Utilities for Reverberant Speech Analysis

This module provides core audio processing functions for loading, preprocessing,
and feature extraction from reverberant speech signals.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy import signal

# Optional heavy deps (used only when available)
try:
    import torch  # type: ignore
    import torchaudio  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    torchaudio = None  # type: ignore


class AudioProcessor:
    """Core audio processing functionality for reverberant speech."""
    
    def __init__(self, sample_rate: int = 16000, n_fft: int = 1024, 
                 hop_length: int = 256, win_length: Optional[int] = None):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sampling rate
            n_fft: FFT size for spectrograms
            hop_length: Hop length for STFT
            win_length: Window length (defaults to n_fft)
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        
    def load_audio(self, file_path: str, 
                   target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio file and optionally resample.
        
        Args:
            file_path: Path to audio file
            target_sr: Target sampling rate (uses self.sample_rate if None)
            
        Returns:
            Tuple of (audio_data, sampling_rate)
        """
        target_sr = target_sr or self.sample_rate
        audio, sr = librosa.load(file_path, sr=target_sr)
        return audio, sr
    
    def compute_stft(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute Short-Time Fourier Transform.
        
        Args:
            audio: Audio signal
            
        Returns:
            Complex STFT coefficients
        """
        return librosa.stft(audio, n_fft=self.n_fft, 
                           hop_length=self.hop_length, 
                           win_length=self.win_length)
    
    def compute_magnitude_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute magnitude spectrogram."""
        stft = self.compute_stft(audio)
        return np.abs(stft)
    
    def compute_log_mel_spectrogram(self, audio: np.ndarray, 
                                   n_mels: int = 80) -> np.ndarray:
        """
        Compute log-mel spectrogram.
        
        Args:
            audio: Audio signal
            n_mels: Number of mel frequency bins
            
        Returns:
            Log-mel spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=n_mels
        )
        return librosa.power_to_db(mel_spec)
    
    def extract_features(self, audio: np.ndarray) -> dict:
        """
        Extract comprehensive audio features.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary containing various audio features
        """
        features = {}
        
        # Spectral features
        features['magnitude_spec'] = self.compute_magnitude_spectrogram(audio)
        features['log_mel_spec'] = self.compute_log_mel_spectrogram(audio)
        
        # Time-domain features
        features['rms_energy'] = librosa.feature.rms(y=audio, 
                                                    hop_length=self.hop_length)[0]
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
            audio, hop_length=self.hop_length)[0]
        
        # Frequency-domain features
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0]
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0]
        
        return features
    
    def add_noise(self, audio: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Add white noise to audio signal.
        
        Args:
            audio: Clean audio signal
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Noisy audio signal
        """
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
        return audio + noise
    
    def normalize_audio(self, audio: np.ndarray, 
                       target_db: float = -20.0) -> np.ndarray:
        """
        Normalize audio to target RMS level.
        
        Args:
            audio: Input audio
            target_db: Target RMS level in dB
            
        Returns:
            Normalized audio
        """
        rms = np.sqrt(np.mean(audio ** 2))
        target_rms = 10 ** (target_db / 20)
        return audio * (target_rms / rms)
    
    def plot_spectrogram(self, audio: np.ndarray, title: str = "Spectrogram"):
        """Plot magnitude spectrogram."""
        mag_spec = self.compute_magnitude_spectrogram(audio)
        # Import here to avoid hard dependency at module import time
        import librosa.display  # type: ignore
        librosa.display.specshow(
            librosa.amplitude_to_db(mag_spec),
            sr=self.sample_rate, hop_length=self.hop_length,
            x_axis='time', y_axis='hz'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()


def convolve_with_rir(clean_audio: np.ndarray, 
                     rir: np.ndarray) -> np.ndarray:
    """
    Convolve clean audio with room impulse response.
    
    Args:
        clean_audio: Clean speech signal
        rir: Room impulse response
        
    Returns:
        Reverberant audio
    """
    return signal.fftconvolve(clean_audio, rir, mode='full')[:len(clean_audio)]


def estimate_rt60(rir: np.ndarray, sample_rate: int) -> float:
    """
    Estimate RT60 reverberation time from room impulse response.
    
    Args:
        rir: Room impulse response
        sample_rate: Sampling rate
        
    Returns:
        RT60 in seconds
    """
    # Convert to dB scale
    rir_db = 20 * np.log10(np.abs(rir) + 1e-10)
    
    # Find peak and -60dB point
    peak_idx = np.argmax(rir_db)
    peak_level = rir_db[peak_idx]
    
    # Find where signal drops to -60dB from peak
    target_level = peak_level - 60
    decay_portion = rir_db[peak_idx:]
    
    # Find first point below target level
    below_target = np.where(decay_portion < target_level)[0]
    if len(below_target) > 0:
        rt60_samples = below_target[0]
        return rt60_samples / sample_rate
    else:
        return len(decay_portion) / sample_rate