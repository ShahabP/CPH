import os
import numpy as np
import soundfile as sf

from pathlib import Path


def synth_segment(sr=16000, dur=1.2, f0=200.0, rir_len=512):
    t = np.linspace(0, int(dur * sr), int(dur * sr), endpoint=False) / sr
    clean = 0.2 * (np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * (2*f0) * t))
    rir = np.zeros(rir_len)
    rir[0] = 1.0
    if rir_len > 120:
        rir[120] = 0.25
    if rir_len > 240:
        rir[240] = 0.12
    reverb = np.convolve(clean, rir, mode='same')
    reverb = np.clip(reverb, -1.0, 1.0)
    return reverb


def main(out_dir='data/stream'):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pitches = [180.0, 220.0, 260.0, 300.0]
    for i, f0 in enumerate(pitches):
        audio = synth_segment(f0=f0)
        path = Path(out_dir) / f"segment_{i:02d}.wav"
        sf.write(str(path), audio, 16000)
        print("Wrote", path)


if __name__ == '__main__':
    main()
