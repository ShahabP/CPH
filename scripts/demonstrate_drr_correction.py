#!/usr/bin/env python3
"""
Demonstrate the difference between computing DRR on RIR vs on dereverberated speech.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf
from scipy import signal

# Import our modules
import sys
sys.path.append('src')
from neural_rir_agent import NeuralRIRAgent, compute_rir_drr_metric

def generate_test_audio(sample_rate=16000, duration=1.0):
    """Generate test reverberant and clean audio."""
    # Generate clean speech signal (simple speech-like signal)
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Multi-component signal (voice-like)
    clean = (
        0.5 * np.sin(2 * np.pi * 200 * t) +  # Fundamental
        0.3 * np.sin(2 * np.pi * 400 * t) +  # First harmonic  
        0.2 * np.sin(2 * np.pi * 600 * t)    # Second harmonic
    )
    
    # Apply amplitude modulation (speech envelope)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 5 * t))  # 5Hz modulation
    clean = clean * envelope
    
    # Generate simple room impulse response
    rir_length = 1024
    true_rir = np.zeros(rir_length)
    true_rir[0] = 1.0  # Direct sound
    
    # Early reflections (exponential decay)
    for i in range(1, 64):
        true_rir[i] = 0.3 * np.exp(-i / 20) * (1 + 0.1 * np.random.randn())
    
    # Late reverberation (slower decay)
    for i in range(64, 256):
        true_rir[i] = 0.1 * np.exp(-(i-64) / 80) * (1 + 0.2 * np.random.randn())
    
    # Convolve to create reverberant speech
    reverb = signal.convolve(clean, true_rir, mode='same')
    
    return clean, reverb, true_rir

def compare_drr_methods():
    """Compare DRR computation methods."""
    print("🔍 COMPARING DRR COMPUTATION METHODS")
    print("=" * 50)
    
    # Generate test signals
    clean_audio, reverb_audio, true_rir = generate_test_audio()
    
    # Create agent for dereverberation
    agent = NeuralRIRAgent(rir_length=1024)
    
    print("📊 Test Scenario:")
    print(f"   Clean audio length: {len(clean_audio)} samples")
    print(f"   Reverb audio length: {len(reverb_audio)} samples") 
    print(f"   True RIR length: {len(true_rir)} samples")
    print()
    
    # Method 1: DRR computed on RIR structure (OLD - INCORRECT)
    rir_structural_drr = compute_rir_drr_metric(true_rir)
    print("❌ OLD METHOD - DRR from RIR structure:")
    print(f"   Structural RIR 'DRR': {rir_structural_drr:.2f} dB")
    print("   (This is NOT the actual DRR - just RIR structure metric)")
    print()
    
    # Method 2: DRR computed on dereverberated speech (NEW - CORRECT)
    try:
        # Use the true RIR to dereverberate
        dereverberated = agent._dereverberate_with_rir(reverb_audio, true_rir)
        speech_drr = agent._compute_speech_drr(dereverberated)
        
        print("✅ NEW METHOD - DRR from dereverberated speech:")
        print(f"   Speech DRR: {speech_drr:.2f} dB") 
        print("   (This is the actual DRR of the dereverberated speech)")
        print()
        
        # Also compute DRR of original reverberant speech for comparison
        original_drr = agent._compute_speech_drr(reverb_audio)
        print("📈 For comparison:")
        print(f"   Original reverb speech DRR: {original_drr:.2f} dB")
        print(f"   Improvement: {speech_drr - original_drr:.2f} dB")
        print()
        
    except Exception as e:
        print(f"❌ Error in dereverberation: {e}")
        speech_drr = None
    
    # Demonstrate with different RIR estimates
    print("🧪 TESTING WITH DIFFERENT RIR ESTIMATES:")
    print("-" * 40)
    
    # Test case 1: Perfect RIR (true RIR)
    if speech_drr is not None:
        print(f"Perfect RIR → Speech DRR: {speech_drr:.2f} dB")
    
    # Test case 2: Direct-path only RIR 
    direct_only_rir = np.zeros_like(true_rir)
    direct_only_rir[0] = 1.0
    
    try:
        direct_dereverberated = agent._dereverberate_with_rir(reverb_audio, direct_only_rir)
        direct_drr = agent._compute_speech_drr(direct_dereverberated)
        print(f"Direct-only RIR → Speech DRR: {direct_drr:.2f} dB")
        
        direct_structural = compute_rir_drr_metric(direct_only_rir)
        print(f"   (RIR structural metric: {direct_structural:.2f} dB)")
        
    except Exception as e:
        print(f"❌ Error with direct-only RIR: {e}")
    
    # Test case 3: Random RIR
    random_rir = np.random.randn(1024) * 0.1
    random_rir[0] = 1.0  # Ensure direct sound
    
    try:
        random_dereverberated = agent._dereverberate_with_rir(reverb_audio, random_rir)
        random_drr = agent._compute_speech_drr(random_dereverberated)
        print(f"Random RIR → Speech DRR: {random_drr:.2f} dB")
        
        random_structural = compute_rir_drr_metric(random_rir)
        print(f"   (RIR structural metric: {random_structural:.2f} dB)")
        
    except Exception as e:
        print(f"❌ Error with random RIR: {e}")
    
    print()
    print("🎯 KEY INSIGHTS:")
    print("   1. RIR structural metrics don't reflect dereverberation quality")
    print("   2. Speech DRR measures actual dereverberation performance") 
    print("   3. The same RIR structure can give very different speech results")
    print("   4. Training should optimize speech DRR, not RIR structure")
    print()
    print("✅ CORRECTED METHOD: Reward based on dereverberated speech DRR!")

if __name__ == "__main__":
    compare_drr_methods()