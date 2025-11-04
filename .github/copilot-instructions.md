# Reinforcement Learning for Room Impulse Response Estimation

## Project Overview
This project develops a reinforcement learning system for extracting room impulse responses from reverberant speech through iterative blind dereverberation.

## Key Components
- Blind dereverberation algorithms (spectral subtraction, Wiener filtering, deep learning approaches)
- Room impulse response (RIR) estimation using deconvolution techniques
- Reinforcement learning framework for iterative improvement
- Audio processing utilities for feature extraction and preprocessing
- Training pipelines for RL agents and evaluation metrics

## Technical Stack
- Python 3.8+ with PyTorch for deep learning and RL
- librosa, scipy, numpy for audio signal processing
- gym/gymnasium for RL environment framework
- matplotlib, tensorboard for visualization and monitoring
- pytest for testing audio processing and RL components

## Development Guidelines
- Use proper audio signal processing practices (windowing, overlap-add, etc.)
- Implement modular architecture for easy experimentation
- Focus on real-time feasibility for practical applications
- Validate algorithms with both synthetic and real acoustic data
- Document acoustic assumptions and model limitations
 
Progress summary:
- Project scaffolded with src/ layout and core modules
- Minimal deps installed; smoke test task added and verified
- README created with quick-start commands