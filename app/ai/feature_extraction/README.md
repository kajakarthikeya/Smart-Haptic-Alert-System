# Feature Extraction Subsystem (`app/ai/feature_extraction/`)

## Purpose
Transforms 1D time-domain audio signals into 2D time-frequency acoustic representations (Log-Mel Spectrograms) suitable for Convolutional Neural Network (CNN) classification models.

## Key Hyperparameters
- **Sample Rate**: 16,000 Hz
- **FFT Window Size (`n_fft`)**: 512 samples (~32 ms)
- **Hop Length (`hop_length`)**: 160 samples (~10 ms)
- **Mel Filterbanks (`n_mels`)**: 64 bins
