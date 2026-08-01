# Audio Preprocessing Package (`app/ai/preprocessing/`)

## Purpose
Prepares incoming raw audio waveforms for feature extraction and model inference.

## Key Operations
1. **Resampling**: Converting microphone audio to target sample rate (16 kHz).
2. **Peak Amplitude Normalization**: Scaling waveform values to [-1.0, 1.0].
3. **Fixed Duration Framing**: Padding or truncating audio clips to 1.0 second windows.

## Interfaces
- `BasePreprocessor`: Abstract Base Class contract.
- `AudioPreprocessor`: Concrete audio processing pipeline component.
