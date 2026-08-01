# Inference Engine Subsystem (`app/ai/inference/`)

## Purpose
Coordinates real-time audio frame processing, feature extraction, and neural network classification into a single unified `SoundInferenceEngine`.

## Pipeline Sequence

```
Raw Audio Stream PCM
        │
        ▼
[AudioPreprocessor] ──> Normalization & Framing (1.0s window)
        │
        ▼
[SpectrogramExtractor] ──> Log-Mel Spectrogram Tensor
        │
        ▼
[BaseSoundClassifier] ──> Sound Label & Confidence Score
```
