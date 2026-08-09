# Feature Extraction Subsystem (`app/ai/feature_extraction/`)

## Subsystem Overview
The Feature Extraction subsystem converts preprocessed 4.0-second 22,050 Hz mono environmental audio waveforms into standardized numerical feature matrices and summary vectors suitable for training AI sound classification models and real-time inference.

Target sound classes:
- `ambulance` (ID: 0)
- `car_horn` (ID: 1)
- `fire_alarm` (ID: 2)
- `doorbell` (ID: 3)
- `dog_bark` (ID: 4)

---

## Architecture & Extraction Workflow

```
Preprocessed Audio (dataset/processed/)
          │
          ▼
   Audio Loader & Validator (Librosa / Scipy)
          │
          ▼
   Acoustic Feature Extraction Engine (Librosa)
   ├── Mel Spectrogram (128 dB bands)
   ├── MFCC (40 coefficients)
   ├── Zero Crossing Rate (ZCR)
   ├── Spectral Centroid
   ├── Spectral Bandwidth
   ├── Spectral Rolloff
   └── Chroma STFT (12 bins)
          │
          ▼
   Validation (NaN / Inf / Shape Check)
          │
          ▼
   Class Label Encoding (LabelEncoder)
          │
          ▼
   Stratified Train / Val / Test Splitter (70% / 15% / 15%)
          │
          ▼
   Feature Normalization (FeatureNormalizer - Z-score / MinMax)
          │
          ▼
   Storage & Metadata Generator (FeatureStorageManager)
   ├── app/ai/features/dataset_splits.npz
   ├── app/ai/features/class_names.json
   ├── app/ai/features/scaler_params.json
   └── app/ai/features/feature_metadata.json
          │
          ▼
   Feature Visualizer (app/outputs/feature_visualizations/)
```

---

## Features Extracted

1. **MFCC (Mel-Frequency Cepstral Coefficients)**: 40 coefficients capturing spectral envelope characteristics.
2. **Mel Spectrogram**: 128 Mel-frequency bands converted to decibel (dB) log-power scale.
3. **Zero Crossing Rate (ZCR)**: Measure of silent/voiced frame transitions and noisiness.
4. **Spectral Centroid**: Center of mass of the spectrum (brightness metric).
5. **Spectral Bandwidth**: Spectral spread around the centroid.
6. **Spectral Rolloff**: Cutoff frequency below which 85% of total spectral energy lies.
7. **Chroma STFT**: 12 pitch class profiles useful for harmonic alerts (e.g., doorbells, alarms).

### Composite Feature Dimensions
For standard 4.0s audio clips at 22,050 Hz with `n_fft=2048`, `hop_length=512`:
- **Composite Matrix Shape**: `[184, 173]` (128 + 40 + 1 + 1 + 1 + 1 + 12 = 184 feature channels across 173 time frames).
- **Summary Vector Shape**: `[368]` (Mean + Std concatenated across time frames).

---

## Class Label Mapping (`class_names.json`)
```json
{
  "class_to_id": {
    "ambulance": 0,
    "car_horn": 1,
    "fire_alarm": 2,
    "doorbell": 3,
    "dog_bark": 4
  },
  "id_to_class": {
    "0": "ambulance",
    "1": "car_horn",
    "2": "fire_alarm",
    "3": "doorbell",
    "4": "dog_bark"
  }
}
```

---

## Configuration Options (`config.py`)

Feature extraction behavior is configured via `FeatureExtractionConfig` in `config.py`:
- `n_mfcc`: 40
- `n_fft`: 2048
- `hop_length`: 512
- `n_mels`: 128
- `fmin`: 0.0 Hz
- `fmax`: null (defaults to `sample_rate / 2` = 11,025 Hz)
- `n_chroma`: 12
- `enable_normalization`: true
- `normalization_type`: "z_score"
- `train_ratio`: 0.70
- `val_ratio`: 0.15
- `test_ratio`: 0.15
- `random_seed`: 42
- `features_dir`: `app/ai/features/`
- `visualization_dir`: `app/outputs/feature_visualizations/`

---

## Usage Instructions

### 1. Programmatic Extraction
```python
from app.ai.feature_extraction import FeatureExtractionPipeline

pipeline = FeatureExtractionPipeline()
summary = pipeline.run(generate_visualizations=True)
print("Pipeline Run Summary:", summary)
```

### 2. Individual Feature Extractor Usage
```python
from app.ai.feature_extraction import FeatureExtractor, LabelEncoder

extractor = FeatureExtractor()
encoder = LabelEncoder()

# Extract composite 2D matrix
matrix = extractor.extract_composite_matrix(waveform, sr=22050)

# Encode label string
label_id = encoder.encode("ambulance")
```

---

## Generated Artifacts & Storage (`app/ai/features/`)

- `dataset_splits.npz`: Compressed archive containing arrays `X_composite_train`, `X_composite_val`, `X_composite_test`, `X_vectors_train`, `X_vectors_val`, `X_vectors_test`, `y_train`, `y_val`, `y_test`.
- `class_names.json`: Persistent label mapping file.
- `scaler_params.json`: Serialized mean and std arrays for zero-data-leakage inference matching.
- `feature_metadata.json`: Dataset extraction summary report.
- Visualizations (`app/outputs/feature_visualizations/`):
  - `class_distribution.png`
  - `mel_spectrogram.png`
  - `mfcc_heatmap.png`

---

## Integration with Future Phases

- **Phase 5 (AI Model Training)**: Loads `dataset_splits.npz`, `class_names.json`, and `scaler_params.json` to train CNN sound classifiers.
- **Phase 7 (Real-Time Sound Recognition)**: Employs `FeatureExtractor`, `LabelEncoder.decode()`, and `FeatureNormalizer.transform()` with loaded `scaler_params.json` to extract features from 1-second audio frame buffers in real-time.
