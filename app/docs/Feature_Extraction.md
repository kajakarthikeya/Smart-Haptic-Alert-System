# Feature Extraction Subsystem Architecture Document

## Executive Summary
This document specifies the technical design, feature set, data formats, normalization strategies, and dataset splitting procedures for **Phase 4: Feature Extraction** of the **Smart Haptic Alert System**.

The Feature Extraction subsystem converts preprocessed mono environmental audio clips (4.0 seconds at 22,050 Hz) from Phase 3 into numerical feature matrices and vectors ready for Phase 5 (AI Model Training) and Phase 7 (Real-Time Sound Recognition).

---

## 1. Primary Objectives
- Extract 7 key acoustic feature representations using **Librosa**.
- Validate feature arrays against NaN, Inf, empty arrays, and dimension mismatches.
- Perform persistent bidirectional class label encoding (`class_names.json`).
- Implement zero-data-leakage feature normalization (Z-score / MinMax) exporting scaler parameters (`scaler_params.json`).
- Provide reproducible 70% / 15% / 15% stratified train/validation/test dataset splitting with configurable random seeds.
- Store feature archives (`dataset_splits.npz`) and metadata (`feature_metadata.json`) in `app/ai/features/`.
- Export feature visualizations to `app/outputs/feature_visualizations/`.

---

## 2. Acoustic Features Summary

| Feature Name | Library Function | Output Shape | Description |
| :--- | :--- | :--- | :--- |
| **Mel Spectrogram** | `librosa.feature.melspectrogram` | `[128, 173]` | 128 Mel bands converted to log-power decibel (dB) scale |
| **MFCC** | `librosa.feature.mfcc` | `[40, 173]` | 40 Mel-Frequency Cepstral Coefficients |
| **Zero Crossing Rate** | `librosa.feature.zero_crossing_rate` | `[1, 173]` | Rate of sign changes in time-domain waveform |
| **Spectral Centroid** | `librosa.feature.spectral_centroid` | `[1, 173]` | Center frequency weighted by energy |
| **Spectral Bandwidth** | `librosa.feature.spectral_bandwidth` | `[1, 173]` | Spectral variance around centroid |
| **Spectral Rolloff** | `librosa.feature.spectral_rolloff` | `[1, 173]` | Frequency below which 85% of energy is concentrated |
| **Chroma STFT** | `librosa.feature.chroma_stft` | `[12, 173]` | 12 pitch class distribution bins |

### Composite Array Dimensions
- **Composite Matrix Shape**: `[184, 173]` per 4.0-second clip (128 + 40 + 1 + 1 + 1 + 1 + 12 = 184 features across 173 time frames).
- **Summary Vector Shape**: `[368]` per clip (Mean + Std across time frames).

---

## 3. Persistent Label Encoding (`class_names.json`)
The `LabelEncoder` establishes deterministic integer mappings for target sound classes:

| Class Name | Integer Label |
| :--- | :--- |
| `ambulance` | `0` |
| `car_horn` | `1` |
| `fire_alarm` | `2` |
| `doorbell` | `3` |
| `dog_bark` | `4` |

---

## 4. Normalization & Dataset Splitting
1. **Stratified Dataset Splitter**: Splits dataset into 70% Training, 15% Validation, and 15% Testing. Stratification ensures balanced class distribution across all splits.
2. **Feature Normalizer**: Fits Z-score statistics (`mean` and `std`) strictly on the training set to eliminate data leakage. The fitted scaler is saved to `app/ai/features/scaler_params.json` for live real-time inference matching.

---

## 5. Directory & Storage Artifacts
Extracted features and metadata are stored in `app/ai/features/`:
- `dataset_splits.npz`: Compressed archive containing arrays `X_composite_train`, `X_composite_val`, `X_composite_test`, `X_vectors_train`, `X_vectors_val`, `X_vectors_test`, `y_train`, `y_val`, `y_test`.
- `class_names.json`: Serialized class mapping dictionary.
- `scaler_params.json`: Serialized feature mean and standard deviation matrices.
- `feature_metadata.json`: Feature extraction run report.
- Visualizations (`app/outputs/feature_visualizations/`):
  - `class_distribution.png`
  - `mel_spectrogram.png`
  - `mfcc_heatmap.png`

---

## 6. Error Handling & Validation
Custom exception hierarchy defined in `app/ai/feature_extraction/exceptions.py`:
- `FeatureExtractionError`: Base exception class.
- `InvalidFeatureError`: Raised when features contain NaN, Inf, or empty arrays.
- `FeatureShapeError`: Raised on feature shape mismatches.
- `LabelEncodingError`: Raised on unknown class names or IDs.
- `FeatureStorageError`: Raised on serialization/deserialization failures.

All batch pipeline operations utilize `get_logger(__name__)` for centralized structured logging.
