# Module Relationship Specification - Smart Haptic Alert System

## 1. System Module Interactions

This document describes the structural relationships, data flow dependencies, and integration interfaces between subsystems in the **Smart Haptic Alert System**.

```
                       +-----------------------------------+
                       |        config.py / settings       |
                       +-----------------+-----------------+
                                         |
               +-------------------------+-------------------------+
               |                         |                         |
               v                         v                         v
+-----------------------------+ +-----------------+ +------------------------------+
| app/ai/dataset/             | | app/context/    | | app/bluetooth/               |
| - DatasetDirectoryManager   | | - ModeProfile   | | - HapticPacketSerializer     |
| - AudioDatasetLoader        | | - Priority      | | - ESP32BLEManager            |
| - DatasetValidator          | | - ContextMgr    | +--------------+---------------+
| - DatasetStatistics         | +--------+--------+                |
| - DatasetExplorer           |          |                         |
+--------------+--------------+          |                         |
               |                         |                         |
               | Raw Audio Paths         | Mode Priority           | Haptic Binary
               v                         v                         v
+-----------------------------+ +-----------------+ +------------------------------+
| app/ai/preprocessing/       | | app/services/   | | ESP32 Wearable Device        |
| - AudioLoader               | | - AlertService  |<+ (Haptic Vibration Motor)     |
| - AudioStandardizer (22kHz) | | - AudioService  | +------------------------------+
| - SilenceProcessor (-40dB)  | +--------+--------+
| - NoiseReducer              |          ^ Handled Alert Record
| - LengthStandardizer (4.0s) |          |
| - PreprocessingPipeline     |          |
+--------------+--------------+          |
               | 22050Hz 4.0s WAV        |
               v                         |
+-----------------------------+          |
| app/ai/feature_extraction/  |          |
| - FeatureExtractor (7 Feats)|          |
| - LabelEncoder              |          |
| - FeatureNormalizer         |          |
| - StratifiedDatasetSplitter |          |
| - FeatureStorageManager     |          |
| - FeatureVisualizer         |          |
| - FeatureExtractionPipeline |          |
+--------------+--------------+          |
               | dataset_splits.npz      |
               | class_names.json        |
               v                         |
+-----------------------------+          |
| app/ai/inference/           |          |
| - SoundInferenceEngine -----+----------+
+-----------------------------+
```

---

## 2. Detailed Data Handshake Specifications

### 2.1 Dataset Subsystem -> Preprocessing Subsystem
- **Provider**: `app.ai.dataset.AudioDatasetLoader` & `DatasetDirectoryManager`
- **Consumer**: `app.ai.preprocessing.PreprocessingPipeline`
- **Data Object**: Raw sound file paths (`dataset/raw/{ambulance, car_horn, fire_alarm, doorbell, dog_bark}/*.wav`).
- **Contract**: `PreprocessingPipeline` processes raw files and outputs standardized samples to `dataset/processed/{classes}`.

### 2.2 Preprocessing Subsystem -> Feature Extraction
- **Provider**: `app.ai.preprocessing.PreprocessingPipeline`
- **Consumer**: `app.ai.feature_extraction.FeatureExtractionPipeline` & `FeatureExtractor`
- **Data Object**: Standardized 16-bit PCM WAV audio file (22,050 Hz, Mono, 4.0s = 88,200 samples) and `preprocessed_metadata.json`.
- **Contract**: `FeatureExtractionPipeline` loads preprocessed audio from `dataset/processed/{classes}`, computes 7 acoustic feature representations, and outputs dataset archives to `app/ai/features/`.

### 2.3 Feature Extraction -> AI Model Training / Real-Time Inference
- **Provider**: `app.ai.feature_extraction` (`FeatureExtractor`, `LabelEncoder`, `FeatureNormalizer`, `FeatureStorageManager`)
- **Consumer**: `BaseSoundClassifier` (Phase 5) & `SoundInferenceEngine` (Phase 7)
- **Data Object**: `dataset_splits.npz` (`X_composite_train`, `X_composite_val`, `X_composite_test`, `X_vectors_train`, `y_train`, etc.), `class_names.json`, `scaler_params.json`, and `feature_metadata.json`.
- **Contract**: Phase 5 training loads normalized dataset splits to train CNN models. Phase 7 real-time inference uses `FeatureExtractor` and `scaler_params.json` to extract and normalize features from incoming streaming audio frames.

### 2.4 AI Inference Engine -> Context Engine & Alert Service
- **Provider**: `SoundInferenceEngine`
- **Consumer**: `ContextManager` & `AlertService`
- **Data Object**: Tuple of `(sound_label: str, confidence_score: float)`.

### 2.5 Alert Service -> Bluetooth Subsystem
- **Provider**: `AlertService`
- **Consumer**: `HapticPacketSerializer` & `ESP32BLEManager`
- **Data Object**: `alert_id` string and `SoundPriority` enum.
- **Contract**: `HapticPacketSerializer.encode()` produces 6-byte binary payload transmitted over BLE GATT write to ESP32.
