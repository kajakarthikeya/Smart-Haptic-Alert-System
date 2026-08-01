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
| - SpectrogramExtractor      |          |
+--------------+--------------+          |
               | Spectrogram Matrix      |
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
- **Consumer**: `SpectrogramExtractor` (Phase 4)
- **Data Object**: Standardized 16-bit PCM WAV audio file (22,050 Hz, Mono, 4.0s = 88,200 samples) and `preprocessed_metadata.json`.
- **Contract**: Feature extractor loads 4.0s 22,050 Hz preprocessed audio samples for spectrogram conversion.

### 2.3 Feature Extraction -> AI Model / Inference Engine
- **Provider**: `SpectrogramExtractor`
- **Consumer**: `BaseSoundClassifier` / `SoundInferenceEngine`
- **Data Object**: 2D Log-Mel Spectrogram matrix `(64, time_frames)`.

### 2.4 AI Inference Engine -> Context Engine & Alert Service
- **Provider**: `SoundInferenceEngine`
- **Consumer**: `ContextManager` & `AlertService`
- **Data Object**: Tuple of `(sound_label: str, confidence_score: float)`.

### 2.5 Alert Service -> Bluetooth Subsystem
- **Provider**: `AlertService`
- **Consumer**: `HapticPacketSerializer` & `ESP32BLEManager`
- **Data Object**: `alert_id` string and `SoundPriority` enum.
- **Contract**: `HapticPacketSerializer.encode()` produces 6-byte binary payload transmitted over BLE GATT write to ESP32.
