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
|  app/ai/dataset/            | | app/context/    | | app/bluetooth/               |
|  - DatasetDirectoryManager  | | - ModeProfile   | | - HapticPacketSerializer     |
|  - AudioDatasetLoader       | | - Priority    | | - ESP32BLEManager            |
|  - DatasetValidator         | | - ContextMgr  | +--------------+---------------+
|  - DatasetStatistics        | +--------+--------+                |
|  - DatasetExplorer          |          |                         |
+--------------+--------------+          |                         |
               |                         |                         |
               | Manifest/Paths          | Mode Priority           | Haptic Binary
               v                         v                         v
+-----------------------------+ +-----------------+ +------------------------------+
| app/ai/preprocessing/       | | app/services/   | | ESP32 Wearable Device        |
| - AudioPreprocessor         | | - AlertService  |<+ (Haptic Vibration Motor)     |
+--------------+--------------+ | - AudioService  | +------------------------------+
               | Cleaned Signal +--------+--------+
               v                         ^
+-----------------------------+          | Handled Alert Record
| app/ai/feature_extraction/  |          |
| - SpectrogramExtractor      |          |
+--------------+--------------+          |
               | Spectrogram             |
               v                         |
+-----------------------------+          |
| app/ai/inference/           |          |
| - SoundInferenceEngine -----+----------+
+-----------------------------+
```

---

## 2. Detailed Data Handshake Specifications

### 2.1 Dataset Subsystem -> Preprocessing Subsystem
- **Provider**: `app.ai.dataset.AudioDatasetLoader`
- **Consumer**: `app.ai.preprocessing.AudioPreprocessor` (Phase 3)
- **Data Object**: `DatasetManifest` containing `DatasetItem` entries with `metadata.file_path` pointing to raw sound samples (`ambulance`, `car_horn`, `fire_alarm`, `doorbell`, `dog_bark`).
- **Contract**: `AudioPreprocessor` consumes valid file paths validated by `DatasetValidator`.

### 2.2 Preprocessing Subsystem -> Feature Extraction
- **Provider**: `AudioPreprocessor`
- **Consumer**: `SpectrogramExtractor` (Phase 4)
- **Data Object**: 1D normalized float32 waveform numpy/tensor array (16,000 samples for 1.0 second).

### 2.3 Feature Extraction -> AI Model / Inference Engine
- **Provider**: `SpectrogramExtractor`
- **Consumer**: `BaseSoundClassifier` / `SoundInferenceEngine`
- **Data Object**: 2D Log-Mel Spectrogram matrix `(64, time_frames)`.

### 2.4 AI Inference Engine -> Context Engine & Alert Service
- **Provider**: `SoundInferenceEngine`
- **Consumer**: `ContextManager` & `AlertService`
- **Data Object**: Tuple of `(sound_label: str, confidence_score: float)`.
- **Contract**: `AlertService` passes sound label and confidence to `ContextManager.evaluate_sound()`.

### 2.5 Alert Service -> Bluetooth Subsystem
- **Provider**: `AlertService`
- **Consumer**: `HapticPacketSerializer` & `ESP32BLEManager`
- **Data Object**: `alert_id` string and `SoundPriority` enum.
- **Contract**: `HapticPacketSerializer.encode()` produces 6-byte binary payload transmitted over BLE GATT write.
