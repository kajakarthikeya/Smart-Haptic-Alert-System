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

### 2.3 Feature Extraction -> AI Model Training
- **Provider**: `app.ai.feature_extraction` (`FeatureExtractor`, `LabelEncoder`, `FeatureNormalizer`, `FeatureStorageManager`)
- **Consumer**: `app.ai.training.TrainingDataLoader` & `TrainingPipeline` (Phase 5)
- **Data Object**: `dataset_splits.npz` (`X_composite_train`, `X_composite_val`, `X_composite_test`), `class_names.json`, and `feature_metadata.json`.
- **Contract**: Phase 5 training pipeline ingests normalized 2D composite matrices `(184, 173)`, validates shapes and data integrity, computes balanced class weights, and trains `CNNSoundClassifier`.

### 2.4 Model Training -> Model Evaluation & Benchmarking (Phase 6)
- **Provider**: `app.ai.training` (`ModelTrainer`, `TrainingPipeline`) & `app.ai.models` (`CNNSoundClassifier`)
- **Consumer**: Phase 6 Evaluation Subsystem (`app.ai.evaluation` - `ModelEvaluator`, `EvaluationDataLoader`, `EvaluationMetricsCalculator`, `ConfusionMatrixGenerator`, `PredictionAnalyzer`, `EvaluationVisualizer`, `EvaluationReportGenerator`)
- **Data Object**: `app/ai/models/sound_classifier_best.keras`, `class_names.json`, and `X_composite_test`/`y_test` splits (`app/ai/features/dataset_splits.npz`).
- **Contract**: Phase 6 loads checkpointed `.keras` weights and unseen test set to compute confusion matrices (raw and normalized), per-class precision/recall/F1-score, confidence distributions, and exports diagnostic artifacts to `app/outputs/model_evaluation/`.

### 2.5 Model Training & Feature Pipeline -> Real-Time Sound Recognition (Phase 7)
- **Provider**: `app.ai.models.CNNSoundClassifier`, `app.ai.preprocessing.AudioStandardizer`, `app.ai.feature_extraction.FeatureExtractor`
- **Consumer**: `app.ai.inference.RealtimeSoundRecognizer`, `app.ai.inference.MicrophoneAudioCapture`, `app.ai.inference.InferenceModelLoader`
- **Data Object**: Live microphone PCM stream (22,050 Hz, Mono) buffered into 4.0s rolling windows (88,200 samples) -> 2D composite feature matrices `(184, 173, 1)` -> Model prediction tensor.
- **Contract**: `RealtimeSoundRecognizer` performs periodic window inference (hop interval = 1.0s), validates confidence against threshold (>= 0.70), applies temporal consensus stabilizer ($N=3, K=2$), and produces `PredictionResult` events (`predicted_class`, `confidence`, `status`, `probabilities`, `latency`).

### 2.6 Real-Time Sound Recognition -> Context-Aware Decision Engine (Phase 8)
- **Provider**: `app.ai.inference.RealtimeSoundRecognizer` (`PredictionResult`)
- **Consumer**: `app.context.ContextDecisionEngine` & `app.context.ContextManager`
- **Data Object**: `PredictionResult` (or `SoundPrediction`) with `predicted_class`, `confidence`, `timestamp`, and `probabilities`.
- **Contract**: Phase 8 filters predictions where `confidence < threshold` (0.70) as `IGNORE` with `alert_required=False`. For confident predictions, it queries `ModeManager` (`HOME`, `ROAD`, `OFFICE`), resolves priority from `PriorityEngine`, and returns a structured `DecisionResult(sound, confidence, mode, priority, alert_required, reason, timestamp)`.

### 2.7 Context Decision Engine -> Alert Service & Bluetooth Subsystem (Phase 9)
- **Provider**: `app.context.ContextDecisionEngine` (`DecisionResult`)
- **Consumer**: `app.services.AlertService`, `app.bluetooth.HapticPacketSerializer`, `app.bluetooth.ESP32BLEManager`
- **Data Object**: `DecisionResult` (specifically `priority: PriorityLevel` and `alert_required: bool`).
- **Contract**: When `alert_required` is `True`, `AlertService` maps the priority level to haptic vibration patterns and encodes a 6-byte binary payload via `HapticPacketSerializer.encode()` for transmission to the ESP32 wearable over BLE GATT characteristic write.

### 2.8 Software Integration Service -> REST API & Web Dashboard Prototype
- **Provider**: `app.services.SoftwareIntegrationService`
- **Consumer**: `app.api.routes` and `app/web/` Dashboard Frontend
- **Data Object**: JSON payloads (`system_status`, `mode`, `test_audio_result`, `demo_simulation`, `alerts_history`).
- **Contract**: The service coordinates Phase 7 `RealtimeSoundRecognizer`, Phase 8 `ContextDecisionEngine`, and `ModeManager`. The web dashboard presents real-time telemetry, synchronized mode buttons, test WAV evaluation, demo simulation, and scenario benchmarking without requiring external hardware.
