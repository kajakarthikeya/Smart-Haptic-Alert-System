# Software Architecture Specification - Smart Haptic Alert System

## 1. Executive Architectural Overview

The **Smart Haptic Alert System** is an AI-powered assistance platform designed for hearing-impaired users. The system captures real-time environmental sound streams, classifies target acoustic events using deep neural networks, prioritizes alerts according to user context (**Home**, **Road**, **Office**), and dispatches haptic vibration patterns to an **ESP32 wearable device**.

The architecture adheres strictly to **Clean Architecture** principles, maintaining clear boundaries between domain business logic, AI pipelines, hardware transport protocols, and delivery layers.

```
+-----------------------------------------------------------------------------------+
|                                 DELIVERY LAYER                                    |
|   +-------------------------------------+     +-------------------------------+   |
|   | FastAPI REST Routers (app/api/)     |     | Mobile App Companion          |   |
|   +------------------+------------------+     +---------------+---------------+   |
+----------------------|----------------------------------------|-------------------+
                       v                                        v
+-----------------------------------------------------------------------------------+
|                               CONTROLLER LAYER                                    |
|   +-------------------------------------+     +-------------------------------+   |
|   | AlertController (app/controllers/)  |     | ModeController                |   |
|   +------------------+------------------+     +---------------+---------------+   |
+----------------------|----------------------------------------|-------------------+
                       v                                        v
+-----------------------------------------------------------------------------------+
|                                SERVICE LAYER                                      |
|   +-------------------------------------+     +-------------------------------+   |
|   | AlertService (app/services/)        |     | AudioService                  |   |
|   +------------------+------------------+     +---------------+---------------+   |
+----------------------|----------------------------------------|-------------------+
                       v                                        v
+-----------------------------------------------------------------------------------+
|                            CONTEXT ENGINE LAYER                                   |
|   +---------------------------------------------------------------------------+   |
|   | ContextManager (Home, Road, Office Mode Priority Matrix & Thresholds)     |   |
|   +------------------------------------+--------------------------------------+   |
+----------------------------------------|------------------------------------------+
                                         |
               +-------------------------+-------------------------+
               |                                                   |
               v                                                   v
+-----------------------------------------+     +-----------------------------------+
|          AI PIPELINE SUBSYSTEM          |     |    BLUETOOTH HARDWARE SUBSYSTEM   |
|  - Dataset Management (app/ai/dataset/) |     |  - HapticPacketSerializer         |
|  - Preprocessing (app/ai/preprocessing/)|     |  - ESP32BLEManager                |
|  - Feature Extraction (Phase 4 Next)    |     +-----------------+-----------------+
|  - BaseSoundClassifier / ModelFactory   |                       |
+-----------------------------------------+                       v
                                                +-----------------------------------+
                                                |     ESP32 Wearable Device         |
                                                |     (Haptic Vibration Motors)     |
                                                +-----------------------------------+
```

---

## 2. Core Subsystems

### 2.1 AI Subsystem (`app/ai/`)
Structured into decoupled pipeline phases:
- **`dataset/`**: Directory management, multi-format loading (`.wav`, `.mp3`, `.flac`), validation, statistics, and exploration. Target classes: `ambulance`, `car_horn`, `fire_alarm`, `doorbell`, `dog_bark`.
- **`preprocessing/`**: Audio loading (`AudioLoader`), format & rate standardization to **22,050 Hz Mono** (`AudioStandardizer`), silence trimming (`SilenceProcessor`), optional noise reduction (`NoiseReducer`), fixed **4.0-second length standardization** (`LengthStandardizer`), and batch execution (`PreprocessingPipeline`).
- **`feature_extraction/`**: Librosa-based 7-feature extraction (`FeatureExtractor`), persistent label encoding (`LabelEncoder`), Z-score/MinMax feature normalization (`FeatureNormalizer`), stratified 70/15/15 train/val/test splitting (`StratifiedDatasetSplitter`), storage manager (`FeatureStorageManager`), visualization generator (`FeatureVisualizer`), and batch execution (`FeatureExtractionPipeline`).
- **`models/`**: `BaseSoundClassifier` contract, `ModelFactory` registry, and `CNNSoundClassifier` (hierarchical 2D CNN with 111,237 parameters optimized for composite 184x173 acoustic feature maps).
- **`training/`**: `TrainingDataLoader` with feature/label validation and class weighting, `ModelTrainer` with early stopping, model checkpointing (`sound_classifier_best.keras`), and learning rate scheduling, `TrainingVisualizer` for accuracy/loss curves, and `TrainingPipeline` for end-to-end training orchestration.
- **`evaluation/`**: `EvaluationDataLoader` with test split verification, `EvaluationMetricsCalculator` (accuracy, cross-entropy loss, per-class metrics, macro/weighted averages), `ConfusionMatrixGenerator` (raw and recall-normalized), `PredictionAnalyzer` (confidence score stats and error pairs), `EvaluationVisualizer` (heatmaps, bar charts, confidence distribution), `EvaluationReportGenerator` (JSON & TXT reports), and master `ModelEvaluator` orchestrator.
- **`inference/`**: Real-time acoustic recognition subsystem (`RealtimeSoundRecognizer`, `AudioDeviceManager`, `MicrophoneAudioCapture`, `InferenceModelLoader`, `PredictionStabilizer`). Implements non-blocking circular buffer capture (4.0s = 88,200 samples at 22,050 Hz), direct reuse of preprocessing and composite feature extraction pipelines, single-load model caching with warm-up pass, configurable confidence gating (>= 70%), temporal consensus stabilization (N=3, K=2), and interactive CLI (`app.ai.inference.cli`).

### 2.2 Environmental Context Subsystem (`app/context/`)
- **`ContextDecisionEngine`**: Evaluates incoming `SoundPrediction` or `PredictionResult` against active `EnvironmentMode` (`HOME`, `ROAD`, `OFFICE`), validates AI confidence against threshold (default: 0.70), and issues structured `DecisionResult` with transparent reasoning.
- **`PriorityEngine`**: Performs configuration-driven $O(1)$ matrix lookups mapping target sounds (`ambulance`, `car_horn`, `fire_alarm`, `doorbell`, `dog_bark`) to `PriorityLevel` (`HIGH`, `MEDIUM`, `LOW`, `IGNORE`).
- **`ModeManager`**: Manages active user operating mode state, validates transitions, alerts observer callbacks, and provides safe resets.
- **`ContextManager`**: High-level facade integrating the decision engine and mode manager while preserving backward compatibility.

### 2.3 Bluetooth Hardware Subsystem (`app/bluetooth/`)
- `HapticPacketSerializer`: Encodes alert ID and priority into a 6-byte binary payload.
- `ESP32BLEManager`: Manages BLE GATT characteristic writes to the ESP32 wristband (to be implemented in Phase 9).

### 2.4 Software Integration Service & Web Prototype (`app/services/` & `app/web/`)
- **`SoftwareIntegrationService`**: Coordinates Phase 7 real-time recognition, Phase 8 context decision engine, and mode manager. Provides sample audio evaluation, demo simulation, live microphone streaming, scenario testing, and in-memory alert history.
- **FastAPI REST API Layer (`app/api/routes.py`)**: Modular endpoints for status, mode control, test audio evaluation, demo simulation, and scenario benchmarking.
- **Web Frontend Prototype (`app/web/`)**: Vanilla HTML5/CSS/JS dashboard displaying system status, mode control, live inference cards, confidence bars, priority badges, test audio analysis, demo simulation, and recent alert history without any external hardware requirements.

### 2.5 Hardware Independence Statement
The system operates completely in **software-only mode** during verification. Physical hardware (ESP32, INMP441, vibration motor, Bluetooth LE) was **NOT required** for this integration phase.

---

## 3. SOLID Principles Enforcement

- **Single Responsibility Principle (SRP)**: Each preprocessing class handles a single signal operation (`AudioStandardizer` standardizes rate/channels/amplitude; `SilenceProcessor` trims silence; `LengthStandardizer` handles clip length).
- **Open/Closed Principle (OCP)**: New preprocessors or noise filters can be registered in `PreprocessingPipeline` without modifying existing signal standardizers.
- **Liskov Substitution Principle (LSP)**: All preprocessor classes adhere to explicit contracts.
- **Interface Segregation Principle (ISP)**: Interfaces remain lean and focused (`BasePreprocessor`, `AudioLoader`, `SilenceProcessor`).
- **Dependency Inversion Principle (DIP)**: Higher-level services (`PreprocessingPipeline`, `AlertService`) depend on abstract contracts rather than hardcoded implementations.
