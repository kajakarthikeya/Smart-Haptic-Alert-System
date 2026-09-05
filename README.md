# Smart Haptic Alert System

An AI-powered wearable assistance system for hearing-impaired users that detects important environmental sounds, prioritizes them according to the user's selected mode (**Home**, **Road**, **Office**), and communicates haptic alert signals to an **ESP32 wearable device**.

---

## Project Status & Progress Tracker

- [x] **Phase 1: Project Initialization & Clean Architecture Setup**
- [x] **Phase 2: Dataset Management Subsystem**
- [x] **Phase 3: Audio Preprocessing Subsystem**
- [x] **Phase 4: Feature Extraction Subsystem**
- [x] **Phase 5: AI Model Training Subsystem**
- [x] **Phase 6: Model Evaluation & Benchmarking**
- [x] **Phase 7: Real-Time Sound Recognition Engine**
- [x] **Phase 8: Context-Aware Decision Engine**
- [ ] **Phase 9: Bluetooth Hardware Communication** *(Next Phase)*
- [ ] **Phase 10: Mobile Companion Application**
- [ ] **Phase 11: End-to-End System Integration**
- [ ] **Phase 12: Field Testing & Verification**

---

## Completed Subsystems & Features

### Phase 1: Architecture & Bootstrap
- Clean Architecture project layout with strict layer separation.
- Centralized configuration manager (`config.py` loading `.env` with fallback support).
- Structured logging setup (`app/utils/logger.py`) supporting console and rotating file output.
- Abstract base classes (`abc.ABC`) and factory pattern (`ModelFactory`).
- Context engine mode profiles (**Home**, **Road**, **Office**) and sound priority matrix.
- 6-byte binary payload serializer (`HapticPacketSerializer`) for ESP32 BLE haptic alerts.
- FastAPI REST delivery endpoints (`app/api/routes.py`).

### Phase 2: Dataset Management Subsystem (`app/ai/dataset/`)
- Target sound classes: `ambulance`, `car_horn`, `fire_alarm`, `doorbell`, `dog_bark`.
- `DatasetDirectoryManager`: Automatic creation & verification of `dataset/raw/{classes}`, `dataset/processed/`, and `dataset/test_audio/`.
- `AudioDatasetLoader`: Multi-format audio file loader (`.wav`, `.mp3`, `.flac`), SHA-256 content checksum calculation, and `DatasetManifest` construction.
- `DatasetValidator`: Comprehensive checks for missing class directories, empty folders, unsupported extensions, corrupted/0-byte files, duplicates, and invalid filenames.
- `DatasetStatisticsCalculator`: Statistical summaries & JSON report generator (`app/outputs/dataset_stats.json`).
- `DatasetExplorer`: Filtering, search, sample previews, and folder details inspection.

### Phase 3: Audio Preprocessing Subsystem (`app/ai/preprocessing/`)
- **Multi-Format Audio Loader**: `AudioLoader` loading raw PCM audio signals from `.wav`, `.mp3`, and `.flac`.
- **Audio Standardization**: `AudioStandardizer` converting stereo audio to mono, resampling signals to **22,050 Hz**, and normalizing peak amplitude to `[-0.95, 0.95]`.
- **Silence Processing**: `SilenceProcessor` removing leading and trailing silent frames using configurable decibel threshold (`-40.0` dB).
- **Background Noise Reduction**: `NoiseReducer` providing optional background noise reduction filter.
- **Fixed-Length Standardization**: `LengthStandardizer` trimming longer clips and zero-padding shorter clips to exactly **4.0 seconds** (88,200 samples at 22,050 Hz).
- **Batch Processing Pipeline**: `PreprocessingPipeline` recursively processing dataset raw folders to `dataset/processed/{classes}`, preserving folder hierarchy, skipping already processed files, displaying progress, and handling per-file errors gracefully.
- **Metadata Generation**: `MetadataGenerator` exporting structured JSON metadata for preprocessed datasets (`dataset/processed/preprocessed_metadata.json`).

### Phase 4: Feature Extraction Subsystem (`app/ai/feature_extraction/`)
- **Librosa Extraction Engine**: `FeatureExtractor` computing 7 acoustic feature representations: MFCC (40 coefficients), Mel Spectrogram (128 dB bands), Zero Crossing Rate (ZCR), Spectral Centroid, Spectral Bandwidth, Spectral Rolloff, and Chroma STFT (12 bins).
- **Label Encoding**: `LabelEncoder` managing bidirectional mapping between class strings and integer IDs (`0, 1, 2, 3, 4`), persisted to `app/ai/features/class_names.json`.
- **Feature Normalization**: `FeatureNormalizer` computing zero-data-leakage Z-score/MinMax scaling parameters on training data and saving parameters to `app/ai/features/scaler_params.json`.
- **Stratified Dataset Splitter**: `StratifiedDatasetSplitter` performing reproducible 70% Train / 15% Validation / 15% Testing dataset splits using configurable random seed (`random_seed=42`).
- **Feature Storage & Metadata**: `FeatureStorageManager` serializing compressed dataset archives (`dataset_splits.npz`) and exporting extraction reports (`app/ai/features/feature_metadata.json`).
- **Feature Visualizer**: `FeatureVisualizer` generating Mel Spectrogram heatmaps, MFCC heatmaps, and class distribution bar charts (`app/outputs/feature_visualizations/`).
- **Batch Extraction Pipeline**: `FeatureExtractionPipeline` orchestrating automated end-to-end dataset feature extraction.

### Phase 5: AI Model Training Subsystem (`app/ai/training/` & `app/ai/models/`)
- **Deep 2D CNN Architecture**: `CNNSoundClassifier` (111,237 parameters) implementing `BaseSoundClassifier` interface with 3 Conv2D blocks (32, 64, 128 filters), Batch Normalization, Max Pooling, Spatial Dropout, Global Average Pooling, and a 5-unit Softmax classification head.
- **Robust Training Data Loader**: `TrainingDataLoader` verifying feature shapes `(184, 173, 1)`, verifying label bounds against `class_names.json`, detecting NaN/Inf anomalies, and computing balanced class weights ($w_c = N / (K \cdot n_c)$).
- **Full-Lifecycle Model Trainer**: `ModelTrainer` executing mini-batch training with `EarlyStopping` (patience=15), `ModelCheckpoint` tracking validation accuracy, and `ReduceLROnPlateau`.
- **Model Checkpoints**: Persisted `sound_classifier_best.keras` (80% validation accuracy) and `sound_classifier_final.keras`.
- **Training Visualizations**: `TrainingVisualizer` rendering publication-quality accuracy curves, loss curves, and composite comparison plots to `app/outputs/model_training/`.
- **Metadata & History**: Comprehensive reports in `model_metadata.json` and `training_history.json`.
- **End-to-End Orchestrator**: `TrainingPipeline` automating loading, building, training, artifact saving, and inference verification.
- **Factory Registration**: Registered in `ModelFactory` under `"cnn"` and `"sound_classifier"`.

### Phase 6: AI Model Evaluation Subsystem (`app/ai/evaluation/`)
- **Offline Evaluation Pipeline**: Evaluates `sound_classifier_best.keras` against unseen test features from Phase 4 (`dataset_splits.npz`).
- **Standardized Metrics**: `EvaluationMetricsCalculator` computes overall test accuracy (80.00%), test loss (1.2487), per-class precision/recall/F1, and macro/weighted summaries using scikit-learn.
- **Confusion Matrix Analysis**: `ConfusionMatrixGenerator` computes 5x5 raw count matrix and row-normalized (recall) matrix, identifying 100% recall on emergency sirens (`ambulance`, `fire_alarm`, `car_horn`).
- **Granular Prediction Analysis**: `PredictionAnalyzer` exports sample-level predictions and probabilities to `predictions.csv`, computes confidence statistics, and isolates misclassifications.
- **Diagnostic Visualizations**: `EvaluationVisualizer` generates publication-grade plots: `confusion_matrix.png`, `normalized_confusion_matrix.png`, `metrics_comparison.png`, and `confidence_distribution.png`.
- **Reporting & Artifacts**: `EvaluationReportGenerator` saves `classification_report.json`, `evaluation_metrics.json`, and human-readable `evaluation_report.txt` to `app/outputs/model_evaluation/`.

### Phase 7: Real-Time Sound Recognition Subsystem (`app/ai/inference/`)
- **Real-Time Microphone Stream Ingestion**: `MicrophoneAudioCapture` thread-safe, non-blocking stream listener capturing audio via `sounddevice` into a circular rolling buffer of 4.0s (88,200 samples at 22,050 Hz).
- **Audio Hardware Manager**: `AudioDeviceManager` enumerating system input microphones and reporting default devices and sampling specs.
- **Exact Pipeline Reuse**: `InferenceFeaturePipeline` reusing Phase 3 `AudioStandardizer` and Phase 4 `FeatureExtractor` to produce exact `(1, 184, 173, 1)` composite matrices without logic duplication.
- **Cached Inference Model**: `InferenceModelLoader` loading `sound_classifier_best.keras` once, validating tensor dimensions, and warming up execution graphs.
- **Confidence Gating & Temporal Stabilization**: `PredictionStabilizer` validating confidence against configurable threshold (default `0.70`) and confirming events via sliding consensus buffer ($N=3, K=2$).
- **Master Recognizer & CLI**: `RealtimeSoundRecognizer` supporting continuous live streaming, offline WAV testing, latency telemetry, and clean terminal UI (`python -m app.ai.inference.cli`).

### Phase 8: Context-Aware Decision Subsystem (`app/context/`)
- **Strongly Typed Mode Context**: `EnvironmentMode` (`HOME`, `ROAD`, `OFFICE`) and `PriorityLevel` (`HIGH`, `MEDIUM`, `LOW`, `IGNORE`).
- **Configuration-Driven Priority Matrix**: `PriorityEngine` mapping 5 AI classes across 3 modes via configurable dictionary rules ($O(1)$ constant time lookup).
- **Mode State Manager**: `ModeManager` with mode transition validation, safe reset, and observer callback listeners on mode changes.
- **Context Decision Engine**: `ContextDecisionEngine` integrating Phase 7 predictions, confidence gating (reusing 0.70 threshold), and alert policies (`HIGH`/`MEDIUM` $\rightarrow$ Alert, `LOW`/`IGNORE` $\rightarrow$ Suppress).
- **Unified Backward-Compatible Facade**: `ContextManager` maintaining compatibility with Phase 1 while offering full Phase 8 decision capabilities.

### Milestone: Software-Only Integration & Web Prototype (`app/web/` & `app/services/`)
- **Pipeline Integration**: `SoftwareIntegrationService` connects Phase 3 Preprocessing $\rightarrow$ Phase 4 Features $\rightarrow$ Phase 5 CNN Model $\rightarrow$ Phase 7 Recognition $\rightarrow$ Phase 8 Context Engine $\rightarrow$ Frontend Dashboard.
- **Hardware-Independent Operation**: Fully operational without ESP32, INMP441, vibration motor, or Bluetooth. Explicitly reports `Hardware: Not Connected (Software Prototype)`.
- **FastAPI REST API**: Comprehensive endpoints for diagnostics, mode control, test audio WAV evaluation, custom file upload, demo simulation, live microphone streaming, and scenario benchmarking.
- **Interactive Web Dashboard**: Glassmorphic UI with dynamic status badges, synchronized mode switcher, active detection telemetry card, latency breakdowns, and recent alert history table.
- **7 Scenarios Verification**: Automated verification matrix for the 7 core context priority scenarios with 100% pass rate.
- **14 New Integration Tests**: 123 total repository tests passing with zero failures.

---

## Audio Feature Extraction Pipeline

```
Preprocessed 22,050 Hz WAV (dataset/processed/)
        │
        ▼
[Audio Validation & Load] ───────> Verify 1D Array & Finite Signal Values
        │
        ▼
[Feature Extractor Engine] ──────> Extract MFCC, Mel Spectrogram, ZCR, Centroid, Bandwidth, Rolloff, Chroma
        │
        ▼
[LabelEncoder] ──────────────────> Map Class Names -> Integer IDs (class_names.json)
        │
        ▼
[Stratified Dataset Splitter] ───> 70% Training / 15% Validation / 15% Testing (Seed=42)
        │
        ▼
[Feature Normalizer] ────────────> Z-Score Scaling fitted on Training Split (scaler_params.json)
        │
        ▼
[Storage & Metadata Generator] ──> Save dataset_splits.npz & feature_metadata.json to app/ai/features/
        │
        ▼
[Feature Visualizer] ────────────> Export Heatmaps & Bar Charts to app/outputs/feature_visualizations/
```

---

## Target Sound Classes

1. **Ambulance**: Emergency siren audio samples (`ambulance/` -> ID `0`)
2. **Car Horn**: Traffic and automotive horn warnings (`car_horn/` -> ID `1`)
3. **Fire Alarm**: Critical fire alarms and smoke detectors (`fire_alarm/` -> ID `2`)
4. **Doorbell**: Domestic entrance chimes and bells (`doorbell/` -> ID `3`)
5. **Dog Bark**: Domestic animal bark warnings (`dog_bark/` -> ID `4`)

---

## Technologies Used

- **Core Runtime**: Python 3.11
- **Architecture**: Clean Architecture, SOLID Principles, Object-Oriented Design
- **Signal & Feature Processing**: `librosa`, `scipy`, `numpy`
- **Visualization**: `matplotlib`
- **Configuration & Environment**: Dataclasses, `python-dotenv`, `pydantic`
- **Web & API Framework**: FastAPI, Uvicorn
- **BLE Hardware Communication**: Bleak, custom 6-byte binary packet protocol
- **Testing**: Python `unittest`, `pytest`, `pytest-asyncio`

---

## Folder Structure

```
Smart-Haptic-Alert-System/
├── README.md                           # Project overview & progress tracker
├── CHANGELOG.md                        # Release history
├── requirements.txt                    # Production & development dependencies
├── main.py                             # System bootstrap entrypoint
├── config.py                           # Central configuration manager
├── .env.example                        # Environment variables template
├── .gitignore                          # Git exclusion rules
│
├── dataset/                            # Managed Dataset Storage
│   ├── raw/                            # Target raw sound classes
│   │   ├── ambulance/
│   │   ├── car_horn/
│   │   ├── fire_alarm/
│   │   ├── doorbell/
│   │   └── dog_bark/
│   ├── processed/                      # Standardized 22050Hz 4.0s 16-bit WAV storage
│   └── test_audio/                     # Test audio evaluation samples
│
├── app/
│   ├── ai/                             # Machine Learning Subsystem
│   │   ├── dataset/                    # Dataset Management (Loader, Validator, Explorer, Stats)
│   │   ├── preprocessing/              # Audio Preprocessing (Standardizer, Silence, Noise, Length)
│   │   ├── feature_extraction/        # Feature Extractor, Label Encoder, Normalizer, Splitter, Storage, Visualizer
│   │   ├── features/                   # Extracted .npz arrays, class_names.json, scaler_params.json, metadata
│   │   ├── training/                   # Trainer pipeline & model exporter
│   │   ├── inference/                  # Real-time sound inference engine
│   │   ├── models/                     # Base model contracts & ModelFactory
│   │   └── utils/                      # Metrics & confusion matrix helpers
│   │
│   ├── context/                        # Context Prioritization Engine
│   │   ├── config/                     # Home, Road, Office Mode profiles
│   │   └── context_manager.py          # Sound priority evaluator
│   │
│   ├── bluetooth/                      # Hardware Communication Subsystem
│   │   ├── ble_manager.py              # ESP32 BLE client interface
│   │   └── protocol.py                 # Haptic binary packet serializer
│   │
│   ├── api/                            # FastAPI REST Delivery Layer
│   ├── controllers/                    # Request & command dispatchers
│   ├── services/                       # Core business logic services
│   ├── utils/                          # Structured logger & helper utilities
│   ├── tests/                          # Automated test suites
│   ├── outputs/                        # Feature visualizations & model binaries
│   ├── logs/                           # System execution log storage
│   └── docs/                           # Software Architecture & progress specifications
│
└── mobile_app/                         # Mobile Companion App integration guide
```

---

## Next Phase to Implement

**Phase 9: Bluetooth Hardware Communication (`app/bluetooth/`)**
- Implement robust BLE GATT client for ESP32 wearable hardware.
- Connect Phase 8 `DecisionResult` alerts to 6-byte binary payload serialization (`HapticPacketSerializer`).
- Implement automatic reconnection, connection pooling, and queueing for hardware transmissions.
- Prepare hardware alert dispatch for wearable vibration motors.

---

## Running System Verification & Tests

### Launch Web Dashboard Prototype & API
```bash
python main.py
```
Open your browser at:
- **Interactive Web Dashboard**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Execute Real-Time Sound Recognition CLI
```bash
# 1. List input audio recording hardware devices
python -m app.ai.inference.cli --list-devices

# 2. Test offline WAV audio sample
python -m app.ai.inference.cli --test-file dataset/processed/car_horn/sample_0.wav --threshold 0.30

# 3. Start live microphone continuous stream
python -m app.ai.inference.cli --live --threshold 0.70 --hop-sec 1.0
```

### Run Automated Test Suite
```bash
pytest app/tests/ -v
```
