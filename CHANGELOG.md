# Changelog - Smart Haptic Alert System

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.8.1] - 2026-09-05

### Added (Milestone: Software-Only Integration & Web Prototype)
- **Central Software Integration Service (`app/services/integration_service.py`)**:
  - Connects the entire Phase 1–8 software pipeline: Audio Input -> Phase 3 Preprocessing -> Phase 4 Feature Extraction -> Phase 5 CNN Model Checkpoint -> Phase 7 RealtimeSoundRecognizer -> Phase 8 ContextDecisionEngine -> ModeManager -> Web Dashboard.
  - Supports sample audio evaluation, custom WAV uploads, live microphone streaming, demo simulation, scenario benchmarking, and in-memory alert history tracking (up to 100 entries).
- **FastAPI REST API Delivery Layer (`app/api/routes.py`)**:
  - `GET /api/v1/system/status`: Diagnostic system status reporting state of AI Model, Inference Engine, Context Engine, Audio Input, and Hardware (explicitly showing `Not Connected - Software Prototype`).
  - `GET /api/v1/mode` & `POST /api/v1/mode`: Real-time operating mode inspection and switching in Phase 8 `ModeManager`.
  - `GET /api/v1/test-audio/samples` & `POST /api/v1/test-audio/evaluate`: Test audio WAV enumeration and full pipeline execution.
  - `POST /api/v1/test-audio/upload`: Multipart WAV upload and immediate inference pass.
  - `POST /api/v1/demo/simulate`: Target sound simulation through genuine Phase 8 Context Engine.
  - `POST /api/v1/recognition/start|stop` & `GET /api/v1/recognition/latest`: Live microphone recognition streaming and telemetry polling.
  - `GET /api/v1/alerts/history` & `POST /api/v1/alerts/clear`: Alert event history inspection and clearing.
  - `GET /api/v1/scenarios/run`: Execution of the 7 mandatory benchmark scenarios.
- **Hardware-Independent Web Frontend Prototype (`app/web/`)**:
  - `index.html`: Responsive single-page dashboard with glassmorphism UI, diagnostic status badges, synchronized mode switcher, active detection telemetry card with latency metrics and confidence meter, interactive tabs (Test Audio, Demo Mode, Live Microphone, 7 Scenarios Matrix), and recent alerts table.
  - `style.css`: Modern styling with Outfit/JetBrains Mono typography, dark mode theme, pulsing status indicators, and glassmorphic cards.
  - `app.js`: Reactive vanilla client communicating with `/api/v1` endpoints.
  - `README.md`: Comprehensive frontend and integration documentation.
- **Test Audio Sample Directory (`dataset/test_audio/`)**:
  - Populated with 5 verified PCM WAV audio files (`ambulance.wav`, `car_horn.wav`, `fire_alarm.wav`, `doorbell.wav`, `dog_bark.wav`).
- **Automated Integration Test Suite (`app/tests/test_software_integration.py`)**:
  - Added 14 unit and end-to-end integration tests (14/14 passing; 123/123 total repository tests passing).
- **Documentation**:
  - Created `app/docs/API_Documentation.md` and `app/web/README.md`.
  - Updated `Software_Architecture.md`, `Development_Progress.md`, and `Module_Relationships.md`.

### Next Planned Module
- **Phase 9**: Bluetooth Hardware Communication (`app/bluetooth/`).

---

## [0.8.0] - 2026-09-05

### Added (Phase 8: Context-Aware Decision Subsystem)
- **Modular Context-Aware Decision Engine**: Created dedicated `app/context/` architecture:
  - `enums.py`: Defines strongly typed `EnvironmentMode` (`HOME`, `ROAD`, `OFFICE`), `PriorityLevel` (`HIGH`, `MEDIUM`, `LOW`, `IGNORE`), and legacy alias `SoundPriority`.
  - `models.py`: Defines `SoundPrediction` (with Phase 7 `from_prediction_result` converter and strict confidence validation), `AlertPolicy`, and structured `DecisionResult`.
  - `rules.py`: Configuration-driven `DEFAULT_PRIORITY_MATRIX` covering all 15 sound/mode permutations with schema validation.
  - `priority_engine.py`: `PriorityEngine` validating sounds against target classes and modes, performing $O(1)$ constant-time matrix lookups.
  - `mode_manager.py`: `ModeManager` managing operating mode state, validating mode transitions, notifying registered observer callbacks on transitions, and supporting safe resets.
  - `decision_engine.py`: `ContextDecisionEngine` combining AI sound predictions, mode context, confidence threshold gating (reusing 0.70 default), and alert policies to generate human-readable decision reasons.
  - `context_manager.py`: Refactored `ContextManager` facade providing unified high-level access to the decision engine while preserving full backward compatibility for Phase 1 code.
  - `exceptions.py`: Comprehensive domain exception hierarchy (`ContextError`, `InvalidModeError`, `UnknownSoundError`, `InvalidConfidenceError`, `PriorityRuleError`, `ConfigurationError`).
  - `README.md`: Subsystem architecture and integration documentation.
- **Automated Test Suite**: Added `app/tests/test_context_decision.py` with 41 unit and integration tests (109/109 total system tests passing across all packages).
- **Subsystem Architecture Docs**: Added `app/docs/Context_Aware_Decision_Module.md` and updated `Software_Architecture.md`, `Development_Progress.md`, and `Module_Relationships.md`.

### Created Files
- `app/context/enums.py`
- `app/context/models.py`
- `app/context/rules.py`
- `app/context/priority_engine.py`
- `app/context/mode_manager.py`
- `app/context/decision_engine.py`
- `app/context/exceptions.py`
- `app/context/README.md`
- `app/docs/Context_Aware_Decision_Module.md`
- `app/tests/test_context_decision.py`

### Modified Files
- `config.py` (added `ContextConfig` settings registered in `AppSettings`)
- `app/context/config/__init__.py` (exported `ContextConfig` and `get_context_config`)
- `app/context/context_manager.py` (refactored facade with Phase 8 & Phase 1 support)
- `app/context/__init__.py` (clean public interface exports)
- `app/docs/Software_Architecture.md` (updated context subsystem description)
- `app/docs/Development_Progress.md` (marked Phase 8 completed, Phase 9 next)
- `app/docs/Module_Relationships.md` (updated Phase 8 contracts and Phase 9 BLE handshakes)
- `README.md` (updated progress tracker, completed subsystems, and Phase 8 documentation)
- `CHANGELOG.md`

### Next Planned Module
- **Phase 9**: Bluetooth Hardware Communication (`app/bluetooth/`).

---

## [0.7.0] - 2026-09-05

### Added (Phase 7: Real-Time Sound Recognition Subsystem)
- **Modular Real-Time Recognition Engine**: Created dedicated, single-responsibility `app/ai/inference/` package:
  - `AudioDeviceManager`: Queries host OS audio recording hardware devices, identifies system default microphone, maximum input channels, and native sample rates.
  - `MicrophoneAudioCapture`: Implements thread-safe, non-blocking microphone stream capture using `sounddevice.InputStream` and a circular rolling ring buffer (stores latest 4.0s = 88,200 samples at 22,050 Hz).
  - `InferenceFeaturePipeline`: Reuses Phase 3 `AudioStandardizer` (mono conversion, 22,050 Hz resampling, peak normalization) and `LengthStandardizer` (4.0s trimming/padding), and Phase 4 `FeatureExtractor` (`extract_composite_matrix`), yielding identical `(1, 184, 173, 1)` feature tensors without duplicated logic.
  - `InferenceModelLoader`: Caches trained CNN model (`sound_classifier_best.keras`), validates input/output dimensions against `training_metadata.json` and `class_mapping.json`, and runs a warmup zero-tensor forward pass.
  - `PredictionStabilizer`: Implements temporal consensus agreement buffer ($N=3, K=2$) and confidence gating (default `>= 0.70`), yielding categorized prediction statuses (`CONFIRMED`, `TENTATIVE`, `LOW_CONFIDENCE`).
  - `RealtimeSoundRecognizer`: Master inference engine supporting `recognize_file()`, `recognize_window()`, and `start_streaming()` with non-blocking microphone capture, window hopping, callback dispatches, and latency telemetry.
  - `cli.py`: Interactive command-line interface with `--list-devices`, `--test-file`, and `--live` real-time microphone streaming with graceful Ctrl+C session summaries.
  - `exceptions.py`: Real-time inference domain exception hierarchy (`InferenceError`, `AudioCaptureError`, `MicrophoneInitializationError`, `DeviceNotFoundError`, `RealtimeInferenceError`, `FeaturePipelineError`, `ModelLoadingError`, `PredictionError`, `ConfigurationError`).
- **End-to-End Latency Profile**: Benchmark verified under 150 ms total pipeline latency per window (well within 1.0s hop interval).
- **Automated Test Suite**: Added `app/tests/test_realtime_recognition.py` with 14 unit tests passing (68/68 total system unit tests passing across all packages).
- **Subsystem Architecture Docs**: Added `app/docs/Real_Time_Sound_Recognition.md` and updated `app/ai/inference/README.md`, `Software_Architecture.md`, `Development_Progress.md`, and `Module_Relationships.md`.

### Created Files
- `app/ai/inference/__init__.py`
- `app/ai/inference/config.py`
- `app/ai/inference/exceptions.py`
- `app/ai/inference/audio_capture.py`
- `app/ai/inference/feature_pipeline.py`
- `app/ai/inference/model_loader.py`
- `app/ai/inference/prediction.py`
- `app/ai/inference/realtime_recognizer.py`
- `app/ai/inference/cli.py`
- `app/ai/inference/README.md`
- `app/tests/test_realtime_recognition.py`
- `app/docs/Real_Time_Sound_Recognition.md`

### Modified Files
- `requirements.txt` (added `sounddevice>=0.5.0` and `scikit-learn>=1.3.0`)
- `config.py` (added `InferenceConfig` settings)
- `README.md` (updated progress tracker, architecture diagram, and Phase 7 documentation)
- `app/docs/Software_Architecture.md` (updated inference subsystem description)
- `app/docs/Development_Progress.md` (marked Phase 7 completed, Phase 8 next)
- `app/docs/Module_Relationships.md` (updated Phase 7 contracts and Phase 8 interfaces)
- `CHANGELOG.md`

### Next Planned Module
- **Phase 8**: Context-Aware Decision Engine (`app/context/`).

---

## [0.6.0] - 2026-09-05

### Added (Phase 6: AI Model Evaluation Subsystem)
- **Modular Evaluation Engine**: Created dedicated, single-responsibility `app/ai/evaluation/` package:
  - `EvaluationDataLoader`: Loads and validates test dataset splits (`dataset_splits.npz`), class mappings (`class_names.json`), and trained model checkpoint (`sound_classifier_best.keras`).
  - `EvaluationMetricsCalculator`: Computes categorical cross-entropy loss, overall test accuracy, per-class precision, recall, F1-score, support, One-vs-Rest accuracy, and macro/weighted summaries using scikit-learn.
  - `ConfusionMatrixGenerator`: Computes 5x5 raw count confusion matrix and row-normalized (recall-based) matrix.
  - `PredictionAnalyzer`: Generates sample-level prediction records (`SamplePrediction` / `PredictionRecord`), computes confidence statistics across correct/incorrect samples, ranks error pairs, filters low-confidence predictions (< 0.50 threshold), and exports `predictions.csv`.
  - `EvaluationVisualizer`: Headless matplotlib visualizer generating publication-quality figures: `confusion_matrix.png`, `normalized_confusion_matrix.png`, `metrics_comparison.png`, and `confidence_distribution.png`.
  - `EvaluationReportGenerator`: Exports standardized `classification_report.json`, comprehensive `evaluation_metrics.json`, and human-readable executive diagnostic report `evaluation_report.txt`.
  - `ModelEvaluator`: Master pipeline orchestrator coordinating the complete evaluation workflow.
  - `exceptions.py`: Evaluation domain exception hierarchy (`EvaluationError`, `InvalidEvaluationData`, `ModelLoadError`, `PredictionError`, `MetricCalculationError`, `ReportGenerationError`, `VisualizationError`).
- **Benchmark Results on Unseen Test Dataset**:
  - Overall Test Accuracy: **80.00%** (4 / 5 correct).
  - Categorical Cross-Entropy Loss: **1.2487**.
  - Emergency Siren Recall: **100.0%** for `ambulance`, `fire_alarm`, and `car_horn`.
- **Automated Test Suite**: Added `app/tests/test_model_evaluation.py` with 12 unit tests passing (54/54 total system unit tests passing across all packages).
- **Subsystem Architecture Docs**: Added `app/docs/AI_Model_Evaluation.md` and updated `app/ai/evaluation/README.md`, `Software_Architecture.md`, `Development_Progress.md`, and `Module_Relationships.md`.

### Created Files
- `app/ai/evaluation/__init__.py`
- `app/ai/evaluation/data_loader.py`
- `app/ai/evaluation/metrics.py`
- `app/ai/evaluation/confusion_matrix.py`
- `app/ai/evaluation/prediction_analyzer.py`
- `app/ai/evaluation/visualization.py`
- `app/ai/evaluation/report_generator.py`
- `app/ai/evaluation/evaluator.py`
- `app/ai/evaluation/exceptions.py`
- `app/ai/evaluation/README.md`
- `app/tests/test_model_evaluation.py`
- `app/docs/AI_Model_Evaluation.md`

### Modified Files
- `config.py` (added `EvaluationConfig` settings and `Config = settings` alias)
- `README.md` (updated master progress tracker and evaluation documentation)
- `app/docs/Software_Architecture.md` (updated architecture documentation)
- `app/docs/Development_Progress.md` (marked Phase 6 completed, Phase 7 next)
- `app/docs/Module_Relationships.md` (updated handshakes between Phase 5, Phase 6, and Phase 7)
- `CHANGELOG.md`

### Next Planned Module
- **Phase 7**: Real-Time Sound Recognition Engine (`app/ai/recognition/`).

---

## [0.5.0] - 2026-09-05

### Added (Phase 5: AI Model Training Subsystem)
- **CNN Architecture**: `CNNSoundClassifier` 2D Convolutional Neural Network implementing `BaseSoundClassifier` interface with 3 convolutional blocks (32, 64, 128 filters), Batch Normalization, ReLU activations, Max Pooling, Spatial Dropout, Global Average Pooling, and a 5-unit Softmax classification head (111,237 total trainable parameters).
- **Training Data Loader**: `TrainingDataLoader` validating acoustic feature matrix dimensions (`184 x 173 x 1`), verifying label counts and bounds against `class_names.json`, detecting NaN and infinite anomalies, and computing balanced class weights ($w_c = N / (K \cdot n_c)$).
- **Model Trainer**: `ModelTrainer` subclassing `BaseTrainer` with validation loops, `EarlyStopping` (patience=15), `ModelCheckpoint` tracking `val_accuracy`, and `ReduceLROnPlateau` (factor=0.5, patience=7).
- **Checkpoint Serialization**: Automated persistence of `sound_classifier_best.keras` (best validation checkpoint) and `sound_classifier_final.keras` (final epoch).
- **History & Metadata Persistence**: Exported `training_history.json` and comprehensive `model_metadata.json` capturing dataset splits, hyperparameters, class mappings, parameter counts, and framework info.
- **Training Visualizer**: `TrainingVisualizer` generating publication-quality accuracy progression curves (`accuracy_curve.png`), loss progression curves (`loss_curve.png`), and combined comparison figures (`training_metrics.png`) into `app/outputs/model_training/`.
- **Training Pipeline**: `TrainingPipeline` orchestrating end-to-end data loading, CNN compilation, training execution, artifact saving, and post-training single-sample inference verification.
- **Model Factory Registration**: Registered `CNNSoundClassifier` in `ModelFactory` under keys `"cnn"` and `"sound_classifier"`.
- **Domain Exceptions**: Added `InvalidFeatureData`, `InvalidLabelData`, `ModelTrainingError`, `ModelSaveError`, and `ConfigurationError` to `app/ai/training/exceptions.py`, and model exceptions to `app/ai/models/exceptions.py`.
- **Automated Test Suite**: Added `app/tests/test_model_training.py` with 15 unit tests passing (42 total system unit tests passing across all packages).
- **Subsystem Architecture Docs**: Added `app/docs/AI_Model_Training.md` and updated `app/ai/training/README.md`, `app/ai/models/README.md`, `Software_Architecture.md`, `Development_Progress.md`, and `Module_Relationships.md`.

### Created Files
- `app/ai/models/cnn_classifier.py`
- `app/ai/models/exceptions.py`
- `app/ai/training/data_loader.py`
- `app/ai/training/exceptions.py`
- `app/ai/training/pipeline.py`
- `app/ai/training/visualizer.py`
- `app/tests/test_model_training.py`
- `app/docs/AI_Model_Training.md`

### Modified Files
- `config.py` (added `TrainingConfig` and `ModelConfig` settings)
- `requirements.txt` (enabled `torch>=2.0.0`, `keras>=3.0.0`, and `matplotlib>=3.7.0`)
- `app/ai/models/__init__.py` (exported `CNNSoundClassifier` and model exceptions)
- `app/ai/models/model_factory.py` (registered `CNNSoundClassifier`)
- `app/ai/training/__init__.py` (exported public training API)
- `app/ai/training/trainer.py` (implemented complete `ModelTrainer` lifecycle)
- `app/ai/training/README.md` (updated public documentation)
- `app/ai/models/README.md` (updated model architecture documentation)
- `README.md` (updated master progress tracker and architecture specs)
- `app/docs/Software_Architecture.md` (updated architecture documentation)
- `app/docs/Development_Progress.md` (updated phase roadmap matrix and logs)
- `app/docs/Module_Relationships.md` (updated handshakes between Phase 4, Phase 5, Phase 6, and Phase 7)
- `CHANGELOG.md`

### Next Planned Module
- **Phase 6**: Model Evaluation & Benchmarking Subsystem (`app/ai/evaluation/`).

---

## [0.4.0] - 2026-08-09

### Added (Phase 4: Feature Extraction Subsystem)
- **Feature Extractor Engine**: `FeatureExtractor` computing 7 acoustic feature representations using **Librosa**: MFCC (40 coefficients), Mel Spectrogram (128 dB bands), Zero Crossing Rate (ZCR), Spectral Centroid, Spectral Bandwidth, Spectral Rolloff, and Chroma STFT (12 bins). Supports stacked 2D composite matrices (`[184, 173]`) and 1D summary vectors (`[368]`).
- **Label Encoder**: `LabelEncoder` implementing bidirectional mapping between sound class strings (`ambulance`, `car_horn`, `fire_alarm`, `doorbell`, `dog_bark`) and integer IDs (`0, 1, 2, 3, 4`), persisting to `app/ai/features/class_names.json`.
- **Feature Normalizer**: `FeatureNormalizer` computing Z-score and Min-Max scaling parameters on training data without data leakage, saving scaler parameters to `app/ai/features/scaler_params.json`.
- **Stratified Dataset Splitter**: `StratifiedDatasetSplitter` performing reproducible 70% Train / 15% Validation / 15% Testing dataset splits using configurable random seed (`random_seed=42`).
- **Storage & Metadata Management**: `FeatureStorageManager` managing compressed dataset archive `.npz` files and exporting comprehensive metadata reports to `app/ai/features/feature_metadata.json`.
- **Visualization Generator**: `FeatureVisualizer` rendering Mel Spectrogram heatmaps, MFCC heatmaps, and class distribution bar charts to `app/outputs/feature_visualizations/`.
- **Batch Processing Pipeline**: `FeatureExtractionPipeline` automatically orchestrating dataset feature extraction, validation, splitting, normalization, and file storage.
- **Domain Exceptions**: Custom exceptions `FeatureExtractionError`, `InvalidFeatureError`, `FeatureShapeError`, `LabelEncodingError`, and `FeatureStorageError`.
- **Automated Test Suite**: Added `app/tests/test_feature_extraction.py` (27 total system unit tests passing).
- **Subsystem Architecture Docs**: Added `app/docs/Feature_Extraction.md` and updated `README.md`, `app/ai/feature_extraction/README.md`, `Software_Architecture.md`, `Development_Progress.md`, `Module_Relationship.md`, and `Module_Relationships.md`.

### Created Files
- `app/ai/feature_extraction/exceptions.py`
- `app/ai/feature_extraction/label_encoder.py`
- `app/ai/feature_extraction/normalizer.py`
- `app/ai/feature_extraction/feature_extractor.py`
- `app/ai/feature_extraction/dataset_splitter.py`
- `app/ai/feature_extraction/storage.py`
- `app/ai/feature_extraction/visualizer.py`
- `app/ai/feature_extraction/pipeline.py`
- `app/tests/test_feature_extraction.py`
- `app/docs/Feature_Extraction.md`
- `app/docs/Module_Relationships.md`

### Modified Files
- `config.py` (added `FeatureExtractionConfig` settings and target paths)
- `app/ai/feature_extraction/__init__.py` (exported public API)
- `app/ai/feature_extraction/spectrogram_extractor.py` (integrated Librosa feature extraction backend)
- `app/ai/feature_extraction/README.md` (expanded public documentation and usage instructions)
- `README.md` (updated progress tracker, completed features, workflow diagram, and next phase)
- `app/docs/Software_Architecture.md` (updated architecture specs)
- `app/docs/Development_Progress.md` (updated phase roadmap matrix and logs)
- `app/docs/Module_Relationship.md` (updated module handshakes)
- `CHANGELOG.md`

### Next Planned Module
- **Phase 5**: AI Model Training & Quantization Subsystem (`app/ai/training/` & `app/ai/models/`).

---

## [0.3.0] - 2026-08-01

### Added (Phase 3: Audio Preprocessing Subsystem)
- **Audio Loader**: `AudioLoader` loading raw PCM waveform signals from `.wav`, `.mp3`, and `.flac` files into memory arrays.
- **Audio Standardization**: `AudioStandardizer` executing stereo to mono conversion, resampling audio to **22,050 Hz**, and normalizing peak amplitude to `[-0.95, 0.95]`.
- **Silence Processing**: `SilenceProcessor` trimming leading and trailing silent frames using configurable decibel threshold (`-40.0` dB).
- **Background Noise Reduction**: `NoiseReducer` providing optional background noise reduction filter with configurable enable/disable toggle.
- **Fixed-Length Standardization**: `LengthStandardizer` trimming longer clips and zero-padding shorter clips to exact **4.0 seconds (88,200 samples at 22,050 Hz)**.
- **Batch Processing Pipeline**: `PreprocessingPipeline` recursively processing dataset raw folders to `dataset/processed/{classes}`, maintaining raw dataset folder tree structure, skipping already processed files, displaying progress, and handling per-file errors gracefully.
- **Metadata Generation**: `MetadataGenerator` exporting summary metadata JSON reports (`dataset/processed/preprocessed_metadata.json`).
- **Domain Exceptions & Data Models**: Added `PreprocessingError`, `AudioLoadError`, `UnsupportedFormatError`, `ProcessingError`, `CorruptedAudioError`, `RawAudioData`, `ProcessedAudioSignal`, `ProcessedFileMetadata`, and `BatchPreprocessingSummary`.
- **Automated Test Suite**: Added `app/tests/test_audio_preprocessing.py` (17/17 system tests passing).
- **Subsystem Architecture Docs**: Added `app/docs/Audio_Preprocessing.md` and updated `README.md`, `app/ai/preprocessing/README.md`, `Software_Architecture.md`, `Development_Progress.md`, and `Module_Relationship.md`.

### Created Files
- `app/ai/preprocessing/exceptions.py`
- `app/ai/preprocessing/models.py`
- `app/ai/preprocessing/audio_loader.py`
- `app/ai/preprocessing/audio_standardizer.py`
- `app/ai/preprocessing/silence_processor.py`
- `app/ai/preprocessing/noise_reducer.py`
- `app/ai/preprocessing/length_standardizer.py`
- `app/ai/preprocessing/metadata_generator.py`
- `app/ai/preprocessing/preprocessing_pipeline.py`
- `app/tests/test_audio_preprocessing.py`
- `app/docs/Audio_Preprocessing.md`

### Modified Files
- `config.py` (added `PreprocessingConfig` settings)
- `app/ai/preprocessing/__init__.py` (exported clean public API)
- `app/ai/preprocessing/audio_preprocessor.py` (integrated audio preprocessing standardizers)
- `app/ai/preprocessing/README.md` (expanded public methods, configuration options, and pipeline workflow)
- `README.md` (updated progress tracker, completed features, workflow diagram, and next phase)
- `app/docs/Software_Architecture.md` (updated architecture specs)
- `app/docs/Development_Progress.md` (updated phase roadmap matrix)
- `app/docs/Module_Relationship.md` (updated module handshakes)
- `CHANGELOG.md`

### Next Planned Module
- **Phase 4**: Feature Extraction Subsystem (`app/ai/feature_extraction/`).

---

## [0.2.0] - 2026-08-01

### Added (Phase 2: Dataset Management Subsystem)
- **Target Sound Classes**: Configured target classes: `ambulance`, `car_horn`, `fire_alarm`, `doorbell`, `dog_bark`.
- **Dataset Directory Manager**: `DatasetDirectoryManager` automatically initializing `dataset/raw/{classes}`, `dataset/processed/`, and `dataset/test_audio/`.
- **Dataset Loader**: `AudioDatasetLoader` scanning `.wav`, `.mp3`, and `.flac` files, computing SHA-256 hashes, and building `DatasetManifest`.
- **Dataset Validator**: `DatasetValidator` performing automated checks for missing class folders, empty directories, unsupported file formats, corrupted/0-byte files, duplicate files (SHA-256 matching), and invalid filenames.
- **Dataset Statistics**: `DatasetStatisticsCalculator` computing total files, dataset size in MB, per-class counts, duration min/max/avg, sample rate breakdowns, and exporting JSON reports to `app/outputs/dataset_stats.json`.
- **Dataset Explorer**: `DatasetExplorer` providing search, filtering (by query keyword, class, or duration range), sample previews, and folder summary inspection.

---

## [0.1.0] - 2026-08-01

### Added (Phase 1: Project Initialization)
- Clean Architecture directory structure and package setup.
- Centralized configuration manager (`config.py`) loading `.env` with fallback support.
- Centralized structured logging (`app/utils/logger.py`) supporting console and rotating file output.
- Abstract Base Classes for AI models (`BaseSoundClassifier`) and model factory (`ModelFactory`).
- Environmental context prioritization engine (`ContextManager`) with **Home**, **Road**, and **Office** profiles.
- ESP32 BLE client interface (`ESP32BLEManager`) and 6-byte binary payload serializer (`HapticPacketSerializer`).
- FastAPI REST delivery endpoints (`app/api/routes.py`), controllers, services, and system entrypoint (`main.py`).
