# Changelog - Smart Haptic Alert System

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
