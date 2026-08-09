# Smart Haptic Alert System

An AI-powered wearable assistance system for hearing-impaired users that detects important environmental sounds, prioritizes them according to the user's selected mode (**Home**, **Road**, **Office**), and communicates haptic alert signals to an **ESP32 wearable device**.

---

## Project Status & Progress Tracker

- [x] **Phase 1: Project Initialization & Clean Architecture Setup**
- [x] **Phase 2: Dataset Management Subsystem**
- [x] **Phase 3: Audio Preprocessing Subsystem**
- [x] **Phase 4: Feature Extraction Subsystem**
- [ ] **Phase 5: AI Model Training & Quantization** *(Next Phase)*
- [ ] **Phase 6: Model Evaluation & Benchmarking**
- [ ] **Phase 7: Real-Time Sound Recognition Engine**
- [ ] **Phase 8: Context-Aware Decision Engine**
- [ ] **Phase 9: Bluetooth Hardware Communication**
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

**Phase 5: AI Model Training & Quantization (`app/ai/training/` & `app/ai/models/`)**
- Load extracted dataset splits (`dataset_splits.npz`), class mappings (`class_names.json`), and scaler parameters (`scaler_params.json`).
- Build and train deep Convolutional Neural Network (CNN) architecture for environmental sound classification across target classes.
- Export trained model weights and TFLite quantized model binaries for edge device deployment.

---

## Running System Verification & Tests

### Execute System Bootstrap
```bash
python main.py
```

### Run Feature Extraction Batch Pipeline
```python
from app.ai.feature_extraction import FeatureExtractionPipeline
pipeline = FeatureExtractionPipeline()
summary = pipeline.run(generate_visualizations=True)
```

### Run Automated Test Suite
```bash
python -m unittest discover -s app/tests
```
