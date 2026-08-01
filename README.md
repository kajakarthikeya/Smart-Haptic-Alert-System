# Smart Haptic Alert System

An AI-powered wearable assistance system for hearing-impaired users that detects important environmental sounds, prioritizes them according to the user's selected mode (**Home**, **Road**, **Office**), and communicates haptic alert signals to an **ESP32 wearable device**.

---

## Project Status & Progress Tracker

- [x] **Phase 1: Project Initialization & Clean Architecture Setup**
- [x] **Phase 2: Dataset Management Subsystem**
- [x] **Phase 3: Audio Preprocessing Subsystem**
- [ ] **Phase 4: Feature Extraction Subsystem** *(Next Phase)*
- [ ] **Phase 5: AI Model Training & Quantization**
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
- **Audio Standardization**: `AudioStandardizer` converting stereo audio to mono, resampling signals to **22,050 Hz**, and normalizing peak amplitude to `[-1.0, 1.0]`.
- **Silence Processing**: `SilenceProcessor` removing leading and trailing silent frames using configurable decibel threshold (`-40.0` dB).
- **Background Noise Reduction**: `NoiseReducer` providing optional background noise reduction filter.
- **Fixed-Length Standardization**: `LengthStandardizer` trimming longer clips and zero-padding shorter clips to exactly **4.0 seconds** (88,200 samples at 22050 Hz).
- **Batch Processing Pipeline**: `PreprocessingPipeline` recursively processing dataset raw folders to `dataset/processed/{classes}`, preserving folder hierarchy, skipping already processed files, displaying progress, and handling per-file errors gracefully.
- **Metadata Generation**: `MetadataGenerator` exporting structured JSON metadata for preprocessed datasets (`dataset/processed/preprocessed_metadata.json`).

---

## Audio Preprocessing Workflow

```
Raw Audio Sample (.wav, .mp3, .flac)
        │
        ▼
[AudioLoader] ───────────────> Unpack PCM Waveform Data
        │
        ▼
[AudioStandardizer] ─────────> Convert to Mono -> Resample to 22,050 Hz -> Peak Normalize
        │
        ▼
[SilenceProcessor] ──────────> Trim Leading & Trailing Silence (-40 dB Threshold)
        │
        ▼
[NoiseReducer] ──────────────> Apply Background Noise Filter (Optional)
        │
        ▼
[LengthStandardizer] ────────> Trim / Zero-Pad to Exact 4.0s (88,200 Samples)
        │
        ▼
[MetadataGenerator & Writer] ─> Save 16-bit PCM WAV to dataset/processed/{class}/ & Export JSON
```

---

## Target Sound Classes

1. **Ambulance**: Emergency siren audio samples (`ambulance/`)
2. **Car Horn**: Traffic and automotive horn warnings (`car_horn/`)
3. **Fire Alarm**: Critical fire alarms and smoke detectors (`fire_alarm/`)
4. **Doorbell**: Domestic entrance chimes and bells (`doorbell/`)
5. **Dog Bark**: Domestic animal bark warnings (`dog_bark/`)

---

## Technologies Used

- **Core Runtime**: Python 3.11
- **Architecture**: Clean Architecture, SOLID Principles, Object-Oriented Design
- **Configuration & Environment**: Dataclasses, `python-dotenv`, `pydantic`
- **Data & Signal Utilities**: Standard library `wave`, `struct`, `math`, `hashlib`, `json`, `dataclasses`
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
│   │   ├── feature_extraction/        # Log-Mel Spectrogram extraction
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
│   ├── outputs/                        # Model binaries & visual artifacts
│   ├── logs/                           # System execution log storage
│   └── docs/                           # Software Architecture & progress specifications
│
└── mobile_app/                         # Mobile Companion App integration guide
```

---

## Next Phase to Implement

**Phase 4: Feature Extraction Subsystem (`app/ai/feature_extraction/`)**
- Transform 4.0s 22050 Hz preprocessed audio files from `dataset/processed/` into Log-Mel Spectrogram feature matrices.
- Configurable FFT window size (`n_fft=512`), hop length (`hop_length=160`), and Mel filterbank bins (`n_mels=64`).
- Save extracted feature tensors for neural network model training.

---

## Running System Verification & Tests

### Execute System Bootstrap
```bash
python main.py
```

### Run Automated Test Suite
```bash
python -m unittest discover -s app/tests
```
