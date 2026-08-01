# Smart Haptic Alert System

An AI-powered wearable assistance system for hearing-impaired users that detects important environmental sounds, prioritizes them according to the user's selected mode (**Home**, **Road**, **Office**), and communicates haptic alert signals to an **ESP32 wearable device**.

---

## Project Status & Progress Tracker

- [x] **Phase 1: Project Initialization & Clean Architecture Setup**
- [x] **Phase 2: Dataset Management Subsystem**
- [ ] **Phase 3: Audio Preprocessing Subsystem** *(Next Phase)*
- [ ] **Phase 4: Feature Extraction Subsystem**
- [ ] **Phase 5: AI Model Training & Quantization**
- [ ] **Phase 6: Model Evaluation & Benchmarking**
- [ ] **Phase 7: Real-Time Sound Recognition Engine**
- [ ] **Phase 8: Context-Aware Decision Engine**
- [ ] **Phase 9: Bluetooth Hardware Communication**
- [ ] **Phase 10: Mobile Companion Application**
- [ ] **Phase 11: End-to-End System Integration**
- [ ] **Phase 12: Field Testing & Verification**

---

## Current Completed Phase: Phase 2 - Dataset Management

### Completed Features

#### Phase 1: Architecture & Bootstrap
- Clean Architecture project layout with strict layer separation.
- Centralized configuration manager (`config.py` loading `.env` with fallback support).
- Structured logging setup (`app/utils/logger.py`) supporting console and rotating file output.
- Abstract base classes (`abc.ABC`) and factory pattern (`ModelFactory`).
- Context engine mode profiles (**Home**, **Road**, **Office**) and sound priority matrix.
- 6-byte binary payload serializer (`HapticPacketSerializer`) for ESP32 BLE haptic alerts.
- FastAPI REST delivery endpoints (`app/api/routes.py`).

#### Phase 2: Dataset Management Subsystem
- **Automated Directory Management**: `DatasetDirectoryManager` automatically creates and verifies dataset folder hierarchies:
  - `dataset/raw/ambulance/`
  - `dataset/raw/car_horn/`
  - `dataset/raw/fire_alarm/`
  - `dataset/raw/doorbell/`
  - `dataset/raw/dog_bark/`
  - `dataset/processed/`
  - `dataset/test_audio/`
- **Multi-Format Audio Loader**: `AudioDatasetLoader` scans `.wav`, `.mp3`, and `.flac` files, extracts audio parameters (duration, sample rate, channels), computes SHA-256 content hashes, and returns a structured `DatasetManifest`.
- **Dataset Validator**: `DatasetValidator` performs automated checks for missing class directories, empty folders, unsupported extensions, corrupted/0-byte audio files, duplicate contents (SHA-256 matching), and invalid filenames.
- **Dataset Statistics Calculator**: `DatasetStatisticsCalculator` calculates aggregate file counts, dataset size in MB, per-class counts, duration min/max/avg, sample rate breakdowns, and exports JSON reports.
- **Dataset Explorer**: `DatasetExplorer` provides search, filtering (by query keyword, class, or duration range), sample previews, and folder summary inspection.
- **Domain Exceptions & Data Models**: Custom exception hierarchy (`DatasetNotFoundError`, `MissingClassError`, `CorruptedAudioError`, `InvalidDatasetError`) and dataclass models (`AudioFileMetadata`, `DatasetItem`, `DatasetManifest`, `ValidationReport`, `DatasetStats`).

---

## Target Sound Classes

1. **Ambulance**: Siren detection for road safety (`ambulance/`)
2. **Car Horn**: Vehicle horn warnings for outdoor awareness (`car_horn/`)
3. **Fire Alarm**: Emergency fire alarms and smoke detectors (`fire_alarm/`)
4. **Doorbell**: Domestic entrance chimes and bells (`doorbell/`)
5. **Dog Bark**: Domestic animal alert events (`dog_bark/`)

---

## Technologies Used

- **Core Runtime**: Python 3.11
- **Architecture**: Clean Architecture, SOLID Principles, Object-Oriented Design
- **Configuration & Environment**: Dataclasses, `python-dotenv`, `pydantic`
- **Data & Signal Utilities**: Python standard library (`wave`, `hashlib`, `struct`, `json`, `dataclasses`)
- **Web & API Framework**: FastAPI, Uvicorn
- **BLE Hardware Communication**: Bleak, custom binary packet protocol
- **Testing**: Python `unittest`, `pytest`, `pytest-asyncio`

---

## Folder Structure

```
Smart-Haptic-Alert-System/
├── README.md                           # Project overview & progress tracker
├── CHANGELOG.md                        # Version release history
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
│   ├── processed/                      # Preprocessed output storage
│   └── test_audio/                     # Test audio evaluation samples
│
├── app/
│   ├── ai/                             # Machine Learning Subsystem
│   │   ├── dataset/                    # Dataset Management (Loader, Validator, Explorer, Stats)
│   │   ├── preprocessing/              # Audio signal normalization & framing
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

**Phase 3: Audio Preprocessing Subsystem (`app/ai/preprocessing/`)**
- Implement raw PCM waveform loading.
- Resampling signals to standardized 16,000 Hz sample rate.
- Peak amplitude normalization to [-1.0, 1.0].
- Fixed 1.0-second framing (padding / truncation).
- Batch preprocessed file saving to `dataset/processed/`.

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
