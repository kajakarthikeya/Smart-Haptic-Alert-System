# Changelog - Smart Haptic Alert System

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.0] - 2026-08-01

### Added (Phase 2: Dataset Management Subsystem)
- **Target Sound Classes**: Configured target classes: `ambulance`, `car_horn`, `fire_alarm`, `doorbell`, `dog_bark`.
- **Dataset Directory Manager**: `DatasetDirectoryManager` automatically initializing `dataset/raw/{classes}`, `dataset/processed/`, and `dataset/test_audio/`.
- **Dataset Loader**: `AudioDatasetLoader` scanning `.wav`, `.mp3`, and `.flac` files, computing SHA-256 hashes, and building `DatasetManifest`.
- **Dataset Validator**: `DatasetValidator` performing automated checks for missing class folders, empty directories, unsupported file formats, corrupted/0-byte files, duplicate files (SHA-256 matching), and invalid filenames.
- **Dataset Statistics**: `DatasetStatisticsCalculator` computing total files, dataset size in MB, per-class counts, duration min/max/avg, sample rate breakdowns, and exporting JSON reports to `app/outputs/dataset_stats.json`.
- **Dataset Explorer**: `DatasetExplorer` providing search, filtering (by query keyword, class, or duration range), sample previews, and folder summary inspection.
- **Domain Exceptions**: Custom exception hierarchy (`DatasetNotFoundError`, `MissingClassError`, `CorruptedAudioError`, `InvalidDatasetError`).
- **Domain Models**: Value objects (`AudioFileMetadata`, `DatasetItem`, `DatasetManifest`, `ValidationReport`, `DatasetStats`).
- **Documentation & Architecture Docs**: Added `app/docs/Software_Architecture.md`, `app/docs/Development_Progress.md`, `app/docs/Module_Relationship.md`, and updated `README.md` and `app/ai/dataset/README.md`.
- **Automated Test Suite**: Added `app/tests/test_dataset_management.py` with 100% test pass rate (11/11 tests).

### Created Files
- `app/ai/dataset/exceptions.py`
- `app/ai/dataset/models.py`
- `app/ai/dataset/dataset_directory_manager.py`
- `app/ai/dataset/dataset_validator.py`
- `app/ai/dataset/dataset_statistics.py`
- `app/ai/dataset/dataset_explorer.py`
- `app/tests/test_dataset_management.py`
- `app/docs/Software_Architecture.md`
- `app/docs/Development_Progress.md`
- `app/docs/Module_Relationship.md`
- `CHANGELOG.md`

### Modified Files
- `config.py` (added `DatasetConfig` settings)
- `app/ai/dataset/__init__.py` (exported clean public API)
- `app/ai/dataset/dataset_loader.py` (implemented audio metadata reader & scanner)
- `app/ai/dataset/README.md` (expanded public methods and usage workflow)
- `README.md` (updated progress tracker, completed features, folder structure, next phase)

### Known Limitations
- Audio signal resampling and spectrogram feature extraction are not included in this phase (scheduled for Phase 3 & 4).
- Audio header reading currently relies on Python standard library `wave` for `.wav` files with fallback parameters for `.mp3`/`.flac`.

### Next Planned Module
- **Phase 3**: Audio Preprocessing Subsystem (`app/ai/preprocessing/`).

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
- Initial test suite (`app/tests/`).
