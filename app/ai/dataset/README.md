# Dataset Management Subsystem (`app/ai/dataset/`)

## Purpose
The **Dataset Management Subsystem** provides a clean, robust, object-oriented framework for initializing, loading, validating, analyzing, and exploring environmental sound datasets for the **Smart Haptic Alert System**.

> [!IMPORTANT]
> This module manages and validates dataset integrity ONLY. It does **not** perform signal preprocessing, spectrogram feature extraction, or neural network training.

---

## Target Sound Classes

1. **Ambulance**: Emergency vehicle sirens (`ambulance/`)
2. **Car Horn**: Traffic and vehicle horn sounds (`car_horn/`)
3. **Fire Alarm**: Critical fire alarms and smoke detectors (`fire_alarm/`)
4. **Doorbell**: Domestic chimes and entrance bells (`doorbell/`)
5. **Dog Bark**: Domestic animal warnings (`dog_bark/`)

---

## Architecture & Directory Layout

```
dataset/                                # Managed Dataset Root (configured via config.py)
├── raw/                                # Raw un-preprocessed audio samples
│   ├── ambulance/
│   ├── car_horn/
│   ├── fire_alarm/
│   ├── doorbell/
│   └── dog_bark/
├── processed/                          # Destination for preprocessed spectrogram tensors
└── test_audio/                         # Independent evaluation samples

app/ai/dataset/                         # Module Implementation
├── __init__.py                         # Public API exports
├── exceptions.py                       # Custom domain exceptions
├── models.py                           # Dataclasses & value objects (DatasetManifest, ValidationReport)
├── dataset_directory_manager.py        # Automatic folder hierarchy creation
├── dataset_loader.py                   # Audio file scanner & DatasetManifest builder
├── dataset_validator.py                # Integrity, missing class, corruption & duplicate checks
├── dataset_statistics.py               # Summary statistical calculator & JSON exporter
├── dataset_explorer.py                # Querying, filtering, sample preview & search tool
└── README.md                           # Documentation
```

---

## Key Classes & Responsibilities

| Class | Description |
| :--- | :--- |
| `DatasetDirectoryManager` | Ensures automatic creation and verification of `dataset/raw/{classes}`, `dataset/processed/`, and `dataset/test_audio/`. |
| `AudioDatasetLoader` | Scans audio files (`.wav`, `.mp3`, `.flac`), extracts metadata (duration, sample rate, channels, SHA-256 hash), and constructs a `DatasetManifest`. |
| `DatasetValidator` | Checks for missing class folders, empty directories, unsupported file formats, corrupted/0-byte files, duplicate contents, and invalid filenames. |
| `DatasetStatisticsCalculator` | Computes aggregate total files, size in MB, per-class distribution, duration statistics (min/max/mean), sample rate breakdown, and exports JSON reports. |
| `DatasetExplorer` | Provides utilities to count files, query class distributions, preview sample items, get folder details, and search files by label or duration range. |

---

## Custom Exception Hierarchy

- `DatasetError` (Base domain exception)
  - `DatasetNotFoundError`: Raised when target directory or file does not exist.
  - `MissingClassError`: Raised when required sound class directories are missing.
  - `CorruptedAudioError`: Raised when audio file is unreadable, 0 bytes, or has invalid headers.
  - `InvalidDatasetError`: Raised when dataset validation fails critical requirements.

---

## Usage & Workflow Example

```python
from app.ai.dataset import (
    DatasetDirectoryManager,
    AudioDatasetLoader,
    DatasetValidator,
    DatasetStatisticsCalculator,
    DatasetExplorer,
)

# 1. Initialize dataset directory structure
dir_mgr = DatasetDirectoryManager()
dir_mgr.initialize_directories()

# 2. Validate raw dataset directory
validator = DatasetValidator()
report = validator.validate_directory(dir_mgr.raw_dir)
print(f"Dataset Valid: {report.is_valid}")

# 3. Load dataset manifest
loader = AudioDatasetLoader()
manifest = loader.load_dataset(dir_mgr.raw_dir)

# 4. Calculate & export statistics
stats_calc = DatasetStatisticsCalculator()
stats = stats_calc.compute_statistics(manifest)
stats_calc.export_json_report(stats, "dataset_stats.json")

# 5. Explore dataset
explorer = DatasetExplorer(manifest)
print("Classes present:", explorer.list_classes())
print("Doorbell sample count:", explorer.count_files("doorbell"))
doorbell_samples = explorer.preview_samples("doorbell", limit=3)
```

---

## Future Integration

The `DatasetManifest` produced by `AudioDatasetLoader` serves as the direct input to downstream pipeline stages:
- **Phase 3 (Preprocessing)**: `AudioPreprocessor` will consume `DatasetItem.metadata.file_path` to load, normalize, and frame 16 kHz signals into `dataset/processed/`.
- **Phase 4 (Feature Extraction)**: `SpectrogramExtractor` will generate Log-Mel Spectrogram matrices from preprocessed samples.
- **Phase 5 (AI Training)**: `ModelTrainer` will utilize preprocessed spectrogram datasets for neural network training.
