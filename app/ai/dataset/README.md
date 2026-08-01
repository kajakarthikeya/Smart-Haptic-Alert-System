# Dataset Management Subsystem (`app/ai/dataset/`)

## Module Purpose
The **Dataset Management Subsystem** provides a production-ready, object-oriented framework for directory structure initialization, audio loading, multi-level validation, statistical analysis, and data exploration for environmental sound datasets.

It prepares raw audio samples for downstream preprocessing and neural network training while enforcing Clean Architecture and SOLID design principles.

> [!IMPORTANT]
> This module manages and validates dataset integrity ONLY. It does **not** perform signal preprocessing, spectrogram feature extraction, or neural network training.

---

## Target Sound Classes

1. **Ambulance**: Emergency siren audio samples (`ambulance/`)
2. **Car Horn**: Traffic and automotive horn warnings (`car_horn/`)
3. **Fire Alarm**: Critical fire alarms and smoke detectors (`fire_alarm/`)
4. **Doorbell**: Domestic entrance chimes and bells (`doorbell/`)
5. **Dog Bark**: Domestic animal bark warnings (`dog_bark/`)

---

## Architecture & Directory Layout

```
dataset/                                # Managed Dataset Root (configured via config.py)
├── raw/                                # Raw audio input samples
│   ├── ambulance/
│   ├── car_horn/
│   ├── fire_alarm/
│   ├── doorbell/
│   └── dog_bark/
├── processed/                          # Destination for preprocessed spectrogram tensors
└── test_audio/                         # Independent evaluation audio samples

app/ai/dataset/                         # Subsystem Implementation
├── __init__.py                         # Public API exports
├── exceptions.py                       # Custom domain exceptions
├── models.py                           # Dataclasses & value objects (DatasetManifest, ValidationReport, DatasetStats)
├── dataset_directory_manager.py        # Automatic folder hierarchy creation & verification
├── dataset_loader.py                   # Audio scanner & DatasetManifest builder (.wav, .mp3, .flac)
├── dataset_validator.py                # Integrity, missing class, corruption & duplicate checks
├── dataset_statistics.py               # Summary statistical calculator & JSON exporter
├── dataset_explorer.py                # Querying, filtering, sample preview & search tool
└── README.md                           # Subsystem documentation
```

---

## Classes & Public Methods Specification

### 1. `DatasetDirectoryManager` ([dataset_directory_manager.py](file:///d:/PROJECTS/Smart-Haptic-Alert-System/app/ai/dataset/dataset_directory_manager.py))
Manages automatic directory creation and structural verification.

- `initialize_directories() -> Dict[str, Path]`: Automatically creates `dataset/raw/{ambulance, car_horn, fire_alarm, doorbell, dog_bark}`, `dataset/processed/`, and `dataset/test_audio/`.
- `verify_structure() -> Dict[str, bool]`: Verifies presence of all required dataset directories.
- `get_class_directories() -> Dict[str, Path]`: Returns dictionary mapping target class label to its `Path` under `raw_dir`.

### 2. `AudioDatasetLoader` ([dataset_loader.py](file:///d:/PROJECTS/Smart-Haptic-Alert-System/app/ai/dataset/dataset_loader.py))
Scans audio datasets and builds structured `DatasetManifest` objects.

- `load_dataset(data_dir: Union[str, Path]) -> DatasetManifest`: Scans target directory for `.wav`, `.mp3`, and `.flac` files, parses metadata (duration, sample rate, channels), computes SHA-256 content hashes, and returns a `DatasetManifest`.
- `get_supported_extensions() -> Set[str]`: Returns set of supported audio extensions.

### 3. `DatasetValidator` ([dataset_validator.py](file:///d:/PROJECTS/Smart-Haptic-Alert-System/app/ai/dataset/dataset_validator.py))
Validates dataset structural integrity, file formatting, corruption, and duplicates.

- `validate_directory(raw_dir: Union[str, Path]) -> ValidationReport`: Runs validation checks:
  - Missing target sound class directories
  - Empty class folders
  - Unsupported file formats
  - Corrupted/0-byte audio files
  - Duplicate files (SHA-256 hash matching)
  - Invalid filenames (spaces or special characters)
- `validate_manifest(manifest: DatasetManifest) -> ValidationReport`: Validates an in-memory `DatasetManifest`.

### 4. `DatasetStatisticsCalculator` ([dataset_statistics.py](file:///d:/PROJECTS/Smart-Haptic-Alert-System/app/ai/dataset/dataset_statistics.py))
Computes aggregate statistical summaries and exports JSON reports.

- `compute_statistics(manifest: DatasetManifest) -> DatasetStats`: Computes total files, total size in MB, per-class counts, duration min/max/avg, sample rate distribution, and channel breakdown.
- `export_json_report(stats: DatasetStats, report_name: str = "dataset_stats.json") -> Path`: Saves `DatasetStats` as a formatted JSON report under `app/outputs/`.

### 5. `DatasetExplorer` ([dataset_explorer.py](file:///d:/PROJECTS/Smart-Haptic-Alert-System/app/ai/dataset/dataset_explorer.py))
Querying and inspection tool for dataset contents.

- `list_classes() -> List[str]`: Returns sorted list of sound class labels present.
- `count_files(class_label: Optional[str] = None) -> int`: Returns total file count or file count for a specific class.
- `get_class_distribution() -> Dict[str, int]`: Returns dictionary of class label -> count.
- `preview_samples(class_label: str, limit: int = 5) -> List[DatasetItem]`: Retrieves preview samples for a class.
- `search_files(query: Optional[str] = None, class_label: Optional[str] = None, min_duration_sec: Optional[float] = None, max_duration_sec: Optional[float] = None) -> List[DatasetItem]`: Filters dataset items by keyword, class, or duration bounds.
- `get_folder_info(raw_dir: Optional[Path] = None) -> Dict[str, Any]`: Returns summary info for raw directory.

---

## Custom Exception Hierarchy ([exceptions.py](file:///d:/PROJECTS/Smart-Haptic-Alert-System/app/ai/dataset/exceptions.py))

- `DatasetError` (Base domain exception)
  - `DatasetNotFoundError`: Raised when dataset directory or file does not exist.
  - `MissingClassError`: Raised when required sound class directories are missing.
  - `CorruptedAudioError`: Raised when audio file is unreadable, 0 bytes, or has invalid headers.
  - `InvalidDatasetError`: Raised when dataset validation fails critical threshold requirements.

---

## Data Models ([models.py](file:///d:/PROJECTS/Smart-Haptic-Alert-System/app/ai/dataset/models.py))

- `AudioFileMetadata`: File path, class label, file size, duration, sample rate, channels, SHA-256 hash, extension.
- `DatasetItem`: Individual dataset sample wrapper.
- `DatasetManifest`: Immutable aggregate dataset manifest container.
- `ValidationIssue` & `ValidationReport`: Structured validation findings with `.to_json()`.
- `DatasetStats`: Aggregate dataset metrics with `.to_json()`.

---

## Usage Workflow Example

```python
from app.ai.dataset import (
    DatasetDirectoryManager,
    AudioDatasetLoader,
    DatasetValidator,
    DatasetStatisticsCalculator,
    DatasetExplorer,
)

# 1. Automatically create dataset directories
dir_mgr = DatasetDirectoryManager()
dir_mgr.initialize_directories()

# 2. Validate raw dataset directory
validator = DatasetValidator()
report = validator.validate_directory(dir_mgr.raw_dir)
print("Is Dataset Valid?", report.is_valid)

# 3. Load dataset manifest
loader = AudioDatasetLoader()
manifest = loader.load_dataset(dir_mgr.raw_dir)

# 4. Compute statistics & save JSON report
stats_calc = DatasetStatisticsCalculator()
stats = stats_calc.compute_statistics(manifest)
json_path = stats_calc.export_json_report(stats, "dataset_stats.json")
print("Statistics report saved to:", json_path)

# 5. Search & Explore samples
explorer = DatasetExplorer(manifest)
print("Sound classes present:", explorer.list_classes())
doorbell_samples = explorer.preview_samples("doorbell", limit=3)
```

---

## Dependencies

- **Python Standard Library**: `pathlib`, `dataclasses`, `enum`, `json`, `wave`, `hashlib`, `re`, `struct`, `typing`
- **Application Core**: `config.settings`, `app.utils.logger`

---

## Future Improvements & Next Phase Integration

1. **Phase 3 Integration**: The `DatasetManifest` produced by `AudioDatasetLoader` provides file paths and audio metadata directly to `AudioPreprocessor` in `app/ai/preprocessing/`.
2. **Dynamic Class Registration**: Allowing custom user-defined sound classes via configuration extensions.
3. **Automated Dataset Downloading**: Integrations for auto-fetching public sound benchmarks (ESC-50, UrbanSound8K).
