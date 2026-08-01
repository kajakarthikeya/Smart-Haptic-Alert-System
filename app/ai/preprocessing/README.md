# Audio Preprocessing Subsystem (`app/ai/preprocessing/`)

## Module Purpose
The **Audio Preprocessing Subsystem** cleans, standardizes, trims, normalizes, and length-aligns environmental sound recordings from the Dataset Management module (`dataset/raw/{classes}`) before feature extraction and neural network model training.

It guarantees that every audio sample passed to downstream feature extraction is formatted as a **16-bit PCM WAV, 22,050 Hz sample rate, single-channel Mono, peak-normalized, and exactly 4.0 seconds long (88,200 samples)**.

> [!IMPORTANT]
> This module performs signal standardization ONLY. It does **not** extract MFCCs or spectrogram features, nor does it train AI models.

---

## Target Sound Classes

1. **Ambulance**: Emergency siren audio samples (`ambulance/`)
2. **Car Horn**: Traffic and automotive horn warnings (`car_horn/`)
3. **Fire Alarm**: Critical fire alarms and smoke detectors (`fire_alarm/`)
4. **Doorbell**: Domestic entrance chimes and bells (`doorbell/`)
5. **Dog Bark**: Domestic animal bark warnings (`dog_bark/`)

---

## Architecture & Layout

```
dataset/
├── raw/                                # Unprocessed raw audio input samples
│   ├── ambulance/
│   ├── car_horn/
│   ├── fire_alarm/
│   ├── doorbell/
│   └── dog_bark/
└── processed/                          # Standardized output destination
    ├── ambulance/
    ├── car_horn/
    ├── fire_alarm/
    ├── doorbell/
    ├── dog_bark/
    └── preprocessed_metadata.json      # Summary metadata JSON

app/ai/preprocessing/                   # Module Implementation
├── __init__.py                         # Public API exports
├── exceptions.py                       # Custom domain exceptions
├── models.py                           # Dataclasses (RawAudioData, ProcessedFileMetadata, BatchPreprocessingSummary)
├── audio_loader.py                     # Multi-format audio file loader (.wav, .mp3, .flac)
├── audio_standardizer.py               # Mono conversion, 22050Hz resampling, peak amplitude normalization
├── silence_processor.py               # Leading & trailing silence trimming (-40.0 dB threshold)
├── noise_reducer.py                    # Optional background noise reduction filter
├── length_standardizer.py              # Fixed length (4.0s = 88,200 samples) trimming & zero-padding
├── metadata_generator.py               # Metadata JSON generator & exporter
├── preprocessing_pipeline.py           # End-to-end batch processing pipeline
└── README.md                           # Documentation
```

---

## Preprocessing Pipeline Workflow

```
Raw Input File (.wav, .mp3, .flac)
        │
        ▼
[AudioLoader] ───────────────> Read PCM waveform & sample rate
        │
        ▼
[AudioStandardizer] ─────────> Mono Conversion -> 22,050 Hz Resampling -> Peak Amplitude Normalization
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
[PreprocessingPipeline] ─────> Write 16-bit WAV to dataset/processed/{class}/ & Export JSON metadata
```

---

## Classes & Public Methods Specification

### 1. `AudioLoader` ([audio_loader.py](file:///d:/PROJECTS/Smart-Haptic-Alert-System/app/ai/preprocessing/audio_loader.py))
Reads `.wav`, `.mp3`, and `.flac` files into raw PCM floating point arrays.

- `load_audio(file_path: Union[str, Path], class_label: str = "unknown") -> RawAudioData`: Loads audio into `RawAudioData` containing float32 waveform list, sample rate, and channel count.

### 2. `AudioStandardizer` ([audio_standardizer.py](file:///d:/PROJECTS/Smart-Haptic-Alert-System/app/ai/preprocessing/audio_standardizer.py))
Standardizes audio formatting parameters.

- `convert_to_mono(waveform: List[float], channels: int) -> List[float]`: Averages multi-channel audio frames to single-channel mono.
- `resample(waveform: List[float], orig_sr: int, target_sr: int) -> List[float]`: Resamples waveform to target sample rate (22,050 Hz).
- `normalize_peak_amplitude(waveform: List[float], target_peak: float = 0.95) -> List[float]`: Scales waveform peak amplitude to `[-0.95, 0.95]`.
- `standardize(raw_data: RawAudioData) -> List[float]`: Runs full pipeline (Mono -> Resample -> Peak Normalize).

### 3. `SilenceProcessor` ([silence_processor.py](file:///d:/PROJECTS/Smart-Haptic-Alert-System/app/ai/preprocessing/silence_processor.py))
Trims uninformative silence frames.

- `trim_silence(waveform: List[float], frame_length: int = 512) -> List[float]`: Trims leading and trailing silence below decibel threshold (`-40.0` dB).

### 4. `NoiseReducer` ([noise_reducer.py](file:///d:/PROJECTS/Smart-Haptic-Alert-System/app/ai/preprocessing/noise_reducer.py))
Optional noise filter.

- `reduce_noise(waveform: List[float]) -> List[float]`: Applies smoothing filter to suppress stationary background noise floor.

### 5. `LengthStandardizer` ([length_standardizer.py](file:///d:/PROJECTS/Smart-Haptic-Alert-System/app/ai/preprocessing/length_standardizer.py))
Ensures uniform tensor dimensions.

- `standardize_length(waveform: List[float]) -> List[float]`: Trims clips longer than 4.0s or zero-pads clips shorter than 4.0s to exactly **88,200 samples**.

### 6. `MetadataGenerator` ([metadata_generator.py](file:///d:/PROJECTS/Smart-Haptic-Alert-System/app/ai/preprocessing/metadata_generator.py))
Metadata JSON exporter.

- `create_file_metadata(...) -> ProcessedFileMetadata`: Generates metadata record.
- `export_summary_json(...) -> Path`: Exports metadata JSON summary file.

### 7. `PreprocessingPipeline` ([preprocessing_pipeline.py](file:///d:/PROJECTS/Smart-Haptic-Alert-System/app/ai/preprocessing/preprocessing_pipeline.py))
Batch pipeline dispatcher.

- `process_file(file_path: Path, class_label: str, overwrite: bool = False) -> Optional[ProcessedFileMetadata]`: Preprocesses individual file.
- `process_dataset(overwrite: bool = False) -> BatchPreprocessingSummary`: Batch processes dataset raw folder tree into `dataset/processed/`.

---

## Configuration Options (`config.py`)

Configured via `settings.preprocessing`:
- `target_sample_rate`: `22050` Hz
- `target_channels`: `1` (Mono)
- `target_duration_sec`: `4.0` seconds (88,200 samples)
- `silence_threshold_db`: `-40.0` dB
- `enable_noise_reduction`: `True`
- `bit_depth`: `16` (PCM 16-bit)
- `target_format`: `"wav"`
- `processed_dir`: `dataset/processed`

---

## Code Usage Example

```python
from app.ai.preprocessing import PreprocessingPipeline

# 1. Instantiate batch pipeline (reads paths automatically from config.py)
pipeline = PreprocessingPipeline()

# 2. Execute batch preprocessing on raw dataset
summary = pipeline.process_dataset(overwrite=False)

print(f"Batch Preprocessing Complete in {summary.total_time_sec:.2f}s")
print(f"Processed: {summary.processed_count}, Skipped: {summary.skipped_count}, Errors: {summary.error_count}")
print("Class breakdown:", summary.class_breakdown)
```

---

## Future Integration & Next Phase

Preprocessed audio files produced under `dataset/processed/{classes}` serve as the direct input to **Phase 4 (Feature Extraction Subsystem)**:
- `SpectrogramExtractor` in `app/ai/feature_extraction/` will consume these standardized 22,050 Hz 4.0-second WAV files to compute **Log-Mel Spectrogram** feature matrices for CNN model training.
