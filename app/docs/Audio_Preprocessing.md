# Audio Preprocessing Technical Specification - Smart Haptic Alert System

## 1. Overview

The **Audio Preprocessing Subsystem** (`app/ai/preprocessing/`) standardizes, cleans, trims, normalizes, and length-aligns raw environmental sound recordings from the Dataset Management module (`dataset/raw/{classes}`) before feature extraction and deep learning model training.

Every preprocessed output file generated under `dataset/processed/{classes}` is formatted as:
- **Audio Format**: 16-bit PCM WAV
- **Sample Rate**: 22,050 Hz
- **Channels**: 1 (Mono)
- **Duration**: 4.0 seconds (exactly 88,200 samples)
- **Peak Amplitude**: Normalized to `[-0.95, 0.95]`

---

## 2. Target Sound Classes

1. **`ambulance`**: Emergency sirens and warning beacons
2. **`car_horn`**: Vehicle and traffic horns
3. **`fire_alarm`**: Fire alarms and smoke detector sirens
4. **`doorbell`**: Domestic entrance bells and chimes
5. **`dog_bark`**: Domestic animal alert barks

---

## 3. Signal Processing Pipeline & Algorithms

```
Raw Audio File (.wav, .mp3, .flac)
        │
        ▼
Stage 1: Audio Loading (AudioLoader)
        │ Reads raw PCM waveform into float32 array
        ▼
Stage 2: Audio Standardization (AudioStandardizer)
        ├─ Mono Conversion (averages multi-channel interleaved frames)
        ├─ Linear Interpolation Resampling to 22,050 Hz
        └─ Peak Amplitude Normalization ([-0.95, 0.95])
        │
        ▼
Stage 3: Silence Processing (SilenceProcessor)
        └─ RMS frame energy calculation (threshold: -40.0 dB)
        └─ Trims leading & trailing silent frames
        │
        ▼
Stage 4: Noise Reduction (NoiseReducer)
        └─ 3-point moving average spectral smoothing filter (optional)
        │
        ▼
Stage 5: Length Standardization (LengthStandardizer)
        ├─ Trims clips longer than 4.0s (> 88,200 samples)
        └─ Zero-pads clips shorter than 4.0s (< 88,200 samples)
        │
        ▼
Stage 6: Serialization & Metadata (PreprocessingPipeline & MetadataGenerator)
        ├─ Writes 16-bit PCM WAV to dataset/processed/{class}/
        └─ Exports summary JSON metadata to dataset/processed/preprocessed_metadata.json
```

---

## 4. Subsystem Components & Contracts

| Module | Component Class | Responsibility |
| :--- | :--- | :--- |
| `exceptions.py` | `PreprocessingError`, `AudioLoadError`, `UnsupportedFormatError`, `ProcessingError`, `CorruptedAudioError` | Custom domain exceptions. |
| `models.py` | `RawAudioData`, `ProcessedAudioSignal`, `ProcessedFileMetadata`, `BatchPreprocessingSummary` | Dataclasses for waveforms, metadata, and batch statistics. |
| `audio_loader.py` | `AudioLoader` | Reads `.wav`, `.mp3`, and `.flac` audio files into memory arrays. |
| `audio_standardizer.py` | `AudioStandardizer` | Executes mono conversion, 22050 Hz linear resampling, and peak normalization. |
| `silence_processor.py` | `SilenceProcessor` | Trims uninformative leading/trailing silence below `-40.0` dB threshold. |
| `noise_reducer.py` | `NoiseReducer` | Applies background noise reduction filter. |
| `length_standardizer.py` | `LengthStandardizer` | Ensures exact 4.0-second (88,200 samples) duration alignment via trimming/zero-padding. |
| `metadata_generator.py` | `MetadataGenerator` | Generates SHA-256 hashes and exports JSON metadata summaries. |
| `preprocessing_pipeline.py` | `PreprocessingPipeline` | Batch pipeline runner processing dataset trees recursively and handling per-file errors. |

---

## 5. Configuration Settings (`config.py`)

All preprocessing parameters are configured centrally via `settings.preprocessing`:

```python
PREPROCESS_SAMPLE_RATE = 22050          # Hz
PREPROCESS_CHANNELS = 1                # Mono
PREPROCESS_TARGET_DURATION_SEC = 4.0    # Seconds
PREPROCESS_SILENCE_THRESHOLD_DB = -40.0 # dB threshold
PREPROCESS_ENABLE_NOISE_REDUCTION = True # Enable smoothing filter
PREPROCESS_BIT_DEPTH = 16              # PCM bit depth
PREPROCESS_TARGET_FORMAT = "wav"        # Target format extension
```

---

## 6. Resilience & Error Handling

- **Non-blocking Batch Execution**: If an individual raw file is corrupted or unreadable, `PreprocessingPipeline` logs the error, updates `BatchPreprocessingSummary.errors`, and continues processing remaining files without terminating execution.
- **Idempotent Processing**: Existing preprocessed files in `dataset/processed/` are skipped unless `overwrite=True` is specified.
- **Zero-Byte Safety**: `AudioLoader` and `DatasetValidator` verify non-zero byte size before frame parsing.
