"""Batch Audio Preprocessing Pipeline orchestrating dataset standardization and output generation."""

from pathlib import Path
import struct
import time
from typing import Dict, List, Optional, Union
import wave

from config import settings
from app.utils.logger import get_logger
from app.ai.preprocessing.audio_loader import AudioLoader
from app.ai.preprocessing.audio_standardizer import AudioStandardizer
from app.ai.preprocessing.silence_processor import SilenceProcessor
from app.ai.preprocessing.noise_reducer import NoiseReducer
from app.ai.preprocessing.length_standardizer import LengthStandardizer
from app.ai.preprocessing.metadata_generator import MetadataGenerator
from app.ai.preprocessing.models import BatchPreprocessingSummary, ProcessedFileMetadata, ProcessedAudioSignal
from app.ai.preprocessing.exceptions import PreprocessingError

logger = get_logger(__name__)


def write_wav_file(file_path: Path, waveform: List[float], sample_rate: int = 22050) -> None:
    """Writes 1D float [-1.0, 1.0] waveform to a 16-bit PCM mono WAV file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = f"<{len(waveform)}h"

    # Scale float to 16-bit signed integer [-32768, 32767]
    int16_samples = [max(-32768, min(32767, int(sample * 32767.0))) for sample in waveform]
    raw_bytes = struct.pack(fmt, *int16_samples)

    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(raw_bytes)


class PreprocessingPipeline:
    """Orchestrates end-to-end audio preprocessing pipeline across dataset folders."""

    def __init__(
        self,
        loader: Optional[AudioLoader] = None,
        standardizer: Optional[AudioStandardizer] = None,
        silence_processor: Optional[SilenceProcessor] = None,
        noise_reducer: Optional[NoiseReducer] = None,
        length_standardizer: Optional[LengthStandardizer] = None,
        metadata_generator: Optional[MetadataGenerator] = None,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
    ) -> None:
        """Initializes PreprocessingPipeline dependencies.

        Args:
            loader: AudioLoader instance.
            standardizer: AudioStandardizer instance.
            silence_processor: SilenceProcessor instance.
            noise_reducer: NoiseReducer instance.
            length_standardizer: LengthStandardizer instance.
            metadata_generator: MetadataGenerator instance.
            raw_dir: Source raw dataset path.
            processed_dir: Destination processed dataset path.
        """
        self._loader = loader or AudioLoader()
        self._standardizer = standardizer or AudioStandardizer()
        self._silence_processor = silence_processor or SilenceProcessor()
        self._noise_reducer = noise_reducer or NoiseReducer()
        self._length_standardizer = length_standardizer or LengthStandardizer()
        self._metadata_generator = metadata_generator or MetadataGenerator(processed_dir)

        self._raw_dir = raw_dir or settings.dataset.raw_dir
        self._processed_dir = processed_dir or settings.preprocessing.processed_dir
        self._supported_extensions = settings.dataset.supported_extensions

        logger.info("PreprocessingPipeline initialized successfully.")

    def process_file(
        self, file_path: Path, class_label: str, overwrite: bool = False
    ) -> Optional[ProcessedFileMetadata]:
        """Preprocesses an individual audio file through all pipeline stages.

        Pipeline Stages:
        1. Audio Loader (Read raw PCM)
        2. Audio Standardizer (Mono -> 22050 Hz -> Peak Normalize)
        3. Silence Processor (Trim leading/trailing silence)
        4. Noise Reducer (Optional background noise reduction)
        5. Length Standardizer (Trim / zero-pad to exact 4.0s = 88,200 samples)
        6. Output Serialization (Write 16-bit WAV to processed_dir)

        Args:
            file_path: Input raw file path.
            class_label: Target sound class label.
            overwrite: Force overwrite if output file exists.

        Returns:
            ProcessedFileMetadata or None if skipped/failed.
        """
        target_class_dir = self._processed_dir / class_label
        output_filename = f"{file_path.stem}.wav"
        output_path = target_class_dir / output_filename

        # Skip if file exists and overwrite is False
        if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
            logger.debug(f"Skipping already preprocessed file: {output_path}")
            return None

        # 1. Load Audio
        raw_data = self._loader.load_audio(file_path, class_label)

        # 2. Standardize (Mono, Resample to 22050Hz, Peak Normalize)
        standardized_wave = self._standardizer.standardize(raw_data)

        # 3. Silence Processing
        silence_trimmed_wave = self._silence_processor.trim_silence(standardized_wave)

        # 4. Noise Reduction (Optional)
        noise_reduced_wave = self._noise_reducer.reduce_noise(silence_trimmed_wave)

        # 5. Length Standardization (4.0s = 88,200 samples)
        final_waveform = self._length_standardizer.standardize_length(noise_reduced_wave)

        # 6. Save Processed 16-bit WAV file
        write_wav_file(output_path, final_waveform, sample_rate=self._standardizer._target_sample_rate)

        # 7. Generate Metadata Record
        meta = self._metadata_generator.create_file_metadata(
            output_path=output_path,
            class_label=class_label,
            duration_sec=self._length_standardizer.target_duration_sec,
            sample_rate=self._standardizer._target_sample_rate,
            channels=1,
        )

        logger.info(f"Successfully preprocessed '{file_path.name}' -> '{output_path}'")
        return meta

    def process_dataset(self, overwrite: bool = False) -> BatchPreprocessingSummary:
        """Batch processes entire dataset recursively across raw_dir classes.

        Args:
            overwrite: Overwrite existing preprocessed files if True.

        Returns:
            BatchPreprocessingSummary instance detailing total, processed, skipped, and error counts.
        """
        start_time = time.time()
        logger.info(
            f"Initiating batch dataset preprocessing pipeline: raw_dir='{self._raw_dir}' -> "
            f"processed_dir='{self._processed_dir}'"
        )

        if not self._raw_dir.exists():
            logger.error(f"Raw dataset directory does not exist: {self._raw_dir}")
            return BatchPreprocessingSummary(0, 0, 0, 0, 0.0)

        total_files = 0
        processed_count = 0
        skipped_count = 0
        error_count = 0
        class_breakdown: Dict[str, int] = {}
        errors: List[Dict[str, str]] = []
        metadata_records: List[ProcessedFileMetadata] = []

        # Find class folders or direct files
        class_dirs = [d for d in self._raw_dir.iterdir() if d.is_dir()]
        if not class_dirs:
            class_dirs = [self._raw_dir]

        for class_dir in class_dirs:
            class_label = class_dir.name.lower()
            audio_files = [
                f for f in class_dir.rglob("*") if f.is_file() and f.suffix.lower() in self._supported_extensions
            ]

            logger.info(f"Processing sound class '{class_label}' ({len(audio_files)} files found)...")

            for file_path in audio_files:
                total_files += 1
                try:
                    meta = self.process_file(file_path, class_label, overwrite=overwrite)
                    if meta is not None:
                        processed_count += 1
                        metadata_records.append(meta)
                        class_breakdown[class_label] = class_breakdown.get(class_label, 0) + 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    error_count += 1
                    logger.error(f"Failed to preprocess file '{file_path}': {e}")
                    errors.append({"file_path": str(file_path), "error": str(e)})

        # Save metadata summary report
        if metadata_records:
            self._metadata_generator.export_summary_json(metadata_records)

        elapsed_sec = time.time() - start_time
        summary = BatchPreprocessingSummary(
            total_files=total_files,
            processed_count=processed_count,
            skipped_count=skipped_count,
            error_count=error_count,
            total_time_sec=elapsed_sec,
            class_breakdown=class_breakdown,
            errors=errors,
        )

        logger.info(
            f"Batch preprocessing completed in {elapsed_sec:.2f}s: processed={processed_count}, "
            f"skipped={skipped_count}, errors={error_count}."
        )
        return summary
