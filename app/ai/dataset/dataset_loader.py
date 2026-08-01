"""Dataset Loader implementation for reading and organizing audio datasets."""

from abc import ABC, abstractmethod
import hashlib
from pathlib import Path
import struct
from typing import Dict, List, Optional, Set, Tuple, Union
import wave

from config import settings
from app.utils.logger import get_logger
from app.ai.dataset.models import AudioFileMetadata, DatasetItem, DatasetManifest
from app.ai.dataset.exceptions import DatasetNotFoundError, CorruptedAudioError

logger = get_logger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """Computes SHA-256 hash of a file for duplicate detection."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def extract_audio_file_info(file_path: Path) -> Tuple[float, int, int]:
    """Extracts duration (seconds), sample rate (Hz), and channel count from audio headers.

    Supports .wav via standard library wave module; fallback for .mp3 / .flac.

    Args:
        file_path: Target audio file path.

    Returns:
        Tuple of (duration_sec, sample_rate, channels).
    """
    ext = file_path.suffix.lower()
    
    if ext == ".wav":
        try:
            with wave.open(str(file_path), "rb") as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                duration = n_frames / float(sample_rate) if sample_rate > 0 else 0.0
                return duration, sample_rate, channels
        except Exception as e:
            logger.warning(f"Wave header parse failed for '{file_path}': {e}")
            raise CorruptedAudioError(str(file_path), f"Invalid WAV header: {e}")
    
    # Generic fallback metadata reader for .mp3 / .flac placeholder files or non-wav formats
    file_size = file_path.stat().st_size
    if file_size == 0:
        raise CorruptedAudioError(str(file_path), "File size is 0 bytes")
    
    # Default estimated parameters (e.g. 16kHz, mono, calculated duration ratio)
    sample_rate = settings.audio.sample_rate
    channels = settings.audio.channels
    estimated_duration = max(0.1, file_size / (sample_rate * 2))
    return estimated_duration, sample_rate, channels


class BaseDatasetLoader(ABC):
    """Abstract Base Class contract for Dataset Loaders."""

    @abstractmethod
    def load_dataset(self, data_dir: Union[str, Path]) -> DatasetManifest:
        """Scans dataset directory and returns structured DatasetManifest."""
        pass

    @abstractmethod
    def get_supported_extensions(self) -> Set[str]:
        """Returns set of supported file extension strings."""
        pass


class AudioDatasetLoader(BaseDatasetLoader):
    """Concrete Dataset Loader for environmental sound datasets."""

    def __init__(
        self,
        supported_extensions: Optional[Set[str]] = None,
        target_classes: Optional[Tuple[str, ...]] = None,
    ) -> None:
        """Initializes AudioDatasetLoader with configuration defaults.

        Args:
            supported_extensions: Set of supported extensions (e.g., {'.wav', '.mp3', '.flac'}).
            target_classes: Sequence of target class labels.
        """
        self._supported_extensions = supported_extensions or set(settings.dataset.supported_extensions)
        self._target_classes = target_classes or settings.dataset.target_classes
        logger.info(
            f"AudioDatasetLoader initialized with supported extensions: {sorted(list(self._supported_extensions))}"
        )

    def get_supported_extensions(self) -> Set[str]:
        return self._supported_extensions

    def load_dataset(self, data_dir: Union[str, Path]) -> DatasetManifest:
        """Scans raw dataset directory, organizes files by class, and returns a DatasetManifest.

        Args:
            data_dir: Path to dataset directory (e.g. dataset/raw).

        Returns:
            DatasetManifest containing all scanned DatasetItem objects.

        Raises:
            DatasetNotFoundError: If data_dir does not exist.
        """
        root_path = Path(data_dir)
        if not root_path.exists():
            logger.error(f"Dataset root directory not found: {root_path}")
            raise DatasetNotFoundError(f"Dataset directory '{root_path}' does not exist.")

        logger.info(f"Scanning dataset from directory: {root_path}")
        items: List[DatasetItem] = []

        # Iterate over class directories or direct files
        class_dirs = [d for d in root_path.iterdir() if d.is_dir()]
        
        # If no subdirectories exist, check direct directory
        if not class_dirs:
            class_dirs = [root_path]

        for class_dir in class_dirs:
            class_label = class_dir.name.lower()
            for file_path in class_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in self._supported_extensions:
                    try:
                        item = self._process_file(file_path, class_label)
                        items.append(item)
                    except CorruptedAudioError as cae:
                        logger.warning(f"Skipping corrupted file '{file_path}': {cae.reason}")
                    except Exception as e:
                        logger.error(f"Unexpected error loading file '{file_path}': {e}")

        manifest = DatasetManifest(items=items, root_dir=root_path)
        logger.info(
            f"Dataset loading complete: {manifest.total_count} files loaded across "
            f"{len(manifest.class_counts)} classes."
        )
        return manifest

    def _process_file(self, file_path: Path, class_label: str) -> DatasetItem:
        """Reads metadata and constructs a DatasetItem."""
        file_size = file_path.stat().st_size
        if file_size == 0:
            raise CorruptedAudioError(str(file_path), "File size is 0 bytes")

        duration, sample_rate, channels = extract_audio_file_info(file_path)
        file_hash = compute_file_hash(file_path)

        meta = AudioFileMetadata(
            file_path=file_path.resolve(),
            file_name=file_path.name,
            class_label=class_label,
            file_size_bytes=file_size,
            duration_sec=duration,
            sample_rate=sample_rate,
            channels=channels,
            sha256_hash=file_hash,
            extension=file_path.suffix.lower(),
        )
        return DatasetItem(metadata=meta)
