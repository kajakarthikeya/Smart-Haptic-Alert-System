"""Audio Preprocessing Interface & Implementations."""

from abc import ABC, abstractmethod
from typing import Any
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


class BasePreprocessor(ABC):
    """Abstract Base Class for Signal Preprocessors."""

    @abstractmethod
    def process(self, raw_signal: Any) -> Any:
        """Processes raw input audio signal.

        Args:
            raw_signal: Input raw PCM waveform or byte array.

        Returns:
            Preprocessed signal tensor or array.
        """
        pass


class AudioPreprocessor(BasePreprocessor):
    """Standard Audio Preprocessor handling resampling, normalization, and fixed length padding."""

    def __init__(self, target_sample_rate: int = 16000, duration_sec: float = 1.0) -> None:
        """Initializes preprocessor parameters.

        Args:
            target_sample_rate: Standardized sampling rate (e.g. 16kHz).
            duration_sec: Fixed audio chunk duration in seconds.
        """
        self._sample_rate = target_sample_rate or settings.audio.sample_rate
        self._duration_sec = duration_sec or settings.audio.frame_duration_sec
        self._target_length = int(self._sample_rate * self._duration_sec)
        logger.info(
            f"AudioPreprocessor initialized (target_sr={self._sample_rate}Hz, "
            f"duration={self._duration_sec}s, target_samples={self._target_length})"
        )

    def normalize(self, signal: Any) -> Any:
        """Normalizes audio waveform amplitude.

        Args:
            signal: Audio waveform data.

        Returns:
            Normalized signal array.
        """
        # Placeholder amplitude normalization logic
        return signal

    def pad_or_truncate(self, signal: Any) -> Any:
        """Ensures fixed-length audio signal.

        Args:
            signal: Audio signal.

        Returns:
            Fixed length signal array.
        """
        # Placeholder padding / truncation logic
        return signal

    def process(self, raw_signal: Any) -> Any:
        """Applies normalization and padding/truncation pipeline to input signal.

        Args:
            raw_signal: Input PCM waveform.

        Returns:
            Cleaned and framed signal.
        """
        normalized = self.normalize(raw_signal)
        framed = self.pad_or_truncate(normalized)
        return framed
