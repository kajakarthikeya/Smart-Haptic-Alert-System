"""Length Standardization Engine for fixed audio duration trimming and zero-padding."""

from typing import List, Optional
from config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LengthStandardizer:
    """Trims or zero-pads audio waveforms to an exact fixed sample count (default: 4.0s at 22,050 Hz = 88,200 samples)."""

    def __init__(
        self,
        target_duration_sec: Optional[float] = None,
        target_sample_rate: Optional[int] = None,
    ) -> None:
        """Initializes LengthStandardizer.

        Args:
            target_duration_sec: Target duration in seconds (default 4.0s).
            target_sample_rate: Target sample rate in Hz (default 22050 Hz).
        """
        self._duration_sec = target_duration_sec or settings.preprocessing.target_duration_sec
        self._sample_rate = target_sample_rate or settings.preprocessing.target_sample_rate
        self._target_samples = int(self._duration_sec * self._sample_rate)
        logger.info(
            f"LengthStandardizer initialized (duration={self._duration_sec}s, "
            f"sample_rate={self._sample_rate}Hz, target_samples={self._target_samples})"
        )

    @property
    def target_samples(self) -> int:
        return self._target_samples

    @property
    def target_duration_sec(self) -> float:
        return self._duration_sec

    def standardize_length(self, waveform: List[float]) -> List[float]:
        """Trims longer audio clips or zero-pads shorter clips to exact target_samples length.

        Args:
            waveform: Input 1D audio sample array.

        Returns:
            Fixed length 1D float list of size self.target_samples.
        """
        num_samples = len(waveform)

        if num_samples == self._target_samples:
            return list(waveform)

        if num_samples > self._target_samples:
            # Trim center / excess trailing samples
            trimmed = waveform[: self._target_samples]
            logger.debug(f"Trimmed audio clip: {num_samples} -> {self._target_samples} samples")
            return trimmed

        # Zero-pad shorter clips symmetrically or right-side
        padding_needed = self._target_samples - num_samples
        padded = waveform + [0.0] * padding_needed
        logger.debug(f"Padded audio clip: {num_samples} -> {self._target_samples} samples (added {padding_needed} zeros)")
        return padded
