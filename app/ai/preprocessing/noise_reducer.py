"""Optional Background Noise Reduction Filter."""

from typing import List, Optional
from config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class NoiseReducer:
    """Optional background noise reduction processor using moving average spectral smoothing."""

    def __init__(self, enabled: Optional[bool] = None, filter_strength: float = 0.1) -> None:
        """Initializes NoiseReducer.

        Args:
            enabled: Toggle noise reduction filter on/off.
                     Defaults to settings.preprocessing.enable_noise_reduction.
            filter_strength: Smoothing filter factor (0.0 to 1.0).
        """
        self._enabled = enabled if enabled is not None else settings.preprocessing.enable_noise_reduction
        self._filter_strength = filter_strength
        logger.info(f"NoiseReducer initialized (enabled={self._enabled}, strength={self._filter_strength})")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def reduce_noise(self, waveform: List[float]) -> List[float]:
        """Applies background noise reduction smoothing if enabled.

        Args:
            waveform: Input 1D audio waveform.

        Returns:
            Noise-reduced 1D float list.
        """
        if not self._enabled or not waveform or len(waveform) < 3:
            return list(waveform)

        # 3-point weighted moving average filter for noise floor suppression
        n = len(waveform)
        cleaned: List[float] = [0.0] * n
        alpha = self._filter_strength

        cleaned[0] = waveform[0]
        cleaned[-1] = waveform[-1]

        for i in range(1, n - 1):
            smoothed = 0.25 * waveform[i - 1] + 0.5 * waveform[i] + 0.25 * waveform[i + 1]
            cleaned[i] = (1.0 - alpha) * waveform[i] + alpha * smoothed

        logger.debug(f"Applied noise reduction smoothing filter to {n} samples.")
        return cleaned
