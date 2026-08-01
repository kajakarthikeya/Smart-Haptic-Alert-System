"""Feature Extraction Interface & Log-Mel Spectrogram Generator."""

from abc import ABC, abstractmethod
from typing import Any
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


class BaseFeatureExtractor(ABC):
    """Abstract Base Class for Audio Feature Extractors."""

    @abstractmethod
    def extract(self, signal: Any) -> Any:
        """Transforms preprocessed audio signal into model input features.

        Args:
            signal: Preprocessed 1D audio waveform array.

        Returns:
            2D or 3D feature representation (e.g. Log-Mel Spectrogram tensor).
        """
        pass


class SpectrogramExtractor(BaseFeatureExtractor):
    """Computes Log-Mel Spectrograms from audio waveforms for deep learning classifiers."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 64,
    ) -> None:
        """Initializes spectral parameters.

        Args:
            sample_rate: Sampling frequency.
            n_fft: FFT window size.
            hop_length: Number of samples between successive frames.
            n_mels: Number of Mel frequency bands.
        """
        self._sample_rate = sample_rate or settings.audio.sample_rate
        self._n_fft = n_fft or settings.audio.n_fft
        self._hop_length = hop_length or settings.audio.hop_length
        self._n_mels = n_mels or settings.audio.n_mels
        logger.info(
            f"SpectrogramExtractor initialized (n_fft={self._n_fft}, "
            f"hop_length={self._hop_length}, n_mels={self._n_mels})"
        )

    def extract(self, signal: Any) -> Any:
        """Computes log-mel spectrogram placeholder.

        Args:
            signal: Audio waveform.

        Returns:
            Placeholder feature tensor array.
        """
        # Placeholder feature transformation (will use librosa/tensorflow in AI phase)
        return signal
