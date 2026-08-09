"""Feature Extraction Interface & Log-Mel Spectrogram Generator."""

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np

from app.ai.feature_extraction.feature_extractor import FeatureExtractor
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
        sample_rate: Optional[int] = None,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        n_mels: Optional[int] = None,
    ) -> None:
        """Initializes spectral parameters.

        Args:
            sample_rate: Sampling frequency.
            n_fft: FFT window size.
            hop_length: Number of samples between successive frames.
            n_mels: Number of Mel frequency bands.
        """
        self._sample_rate = sample_rate or settings.preprocessing.target_sample_rate
        self._n_fft = n_fft or settings.feature_extraction.n_fft
        self._hop_length = hop_length or settings.feature_extraction.hop_length
        self._n_mels = n_mels or settings.feature_extraction.n_mels

        self._extractor = FeatureExtractor(
            sample_rate=self._sample_rate,
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            n_mels=self._n_mels,
        )

        logger.info(
            f"SpectrogramExtractor initialized (sr={self._sample_rate}, n_fft={self._n_fft}, "
            f"hop_length={self._hop_length}, n_mels={self._n_mels})"
        )

    def extract(self, signal: Any) -> np.ndarray:
        """Computes log-mel spectrogram array using Librosa feature engine.

        Args:
            signal: Audio waveform array.

        Returns:
            2D numpy array of shape (n_mels, time_steps).
        """
        return self._extractor.extract_mel_spectrogram(signal, sr=self._sample_rate)
