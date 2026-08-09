"""Audio Feature Extractor Engine using Librosa.

Extracts MFCC, Mel Spectrogram, Zero Crossing Rate, Spectral Centroid,
Spectral Bandwidth, Spectral Rolloff, and Chroma Features from preprocessed audio signals.
"""

from typing import Any, Dict, Optional, Tuple, Union

import librosa
import numpy as np

from app.ai.feature_extraction.exceptions import InvalidFeatureError, FeatureShapeError
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


class FeatureExtractor:
    """Librosa-based feature extractor computing 7 primary acoustic feature representations."""

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        n_mfcc: Optional[int] = None,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        n_mels: Optional[int] = None,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        n_chroma: Optional[int] = None,
    ) -> None:
        """Initializes acoustic feature extraction parameters.

        Args:
            sample_rate: Target sampling rate (default from config: 22050 Hz).
            n_mfcc: Number of MFCC coefficients.
            n_fft: FFT window size.
            hop_length: Hop length between successive frames.
            win_length: Window duration length in samples.
            n_mels: Number of Mel filter bank channels.
            fmin: Minimum frequency cutoff.
            fmax: Maximum frequency cutoff.
            n_chroma: Number of Chroma bins.
        """
        cfg = settings.feature_extraction
        prep_cfg = settings.preprocessing

        self.sample_rate = sample_rate or prep_cfg.target_sample_rate
        self.n_mfcc = n_mfcc or cfg.n_mfcc
        self.n_fft = n_fft or cfg.n_fft
        self.hop_length = hop_length or cfg.hop_length
        self.win_length = win_length or cfg.win_length
        self.n_mels = n_mels or cfg.n_mels
        self.fmin = fmin if fmin is not None else cfg.fmin
        self.fmax = fmax if fmax is not None else cfg.fmax
        self.n_chroma = n_chroma or cfg.n_chroma

    def validate_audio_signal(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Validates input raw or preprocessed audio waveform array.

        Args:
            y: Audio waveform array.
            sr: Sampling rate.

        Returns:
            Validated 1D numpy array of float32 values.

        Raises:
            InvalidFeatureError: If waveform array is invalid or empty.
        """
        if y is None or len(y) == 0:
            raise InvalidFeatureError("audio", "Audio waveform array is empty or None.")

        y_arr = np.asarray(y, dtype=np.float32)

        if y_arr.ndim != 1:
            y_arr = np.squeeze(y_arr)
            if y_arr.ndim != 1:
                raise InvalidFeatureError("audio", f"Audio waveform must be 1D signal. Got shape {y_arr.shape}.")

        if not np.isfinite(y_arr).all():
            raise InvalidFeatureError("audio", "Audio waveform contains non-finite (NaN or Inf) values.")

        if sr <= 0:
            raise InvalidFeatureError("audio", f"Invalid sampling rate: {sr}")

        return y_arr

    def extract_mfcc(self, y: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """Extracts Mel-Frequency Cepstral Coefficients (MFCC).

        Args:
            y: Audio waveform array.
            sr: Audio sample rate.

        Returns:
            2D numpy array of shape (n_mfcc, time_steps).
        """
        sample_rate = sr or self.sample_rate
        y_arr = self.validate_audio_signal(y, sample_rate)

        mfcc = librosa.feature.mfcc(
            y=y_arr,
            sr=sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        return self._check_feature_matrix("MFCC", mfcc)

    def extract_mel_spectrogram(self, y: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """Extracts Log-Mel Spectrogram representation in decibels (dB).

        Args:
            y: Audio waveform array.
            sr: Audio sample rate.

        Returns:
            2D numpy array of shape (n_mels, time_steps).
        """
        sample_rate = sr or self.sample_rate
        y_arr = self.validate_audio_signal(y, sample_rate)

        mel_spec = librosa.feature.melspectrogram(
            y=y_arr,
            sr=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        # Convert power spectrogram to dB scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return self._check_feature_matrix("Mel Spectrogram", log_mel_spec)

    def extract_zcr(self, y: np.ndarray) -> np.ndarray:
        """Extracts Zero Crossing Rate (ZCR).

        Args:
            y: Audio waveform array.

        Returns:
            2D numpy array of shape (1, time_steps).
        """
        y_arr = self.validate_audio_signal(y, self.sample_rate)
        zcr = librosa.feature.zero_crossing_rate(
            y=y_arr,
            frame_length=self.n_fft,
            hop_length=self.hop_length,
        )
        return self._check_feature_matrix("Zero Crossing Rate", zcr)

    def extract_spectral_centroid(self, y: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """Extracts Spectral Centroid.

        Args:
            y: Audio waveform array.
            sr: Audio sample rate.

        Returns:
            2D numpy array of shape (1, time_steps).
        """
        sample_rate = sr or self.sample_rate
        y_arr = self.validate_audio_signal(y, sample_rate)

        centroid = librosa.feature.spectral_centroid(
            y=y_arr,
            sr=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        return self._check_feature_matrix("Spectral Centroid", centroid)

    def extract_spectral_bandwidth(self, y: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """Extracts Spectral Bandwidth.

        Args:
            y: Audio waveform array.
            sr: Audio sample rate.

        Returns:
            2D numpy array of shape (1, time_steps).
        """
        sample_rate = sr or self.sample_rate
        y_arr = self.validate_audio_signal(y, sample_rate)

        bandwidth = librosa.feature.spectral_bandwidth(
            y=y_arr,
            sr=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        return self._check_feature_matrix("Spectral Bandwidth", bandwidth)

    def extract_spectral_rolloff(self, y: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """Extracts Spectral Rolloff.

        Args:
            y: Audio waveform array.
            sr: Audio sample rate.

        Returns:
            2D numpy array of shape (1, time_steps).
        """
        sample_rate = sr or self.sample_rate
        y_arr = self.validate_audio_signal(y, sample_rate)

        rolloff = librosa.feature.spectral_rolloff(
            y=y_arr,
            sr=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        return self._check_feature_matrix("Spectral Rolloff", rolloff)

    def extract_chroma(self, y: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """Extracts Chroma STFT features.

        Args:
            y: Audio waveform array.
            sr: Audio sample rate.

        Returns:
            2D numpy array of shape (n_chroma, time_steps).
        """
        sample_rate = sr or self.sample_rate
        y_arr = self.validate_audio_signal(y, sample_rate)

        chroma = librosa.feature.chroma_stft(
            y=y_arr,
            sr=sample_rate,
            n_chroma=self.n_chroma,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        return self._check_feature_matrix("Chroma STFT", chroma)

    def extract_all(self, y: np.ndarray, sr: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Extracts all 7 features into a structured dictionary.

        Args:
            y: Audio waveform array.
            sr: Audio sample rate.

        Returns:
            Dictionary mapping feature names to 2D numpy arrays.
        """
        sample_rate = sr or self.sample_rate
        y_arr = self.validate_audio_signal(y, sample_rate)

        return {
            "mfcc": self.extract_mfcc(y_arr, sample_rate),
            "mel_spectrogram": self.extract_mel_spectrogram(y_arr, sample_rate),
            "zero_crossing_rate": self.extract_zcr(y_arr),
            "spectral_centroid": self.extract_spectral_centroid(y_arr, sample_rate),
            "spectral_bandwidth": self.extract_spectral_bandwidth(y_arr, sample_rate),
            "spectral_rolloff": self.extract_spectral_rolloff(y_arr, sample_rate),
            "chroma": self.extract_chroma(y_arr, sample_rate),
        }

    def extract_composite_matrix(self, y: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """Extracts and vertically stacks all features into a single 2D composite matrix.

        Order of stacked features along axis 0:
        1. Mel Spectrogram (n_mels rows)
        2. MFCC (n_mfcc rows)
        3. Zero Crossing Rate (1 row)
        4. Spectral Centroid (1 row)
        5. Spectral Bandwidth (1 row)
        6. Spectral Rolloff (1 row)
        7. Chroma (n_chroma rows)

        Total rows = n_mels + n_mfcc + 1 + 1 + 1 + 1 + n_chroma (default: 128+40+1+1+1+1+12 = 184)

        Args:
            y: Audio waveform array.
            sr: Audio sample rate.

        Returns:
            2D numpy array of shape (total_feature_dim, time_steps).
        """
        feats = self.extract_all(y, sr)
        composite = np.vstack([
            feats["mel_spectrogram"],
            feats["mfcc"],
            feats["zero_crossing_rate"],
            feats["spectral_centroid"],
            feats["spectral_bandwidth"],
            feats["spectral_rolloff"],
            feats["chroma"],
        ])
        return composite.astype(np.float32)

    def extract_summary_vector(self, y: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """Extracts 1D global summary feature vector (mean + std across time frames).

        Ideal for traditional ML tabular classifiers (e.g. SVM, Random Forest, XGBoost).

        Args:
            y: Audio waveform array.
            sr: Audio sample rate.

        Returns:
            1D numpy array of shape (2 * total_feature_dim,).
        """
        matrix = self.extract_composite_matrix(y, sr)
        means = np.mean(matrix, axis=1)
        stds = np.std(matrix, axis=1)
        vector = np.concatenate([means, stds])
        return vector.astype(np.float32)

    def _check_feature_matrix(self, name: str, feature_matrix: np.ndarray) -> np.ndarray:
        """Validates feature output arrays for NaN, Inf, and non-empty status.

        Args:
            name: Feature display name.
            feature_matrix: Output numpy matrix.

        Returns:
            Validated float32 numpy matrix.
        """
        if feature_matrix is None or feature_matrix.size == 0:
            raise InvalidFeatureError(name, "Feature array is empty or None.")

        arr = np.asarray(feature_matrix, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise InvalidFeatureError(name, "Extracted feature matrix contains non-finite (NaN/Inf) values.")

        return arr
