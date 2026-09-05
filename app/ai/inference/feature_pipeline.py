"""
Real-Time Feature Pipeline Subsystem.

Adapts streaming raw audio windows into model-ready (1, 184, 173, 1) feature tensors
by directly reusing Phase 3 signal standardizers and Phase 4 Librosa feature extraction.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from config import Config, InferenceConfig, settings
from app.ai.preprocessing.audio_standardizer import AudioStandardizer
from app.ai.preprocessing.length_standardizer import LengthStandardizer
from app.ai.feature_extraction.feature_extractor import FeatureExtractor
from app.ai.inference.exceptions import FeaturePipelineError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RealtimeFeaturePipeline:
    """
    Transforms raw streaming or recorded audio arrays into normalized (1, 184, 173, 1) feature tensors.
    """

    def __init__(
        self,
        config: Optional[InferenceConfig] = None,
        standardizer: Optional[AudioStandardizer] = None,
        length_standardizer: Optional[LengthStandardizer] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
    ) -> None:
        self.config = config or getattr(settings, "inference", None)

        target_sr = (
            self.config.sample_rate
            if self.config
            else settings.preprocessing.target_sample_rate
        )
        duration_sec = (
            self.config.window_duration_sec
            if self.config
            else settings.preprocessing.target_duration_sec
        )

        self.sample_rate = int(target_sr)
        self.duration_sec = float(duration_sec)
        self.expected_samples = int(self.sample_rate * self.duration_sec)

        # Reuse existing Phase 3 and Phase 4 implementations
        self.standardizer = standardizer or AudioStandardizer(target_sample_rate=self.sample_rate)
        self.length_standardizer = length_standardizer or LengthStandardizer(
            target_duration_sec=self.duration_sec,
            target_sample_rate=self.sample_rate,
        )
        self.feature_extractor = feature_extractor or FeatureExtractor(sample_rate=self.sample_rate)

    def preprocess_signal(
        self,
        raw_waveform: Union[np.ndarray, List[float]],
        orig_sr: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Applies Phase 3 standardization to raw audio waveform:
        1. Mono conversion
        2. Resampling to 22050 Hz (if needed)
        3. Peak amplitude normalization to [-0.95, 0.95]
        4. Length standardization to 4.0s (88,200 samples) via trimming/zero-padding

        Args:
            raw_waveform: 1D or 2D audio samples.
            orig_sr: Input sampling rate; defaults to pipeline sample_rate.

        Returns:
            Tuple of (preprocessed_1d_array, elapsed_time_seconds).
        """
        t0 = time.perf_counter()
        try:
            arr = np.asarray(raw_waveform, dtype=np.float32)
            if arr.size == 0:
                raise FeaturePipelineError("Input audio waveform is empty.")

            # Flatten multi-channel to mono
            if arr.ndim > 1 and arr.shape[1] > 1:
                arr = np.mean(arr, axis=1)
            elif arr.ndim > 1:
                arr = arr.flatten()

            # Resample if needed
            sr = orig_sr or self.sample_rate
            if sr != self.sample_rate:
                raw_list = arr.tolist()
                resampled_list = self.standardizer.resample(raw_list, orig_sr=sr, target_sr=self.sample_rate)
                arr = np.array(resampled_list, dtype=np.float32)

            # Peak amplitude normalization
            raw_list = arr.tolist()
            normalized_list = self.standardizer.normalize_peak_amplitude(raw_list)

            # Length standardization (exact 88,200 samples)
            standardized_list = self.length_standardizer.standardize_length(normalized_list)
            preprocessed_y = np.array(standardized_list, dtype=np.float32)

            # Check finite values
            if np.isnan(preprocessed_y).any() or np.isinf(preprocessed_y).any():
                raise FeaturePipelineError("Preprocessed audio contains NaN or Infinite values.")

            elapsed = time.perf_counter() - t0
            return preprocessed_y, elapsed

        except Exception as exc:
            if isinstance(exc, FeaturePipelineError):
                raise
            raise FeaturePipelineError(f"Audio preprocessing failed: {exc}") from exc

    def extract_features(
        self,
        preprocessed_y: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Extracts composite 2D feature matrix using Phase 4 FeatureExtractor:
        Order: Mel Spectrogram (128) + MFCC (40) + ZCR (1) + Centroid (1) + Bandwidth (1) + Rolloff (1) + Chroma (12)
        Output shape: (1, 184, 173, 1)

        Args:
            preprocessed_y: Standardized 1D audio array of shape (88200,).

        Returns:
            Tuple of (model_input_tensor, elapsed_time_seconds).
        """
        t0 = time.perf_counter()
        try:
            # Extract composite 2D matrix (184, 173)
            matrix = self.feature_extractor.extract_composite_matrix(preprocessed_y, sr=self.sample_rate)

            if matrix.shape[0] != 184:
                raise FeaturePipelineError(
                    f"Composite feature matrix row mismatch: expected 184 rows, got {matrix.shape[0]}"
                )

            # Ensure non-empty time dimension
            if matrix.shape[1] == 0:
                raise FeaturePipelineError("Feature extraction produced zero time steps.")

            # Expand dims to batch and channel: (184, 173) -> (1, 184, 173, 1)
            model_input = np.expand_dims(matrix, axis=(0, -1)).astype(np.float32)

            # Check finite
            if np.isnan(model_input).any() or np.isinf(model_input).any():
                raise FeaturePipelineError("Extracted feature tensor contains NaN or Infinite values.")

            elapsed = time.perf_counter() - t0
            return model_input, elapsed

        except Exception as exc:
            if isinstance(exc, FeaturePipelineError):
                raise
            raise FeaturePipelineError(f"Feature extraction failed: {exc}") from exc

    def process(
        self,
        raw_waveform: Union[np.ndarray, List[float]],
        orig_sr: Optional[int] = None,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Complete end-to-end transformation:
        Raw Audio Window -> Standardized Audio -> Feature Tensor (1, 184, 173, 1).

        Returns:
            Tuple of (feature_tensor, preprocessing_time, extraction_time).
        """
        clean_audio, prep_time = self.preprocess_signal(raw_waveform, orig_sr)
        features, feat_time = self.extract_features(clean_audio)
        return features, prep_time, feat_time
