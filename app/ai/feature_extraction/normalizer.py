"""Feature Normalizer Module.

Provides configurable Z-score or Min-Max feature normalization fitted on training datasets
and serialized for zero-data-leakage real-time inference processing.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from app.ai.feature_extraction.exceptions import FeatureExtractionError
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


class FeatureNormalizer:
    """Normalizes numerical feature arrays using fitted training parameters."""

    def __init__(
        self,
        normalization_type: str = None,
        enabled: bool = True,
        eps: float = 1e-8,
    ) -> None:
        """Initializes normalizer.

        Args:
            normalization_type: 'z_score' or 'min_max'. Defaults to settings.
            enabled: Toggle normalization execution.
            eps: Epsilon parameter to prevent division by zero.
        """
        self.normalization_type = (
            normalization_type or settings.feature_extraction.normalization_type
        ).lower()
        self.enabled = enabled if enabled is not None else settings.feature_extraction.enable_normalization
        self.eps = eps

        # Fitted statistics parameters
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Indicates whether normalizer statistics have been computed."""
        return self._is_fitted

    def fit(self, X: np.ndarray) -> "FeatureNormalizer":
        """Calculates scaling statistics from training data array.

        Args:
            X: Training dataset array of shape (N, ...).

        Returns:
            Fitted normalizer instance.
        """
        if not self.enabled:
            logger.info("FeatureNormalizer is disabled. Skipping fit.")
            return self

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.size == 0:
            raise FeatureExtractionError("Cannot fit FeatureNormalizer on an empty array.")

        # Compute statistics across samples axis (axis 0)
        if self.normalization_type == "z_score":
            self._mean = np.mean(X_arr, axis=0, keepdims=True)
            self._std = np.std(X_arr, axis=0, keepdims=True)
            logger.info(
                f"FeatureNormalizer fitted using Z-score (mean shape: {self._mean.shape}, std shape: {self._std.shape})"
            )
        elif self.normalization_type == "min_max":
            self._min = np.min(X_arr, axis=0, keepdims=True)
            self._max = np.max(X_arr, axis=0, keepdims=True)
            logger.info(
                f"FeatureNormalizer fitted using Min-Max (min shape: {self._min.shape}, max shape: {self._max.shape})"
            )
        else:
            raise FeatureExtractionError(
                f"Unsupported normalization type '{self.normalization_type}'. Expected 'z_score' or 'min_max'."
            )

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Applies fitted normalization scaling to input array.

        Args:
            X: Input feature array.

        Returns:
            Normalized numpy array of float32 type.
        """
        if not self.enabled:
            return np.asarray(X, dtype=np.float32)

        if not self._is_fitted:
            raise FeatureExtractionError("FeatureNormalizer must be fitted before calling transform().")

        X_arr = np.asarray(X, dtype=np.float32)

        if self.normalization_type == "z_score":
            normalized = (X_arr - self._mean) / (self._std + self.eps)
        elif self.normalization_type == "min_max":
            normalized = (X_arr - self._min) / ((self._max - self._min) + self.eps)
        else:
            normalized = X_arr

        return normalized.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits statistics on array X and returns normalized array.

        Args:
            X: Feature array.

        Returns:
            Normalized feature array.
        """
        return self.fit(X).transform(X)

    def save_params(self, filepath: Path) -> Path:
        """Saves scaler statistics to a JSON file.

        Args:
            filepath: Path to destination file (e.g., scaler_params.json).

        Returns:
            Path of saved JSON file.
        """
        dest_path = Path(filepath)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "normalization_type": self.normalization_type,
            "enabled": self.enabled,
            "eps": self.eps,
            "is_fitted": self._is_fitted,
            "mean": self._mean.tolist() if self._mean is not None else None,
            "std": self._std.tolist() if self._std is not None else None,
            "min": self._min.tolist() if self._min is not None else None,
            "max": self._max.tolist() if self._max is not None else None,
        }

        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info(f"Scaler parameters saved to {dest_path}")
        return dest_path

    @classmethod
    def load_params(cls, filepath: Path) -> "FeatureNormalizer":
        """Loads scaler parameters from a JSON file.

        Args:
            filepath: Path to scaler_params.json.

        Returns:
            Loaded FeatureNormalizer instance.
        """
        source_path = Path(filepath)
        if not source_path.exists():
            raise FeatureExtractionError(f"Scaler parameter file not found at '{filepath}'.")

        with open(source_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        normalizer = cls(
            normalization_type=payload["normalization_type"],
            enabled=payload["enabled"],
            eps=payload.get("eps", 1e-8),
        )
        normalizer._is_fitted = payload["is_fitted"]

        if payload.get("mean") is not None:
            normalizer._mean = np.array(payload["mean"], dtype=np.float32)
        if payload.get("std") is not None:
            normalizer._std = np.array(payload["std"], dtype=np.float32)
        if payload.get("min") is not None:
            normalizer._min = np.array(payload["min"], dtype=np.float32)
        if payload.get("max") is not None:
            normalizer._max = np.array(payload["max"], dtype=np.float32)

        return normalizer
