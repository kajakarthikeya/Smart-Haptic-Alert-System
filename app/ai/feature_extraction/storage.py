"""Feature Storage & Metadata Generator Module.

Manages saving and loading extracted feature arrays (.npz / .npy),
label mappings (.json), and feature metadata reports (.json).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from app.ai.feature_extraction.exceptions import FeatureStorageError
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


class FeatureStorageManager:
    """Manages file storage and metadata generation for audio feature datasets."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """Initializes storage directory manager.

        Args:
            output_dir: Directory path for feature files (default: app/ai/features).
        """
        self.output_dir = Path(output_dir or settings.feature_extraction.features_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_dataset_splits(
        self, splits: Dict[str, np.ndarray], filename: str = "dataset_splits.npz"
    ) -> Path:
        """Saves train/validation/test dataset splits into a compressed .npz archive.

        Args:
            splits: Dictionary containing 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'.
            filename: Output filename.

        Returns:
            Path of saved .npz file.
        """
        filepath = self.output_dir / filename
        try:
            np.savez_compressed(filepath, **splits)
            logger.info(f"Saved dataset splits compressed archive to {filepath}")
            return filepath
        except Exception as exc:
            raise FeatureStorageError(str(filepath), f"Failed to save .npz dataset archive: {str(exc)}")

    def load_dataset_splits(self, filename: str = "dataset_splits.npz") -> Dict[str, np.ndarray]:
        """Loads dataset splits from a compressed .npz archive.

        Args:
            filename: Target .npz filename.

        Returns:
            Dictionary containing dataset numpy arrays.
        """
        filepath = self.output_dir / filename
        if not filepath.exists():
            raise FeatureStorageError(str(filepath), f"Dataset archive file does not exist at {filepath}.")
        try:
            with np.load(filepath) as archive:
                return {k: archive[k] for k in archive.files}
        except Exception as exc:
            raise FeatureStorageError(str(filepath), f"Failed to load .npz dataset archive: {str(exc)}")

    def save_feature_metadata(
        self,
        num_samples: int,
        feature_dimensions: Dict[str, Tuple[int, ...]],
        class_distribution: Dict[str, int],
        filename: str = "feature_metadata.json",
    ) -> Path:
        """Generates and saves feature metadata JSON file.

        Args:
            num_samples: Total audio clips processed.
            feature_dimensions: Map of feature names to their shape tuples.
            class_distribution: Count of samples per class name.
            filename: Metadata output JSON filename.

        Returns:
            Path of saved JSON metadata file.
        """
        cfg = settings.feature_extraction
        prep_cfg = settings.preprocessing

        metadata = {
            "feature_types": [
                "mfcc",
                "mel_spectrogram",
                "zero_crossing_rate",
                "spectral_centroid",
                "spectral_bandwidth",
                "spectral_rolloff",
                "chroma",
            ],
            "number_of_samples": num_samples,
            "feature_dimensions": {k: list(v) for k, v in feature_dimensions.items()},
            "class_distribution": class_distribution,
            "sample_rate": prep_cfg.target_sample_rate,
            "fft_size": cfg.n_fft,
            "hop_length": cfg.hop_length,
            "win_length": cfg.win_length or cfg.n_fft,
            "number_of_mfccs": cfg.n_mfcc,
            "number_of_mel_bands": cfg.n_mels,
            "number_of_chroma_bins": cfg.n_chroma,
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        filepath = self.output_dir / filename
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved feature metadata report to {filepath}")
            return filepath
        except Exception as exc:
            raise FeatureStorageError(str(filepath), f"Failed to write metadata JSON: {str(exc)}")

    def load_feature_metadata(self, filename: str = "feature_metadata.json") -> Dict[str, Any]:
        """Reads feature metadata JSON file.

        Args:
            filename: Target JSON filename.

        Returns:
            Metadata dictionary.
        """
        filepath = self.output_dir / filename
        if not filepath.exists():
            raise FeatureStorageError(str(filepath), f"Metadata file not found at {filepath}.")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            raise FeatureStorageError(str(filepath), f"Failed to read metadata JSON: {str(exc)}")
