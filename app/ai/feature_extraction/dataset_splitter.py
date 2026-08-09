"""Stratified Dataset Splitter Module.

Provides reproducible stratified train/validation/test splitting to prepare
extracted audio feature datasets for AI model training.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np

from app.ai.feature_extraction.exceptions import FeatureExtractionError
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


class StratifiedDatasetSplitter:
    """Splits dataset arrays into stratified train, validation, and test subsets."""

    def __init__(
        self,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initializes splitting ratios and random seed.

        Args:
            train_ratio: Fraction of dataset for training (default: 0.70).
            val_ratio: Fraction of dataset for validation (default: 0.15).
            test_ratio: Fraction of dataset for testing (default: 0.15).
            random_seed: Random seed for reproducibility (default: 42).
        """
        cfg = settings.feature_extraction
        self.train_ratio = train_ratio if train_ratio is not None else cfg.train_ratio
        self.val_ratio = val_ratio if val_ratio is not None else cfg.val_ratio
        self.test_ratio = test_ratio if test_ratio is not None else cfg.test_ratio
        self.random_seed = random_seed if random_seed is not None else cfg.random_seed

        # Validate ratios sum approximately to 1.0
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-3):
            raise FeatureExtractionError(
                f"Dataset split ratios must sum to 1.0. Got train={self.train_ratio}, "
                f"val={self.val_ratio}, test={self.test_ratio} (sum={total_ratio})."
            )

    def split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Performs stratified train/validation/test split on feature array X and labels y.

        Args:
            X: Feature array of shape (N, ...).
            y: Label array of shape (N,).

        Returns:
            Dictionary containing 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'.

        Raises:
            FeatureExtractionError: If array lengths mismatch or total samples are insufficient.
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        if len(X_arr) != len(y_arr):
            raise FeatureExtractionError(
                f"Length mismatch between features X ({len(X_arr)}) and labels y ({len(y_arr)})."
            )

        n_samples = len(y_arr)
        if n_samples == 0:
            raise FeatureExtractionError("Cannot split an empty dataset.")

        rng = np.random.RandomState(self.random_seed)
        unique_classes = np.unique(y_arr)

        train_indices = []
        val_indices = []
        test_indices = []

        # Perform per-class stratified sampling
        for cls in unique_classes:
            cls_indices = np.where(y_arr == cls)[0]
            rng.shuffle(cls_indices)
            n_cls = len(cls_indices)

            # Compute split counts for this class
            n_val = max(1 if n_cls >= 3 else 0, int(round(n_cls * self.val_ratio)))
            n_test = max(1 if n_cls >= 3 else 0, int(round(n_cls * self.test_ratio)))
            n_train = n_cls - n_val - n_test

            if n_train < 1 and n_cls > 0:
                n_train = max(1, n_cls - n_val - n_test)

            val_idx = cls_indices[:n_val]
            test_idx = cls_indices[n_val : n_val + n_test]
            train_idx = cls_indices[n_val + n_test :]

            val_indices.extend(val_idx)
            test_indices.extend(test_idx)
            train_indices.extend(train_idx)

        train_indices = np.array(train_indices, dtype=int)
        val_indices = np.array(val_indices, dtype=int)
        test_indices = np.array(test_indices, dtype=int)

        # Shuffle indices
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        rng.shuffle(test_indices)

        splits = {
            "X_train": X_arr[train_indices],
            "y_train": y_arr[train_indices],
            "X_val": X_arr[val_indices],
            "y_val": y_arr[val_indices],
            "X_test": X_arr[test_indices],
            "y_test": y_arr[test_indices],
        }

        logger.info(
            f"Stratified dataset split completed (Seed={self.random_seed}): "
            f"Train={len(splits['X_train'])}, Val={len(splits['X_val'])}, Test={len(splits['X_test'])}"
        )
        return splits
