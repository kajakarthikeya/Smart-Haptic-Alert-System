"""Data Loading and Dataset Validation Subsystem for Model Training."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from app.ai.training.exceptions import (
    ConfigurationError,
    InvalidFeatureData,
    InvalidLabelData,
)
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingDataset:
    """Structured container holding validated train, validation, and test splits."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_to_id: Dict[str, int]
    id_to_class: Dict[int, str]
    class_weights: Optional[Dict[int, float]] = None

    @property
    def num_classes(self) -> int:
        return len(self.class_to_id)

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self.X_train.shape[1:]


class TrainingDataLoader:
    """Loads and validates extracted acoustic feature splits and class mappings."""

    def __init__(
        self,
        features_dir: Optional[Union[str, Path]] = None,
        expected_rows: int = 184,
        expected_cols: int = 173,
    ) -> None:
        """Initializes data loader with target paths and expected feature shapes.

        Args:
            features_dir: Path to directory containing dataset_splits.npz and class_names.json.
            expected_rows: Expected frequency/spectral bins dimension (184).
            expected_cols: Expected time frames dimension (173).
        """
        self.features_dir = Path(features_dir or settings.feature_extraction.features_dir)
        self.expected_rows = expected_rows
        self.expected_cols = expected_cols

    def load_dataset(
        self,
        splits_filename: str = "dataset_splits.npz",
        class_mapping_filename: str = "class_names.json",
        add_channel_dim: bool = True,
    ) -> TrainingDataset:
        """Loads and validates dataset splits and class mappings.

        Args:
            splits_filename: Filename of the .npz archive containing dataset splits.
            class_mapping_filename: Filename of the JSON class name mapping.
            add_channel_dim: If True, adds channel axis (N, H, W, 1) for 2D CNN compatibility.

        Returns:
            TrainingDataset instance containing validated splits and class mappings.

        Raises:
            ConfigurationError: If required files are missing.
            InvalidFeatureData: If features are malformed, empty, or contain NaNs/Infs.
            InvalidLabelData: If labels are out of bounds or do not match feature counts.
        """
        splits_path = self.features_dir / splits_filename
        mapping_path = self.features_dir / class_mapping_filename

        if not splits_path.exists():
            raise ConfigurationError(f"Dataset splits file not found at: {splits_path}")

        if not mapping_path.exists():
            raise ConfigurationError(f"Class mapping file not found at: {mapping_path}")

        # 1. Load class mapping
        class_to_id, id_to_class = self.load_and_validate_class_mapping(mapping_path)
        num_classes = len(class_to_id)

        # 2. Load .npz archive
        try:
            with np.load(splits_path) as data:
                required_keys = [
                    "X_composite_train", "y_train",
                    "X_composite_val", "y_val",
                    "X_composite_test", "y_test",
                ]
                missing_keys = [k for k in required_keys if k not in data]
                if missing_keys:
                    raise InvalidFeatureData(
                        f"Dataset archive '{splits_path.name}' is missing required keys: {missing_keys}"
                    )

                X_train = np.array(data["X_composite_train"], dtype=np.float32)
                y_train = np.array(data["y_train"], dtype=int)
                X_val = np.array(data["X_composite_val"], dtype=np.float32)
                y_val = np.array(data["y_val"], dtype=int)
                X_test = np.array(data["X_composite_test"], dtype=np.float32)
                y_test = np.array(data["y_test"], dtype=int)
        except Exception as exc:
            if isinstance(exc, (InvalidFeatureData, InvalidLabelData)):
                raise
            raise InvalidFeatureData(f"Failed to read dataset archive '{splits_path}': {str(exc)}") from exc

        # 3. Validate features
        self.validate_feature_array(X_train, "X_train")
        self.validate_feature_array(X_val, "X_val")
        self.validate_feature_array(X_test, "X_test")

        # 4. Validate labels
        self.validate_labels(y_train, len(X_train), num_classes, "y_train")
        self.validate_labels(y_val, len(X_val), num_classes, "y_val")
        self.validate_labels(y_test, len(X_test), num_classes, "y_test")

        # 5. Add channel dimension for 2D CNN (H, W) -> (H, W, 1)
        if add_channel_dim:
            X_train = np.expand_dims(X_train, axis=-1)
            X_val = np.expand_dims(X_val, axis=-1)
            X_test = np.expand_dims(X_test, axis=-1)

        # 6. Analyze class distribution and calculate balanced class weights
        class_weights = self.calculate_class_weights(y_train, num_classes)

        logger.info(
            f"Dataset loaded successfully: Train={len(X_train)}, Val={len(X_val)}, "
            f"Test={len(X_test)}, InputShape={X_train.shape[1:]}, Classes={num_classes}"
        )

        return TrainingDataset(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            class_to_id=class_to_id,
            id_to_class=id_to_class,
            class_weights=class_weights,
        )

    def validate_feature_array(self, arr: np.ndarray, split_name: str) -> None:
        """Validates feature array dimensions, NaN/Inf presence, and data integrity.

        Args:
            arr: Numpy array of features.
            split_name: Name of the split (for descriptive error logs).

        Raises:
            InvalidFeatureData: If features are invalid.
        """
        if not isinstance(arr, np.ndarray):
            raise InvalidFeatureData(f"Split '{split_name}' must be a numpy ndarray, got {type(arr)}.")

        if arr.size == 0 or len(arr) == 0:
            raise InvalidFeatureData(f"Feature array for split '{split_name}' is empty.")

        if arr.ndim != 3:
            raise InvalidFeatureData(
                f"Split '{split_name}' must have 3 dimensions (samples, rows, cols), got shape {arr.shape}."
            )

        samples, rows, cols = arr.shape
        if rows != self.expected_rows or cols != self.expected_cols:
            raise InvalidFeatureData(
                f"Feature matrix shape mismatch in '{split_name}': expected (*, {self.expected_rows}, {self.expected_cols}), "
                f"but got {arr.shape}."
            )

        nan_count = int(np.isnan(arr).sum())
        if nan_count > 0:
            raise InvalidFeatureData(
                f"Found {nan_count} NaN values in feature split '{split_name}'."
            )

        inf_count = int(np.isinf(arr).sum())
        if inf_count > 0:
            raise InvalidFeatureData(
                f"Found {inf_count} infinite values in feature split '{split_name}'."
            )

    def validate_labels(
        self,
        labels: np.ndarray,
        expected_count: int,
        num_classes: int,
        split_name: str,
    ) -> None:
        """Validates label array length, bounds, and values.

        Args:
            labels: Integer label array.
            expected_count: Expected number of samples matching features.
            num_classes: Number of distinct classes.
            split_name: Name of the split.

        Raises:
            InvalidLabelData: If labels are invalid.
        """
        if not isinstance(labels, np.ndarray):
            raise InvalidLabelData(f"Labels for '{split_name}' must be a numpy ndarray, got {type(labels)}.")

        if len(labels) != expected_count:
            raise InvalidLabelData(
                f"Label count mismatch in '{split_name}': features have {expected_count} samples, "
                f"but labels have {len(labels)} elements."
            )

        if labels.ndim != 1:
            raise InvalidLabelData(
                f"Labels for '{split_name}' must be a 1D vector, got shape {labels.shape}."
            )

        min_val = int(np.min(labels))
        max_val = int(np.max(labels))

        if min_val < 0 or max_val >= num_classes:
            raise InvalidLabelData(
                f"Labels for '{split_name}' contain out-of-bounds IDs: min={min_val}, max={max_val}. "
                f"Expected range is [0, {num_classes - 1}]."
            )

    def load_and_validate_class_mapping(
        self, mapping_path: Path
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Loads and validates class names JSON mapping.

        Args:
            mapping_path: Path to class_names.json.

        Returns:
            Tuple of (class_to_id, id_to_class).

        Raises:
            InvalidLabelData: If mapping structure is invalid.
        """
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping_data = json.load(f)
        except Exception as exc:
            raise InvalidLabelData(f"Failed to parse class mapping JSON from '{mapping_path}': {str(exc)}") from exc

        if "class_to_id" not in mapping_data or "id_to_class" not in mapping_data:
            raise InvalidLabelData(
                f"Class mapping '{mapping_path.name}' must contain 'class_to_id' and 'id_to_class' objects."
            )

        class_to_id = {k: int(v) for k, v in mapping_data["class_to_id"].items()}
        id_to_class = {int(k): str(v) for k, v in mapping_data["id_to_class"].items()}

        # Verify bijective correspondence
        for cname, cid in class_to_id.items():
            if id_to_class.get(cid) != cname:
                raise InvalidLabelData(
                    f"Inconsistent mapping between class_to_id and id_to_class for '{cname}' -> {cid}."
                )

        target_classes = set(settings.dataset.target_classes)
        mapped_classes = set(class_to_id.keys())
        if not target_classes.issubset(mapped_classes):
            missing = target_classes - mapped_classes
            raise InvalidLabelData(
                f"Class mapping missing required system target classes: {missing}"
            )

        return class_to_id, id_to_class

    def calculate_class_weights(
        self, y: np.ndarray, num_classes: int
    ) -> Dict[int, float]:
        """Calculates balanced class weights to compensate for class imbalance.

        Formula: w_c = N / (K * n_c)
        Where:
            N = total number of training samples
            K = total number of classes
            n_c = number of samples in class c

        Args:
            y: Training labels array.
            num_classes: Total number of classes.

        Returns:
            Dictionary mapping integer class ID to float weight.
        """
        total_samples = len(y)
        class_counts = np.bincount(y, minlength=num_classes)

        weights: Dict[int, float] = {}
        for c in range(num_classes):
            count = class_counts[c]
            if count > 0:
                weights[c] = float(total_samples / (num_classes * count))
            else:
                weights[c] = 1.0

        # Log class distribution
        dist_str = ", ".join([f"Class {c}: {class_counts[c]} (weight={weights[c]:.2f})" for c in range(num_classes)])
        logger.info(f"Class distribution analysis: {dist_str}")

        return weights
