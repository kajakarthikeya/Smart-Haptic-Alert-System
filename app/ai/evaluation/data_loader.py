"""Evaluation Data Loading and Validation Subsystem.

Loads unseen test features, labels, class mappings, and trained model artifacts
with comprehensive dimension, NaN/Inf, and label integrity validation.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from app.ai.evaluation.exceptions import (
    InvalidEvaluationData,
    ModelLoadError,
)
from app.ai.models.cnn_classifier import CNNSoundClassifier
from app.utils.logger import get_logger
from config import EvaluationConfig, settings

logger = get_logger(__name__)


@dataclass(frozen=True)
class EvaluationData:
    """Structured container holding validated test features, labels, model, and class metadata."""
    X_test: np.ndarray
    y_test: np.ndarray
    model: CNNSoundClassifier
    class_to_id: Dict[str, int]
    id_to_class: Dict[int, str]
    class_names: List[str]

    @property
    def num_samples(self) -> int:
        return len(self.X_test)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


class EvaluationDataLoader:
    """Loads and validates test dataset splits, class mappings, and trained model."""

    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        features_dir: Optional[Union[str, Path]] = None,
        model_path: Optional[Union[str, Path]] = None,
        expected_rows: int = 184,
        expected_cols: int = 173,
    ) -> None:
        """Initializes loader with feature paths and model paths.

        Args:
            config: Optional EvaluationConfig instance.
            features_dir: Directory containing dataset_splits.npz and class_names.json.
            model_path: Path to serialized trained model (.keras).
            expected_rows: Expected acoustic feature frequency/spectral rows (184).
            expected_cols: Expected time frames (173).
        """
        self.config = config or getattr(settings, "evaluation", None)
        default_feat_dir = (
            Path(self.config.features_path).parent
            if self.config and hasattr(self.config, "features_path")
            else settings.feature_extraction.features_dir
        )
        default_model_path = (
            Path(self.config.model_path)
            if self.config and hasattr(self.config, "model_path")
            else (settings.training.model_save_dir / settings.model.best_model_filename)
        )

        self.features_dir = Path(features_dir or default_feat_dir)
        self.model_path = Path(model_path or default_model_path)
        self.expected_rows = expected_rows
        self.expected_cols = expected_cols

    def load(
        self,
        features_path: Optional[Union[str, Path]] = None,
        model_path: Optional[Union[str, Path]] = None,
        class_names_path: Optional[Union[str, Path]] = None,
        add_channel_dim: bool = True,
    ) -> EvaluationData:
        """Convenience method to load evaluation data from specific file paths or defaults."""
        raw_feat = features_path if features_path else (
            self.config.features_path if self.config and hasattr(self.config, "features_path")
            else self.features_dir / "dataset_splits.npz"
        )
        feat_p = Path(raw_feat)

        raw_mod = model_path if model_path else (
            self.config.model_path if self.config and hasattr(self.config, "model_path")
            else self.model_path
        )
        mod_p = Path(raw_mod)

        raw_mapping = class_names_path if class_names_path else (
            self.config.class_names_path if self.config and hasattr(self.config, "class_names_path")
            else self.features_dir / "class_names.json"
        )
        mapping_p = Path(raw_mapping)

        if not feat_p.exists():
            raise InvalidEvaluationData(f"Test features file not found at: {feat_p}")
        if not mapping_p.exists():
            raise InvalidEvaluationData(f"Class mapping file not found at: {mapping_p}")

        # Temporarily set model path if passed
        orig_model_path = self.model_path
        try:
            self.model_path = mod_p
            return self._load_from_paths(feat_p, mapping_p, add_channel_dim=add_channel_dim)
        finally:
            self.model_path = orig_model_path

    def load_evaluation_data(
        self,
        splits_filename: str = "dataset_splits.npz",
        class_mapping_filename: str = "class_names.json",
        add_channel_dim: bool = True,
    ) -> EvaluationData:
        """Loads and thoroughly validates evaluation test data and trained model."""
        splits_path = self.features_dir / splits_filename
        mapping_path = self.features_dir / class_mapping_filename

        if not splits_path.exists():
            raise InvalidEvaluationData(f"Test dataset archive not found at: {splits_path}")
        if not mapping_path.exists():
            raise InvalidEvaluationData(f"Class mapping file not found at: {mapping_path}")

        return self._load_from_paths(splits_path, mapping_path, add_channel_dim=add_channel_dim)

    def _load_from_paths(
        self,
        splits_path: Path,
        mapping_path: Path,
        add_channel_dim: bool = True,
    ) -> EvaluationData:
        """Internal helper to load and validate from concrete file paths."""
        # 1. Load and validate class mapping
        class_to_id, id_to_class, class_names = self._load_and_validate_mapping(mapping_path)
        num_classes = len(class_names)

        # 2. Load test split from .npz
        try:
            with np.load(splits_path) as data:
                if "X_composite_test" not in data or "y_test" not in data:
                    raise InvalidEvaluationData(
                        f"Dataset archive '{splits_path.name}' missing 'X_composite_test' or 'y_test' keys."
                    )
                X_test = np.array(data["X_composite_test"], dtype=np.float32)
                y_test = np.array(data["y_test"], dtype=int)
        except Exception as exc:
            if isinstance(exc, InvalidEvaluationData):
                raise
            raise InvalidEvaluationData(f"Failed to load dataset archive '{splits_path}': {str(exc)}") from exc

        # 3. Validate features
        self._validate_features(X_test)

        # 4. Validate labels
        self._validate_labels(y_test, len(X_test), num_classes)

        # 5. Expand channel dimension if requested (N, 184, 173) -> (N, 184, 173, 1)
        if add_channel_dim and X_test.ndim == 3:
            X_test = np.expand_dims(X_test, axis=-1)

        # 6. Load trained model
        classifier = self._load_model(num_classes, class_names, X_test.shape[1:])

        logger.info(
            f"Evaluation data loaded successfully: {len(X_test)} test samples, "
            f"Input Shape={X_test.shape[1:]}, Classes={num_classes}"
        )

        return EvaluationData(
            X_test=X_test,
            y_test=y_test,
            model=classifier,
            class_to_id=class_to_id,
            id_to_class=id_to_class,
            class_names=class_names,
        )

    def _validate_features(self, arr: np.ndarray) -> None:
        """Validates feature array emptiness, dimensionality, and finite values."""
        if not isinstance(arr, np.ndarray):
            raise InvalidEvaluationData(f"Test features must be a numpy ndarray, got {type(arr)}.")

        if arr.size == 0 or len(arr) == 0:
            raise InvalidEvaluationData("Test feature dataset is empty (0 samples).")

        if arr.ndim != 3 and arr.ndim != 4:
            raise InvalidEvaluationData(
                f"Test features must have 3 dimensions (N, {self.expected_rows}, {self.expected_cols}), "
                f"got shape {arr.shape}."
            )

        rows = arr.shape[1]
        cols = arr.shape[2]
        if rows != self.expected_rows or cols != self.expected_cols:
            raise InvalidEvaluationData(
                f"Feature matrix shape mismatch: expected (*, {self.expected_rows}, {self.expected_cols}), "
                f"got {arr.shape}."
            )

        nan_count = int(np.isnan(arr).sum())
        if nan_count > 0:
            raise InvalidEvaluationData(f"Test features contain {nan_count} NaN values.")

        inf_count = int(np.isinf(arr).sum())
        if inf_count > 0:
            raise InvalidEvaluationData(f"Test features contain {inf_count} infinite values.")

    def _validate_labels(self, labels: np.ndarray, expected_count: int, num_classes: int) -> None:
        """Validates test label array dimensions, counts, and ID ranges."""
        if not isinstance(labels, np.ndarray):
            raise InvalidEvaluationData(f"Test labels must be a numpy ndarray, got {type(labels)}.")

        if len(labels) == 0:
            raise InvalidEvaluationData("Test label vector is empty (0 labels).")

        if len(labels) != expected_count:
            raise InvalidEvaluationData(
                f"Sample count mismatch: features contain {expected_count} samples, "
                f"but labels contain {len(labels)} elements."
            )

        if labels.ndim != 1:
            raise InvalidEvaluationData(f"Test labels must be a 1D vector, got shape {labels.shape}.")

        min_val = int(np.min(labels))
        max_val = int(np.max(labels))

        if min_val < 0 or max_val >= num_classes:
            raise InvalidEvaluationData(
                f"Test labels contain out-of-bounds IDs: min={min_val}, max={max_val}. "
                f"Expected range is [0, {num_classes - 1}]."
            )

    def _load_and_validate_mapping(
        self, mapping_path: Path
    ) -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
        """Loads and verifies class mapping JSON file."""
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            raise InvalidEvaluationData(f"Failed to parse class mapping JSON '{mapping_path}': {str(exc)}") from exc

        if "class_to_id" not in data or "id_to_class" not in data:
            raise InvalidEvaluationData(f"Class mapping '{mapping_path.name}' missing required mapping keys.")

        class_to_id = {str(k): int(v) for k, v in data["class_to_id"].items()}
        id_to_class = {int(k): str(v) for k, v in data["id_to_class"].items()}

        # Enforce canonical ordering [0, 1, ..., num_classes - 1]
        num_classes = len(class_to_id)
        class_names = [id_to_class[i] for i in range(num_classes)]

        target_classes = set(settings.dataset.target_classes)
        mapped_classes = set(class_to_id.keys())
        if not target_classes.issubset(mapped_classes):
            missing = target_classes - mapped_classes
            raise InvalidEvaluationData(f"Class mapping is missing target classes: {missing}")

        return class_to_id, id_to_class, class_names

    def _load_model(
        self, num_classes: int, class_names: List[str], input_shape: Tuple[int, ...]
    ) -> CNNSoundClassifier:
        """Loads serialized trained model artifact."""
        if not self.model_path.exists():
            raise ModelLoadError(f"Trained model checkpoint not found at: {self.model_path}")

        classifier = CNNSoundClassifier(
            input_shape=input_shape,
            num_classes=num_classes,
            class_labels=class_names,
        )

        load_ok = classifier.load_model(self.model_path)
        if not load_ok or not classifier.is_loaded:
            raise ModelLoadError(f"Failed to load trained model weights from: {self.model_path}")

        return classifier
