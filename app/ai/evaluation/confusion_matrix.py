"""
Confusion Matrix Generation Subsystem.

Calculates raw and normalized 5x5 multiclass confusion matrices with strict
canonical class ordering.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix

from app.ai.evaluation.exceptions import MetricCalculationError
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


class ConfusionMatrixGenerator:
    """Computes raw and normalized confusion matrices for multiclass evaluation."""

    def __init__(
        self,
        class_names: Optional[Sequence[str]] = None,
        config: Optional[Any] = None,
    ) -> None:
        """Initializes confusion matrix generator with ordered class labels or config.

        Args:
            class_names: List of class names in ascending ID order.
            config: Optional configuration object.
        """
        self.config = config or getattr(settings, "evaluation", None)
        if class_names is not None:
            self.class_names = list(class_names)
        elif hasattr(settings, "dataset") and hasattr(settings.dataset, "target_classes"):
            self.class_names = list(settings.dataset.target_classes)
        else:
            self.class_names = ["ambulance", "car_horn", "fire_alarm", "doorbell", "dog_bark"]

        self.num_classes = len(self.class_names)

    def generate_matrices(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Generates raw count matrix, normalized matrix, and structured dictionary."""
        try:
            labels_list = list(range(self.num_classes))

            # 1. Compute raw count 5x5 confusion matrix
            cm_raw = confusion_matrix(
                y_true,
                y_pred,
                labels=labels_list,
            )

            # Assert shape is strictly (num_classes, num_classes)
            if cm_raw.shape != (self.num_classes, self.num_classes):
                raise MetricCalculationError(
                    f"Expected confusion matrix shape ({self.num_classes}, {self.num_classes}), "
                    f"got {cm_raw.shape}."
                )

            # 2. Compute normalized (recall-based: row sum = 1.0) matrix
            row_sums = cm_raw.sum(axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                cm_norm = np.where(row_sums > 0, cm_raw.astype(np.float64) / row_sums, 0.0)

            cm_norm = np.nan_to_num(cm_norm, nan=0.0, posinf=0.0, neginf=0.0)

            # 3. Create structured dictionary mapping
            structured: Dict[str, Dict[str, int]] = {}
            structured_norm: Dict[str, Dict[str, float]] = {}

            for r_idx, actual_cls in enumerate(self.class_names):
                structured[actual_cls] = {}
                structured_norm[actual_cls] = {}
                for c_idx, pred_cls in enumerate(self.class_names):
                    structured[actual_cls][pred_cls] = int(cm_raw[r_idx, c_idx])
                    structured_norm[actual_cls][pred_cls] = round(float(cm_norm[r_idx, c_idx]), 4)

            payload = {
                "raw_matrix": cm_raw.tolist(),
                "normalized_matrix": [[round(v, 4) for v in row] for row in cm_norm.tolist()],
                "structured_counts": structured,
                "structured_normalized": structured_norm,
                "class_order": self.class_names,
            }

            logger.info("Generated %dx%d confusion matrix.", self.num_classes, self.num_classes)
            return cm_raw, cm_norm, payload

        except Exception as exc:
            if isinstance(exc, MetricCalculationError):
                raise
            raise MetricCalculationError(f"Failed to compute confusion matrix: {str(exc)}") from exc

    def generate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Generates raw and normalized matrices packaged in a dictionary."""
        cm_raw, cm_norm, payload = self.generate_matrices(y_true, y_pred)
        return {
            "raw_matrix": cm_raw,
            "normalized_matrix": cm_norm,
            "classes": self.class_names,
            "payload": payload,
        }
