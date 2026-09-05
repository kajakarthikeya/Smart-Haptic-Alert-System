"""
Comprehensive Multiclass Evaluation Metrics Calculator.

Computes overall accuracy, cross-entropy test loss, per-class precision/recall/F1/support,
and macro/weighted averages using scikit-learn.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    precision_recall_fscore_support,
)

from app.ai.evaluation.exceptions import MetricCalculationError
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


@dataclass(frozen=True)
class PerClassMetric:
    """Detailed performance metrics for an individual class."""
    class_name: str
    class_id: int
    precision: float
    recall: float
    f1_score: float
    support: int
    accuracy: float  # Binary one-vs-rest classification accuracy


@dataclass(frozen=True)
class EvaluationSummaryMetrics:
    """Comprehensive evaluation metrics container for model assessment."""
    test_accuracy: float
    test_loss: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    per_class_metrics: Dict[str, PerClassMetric]
    num_samples: int
    num_classes: int


class EvaluationMetricsCalculator:
    """Calculates multiclass evaluation metrics, losses, and per-class statistics."""

    def __init__(
        self,
        class_names: Optional[Sequence[str]] = None,
        config: Optional[Any] = None,
    ) -> None:
        """Initializes metrics calculator with ordered class names or config.

        Args:
            class_names: List of class names ordered by integer class ID.
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

    def compute_categorical_loss(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
    ) -> float:
        """Computes categorical cross-entropy loss with numerical stability clipping."""
        y_true = np.asarray(y_true, dtype=int)
        probs = np.asarray(probabilities, dtype=float)

        if probs.ndim != 2 or probs.shape[0] != len(y_true):
            raise MetricCalculationError(
                f"Probability array shape {probs.shape} incompatible with {len(y_true)} true labels."
            )

        try:
            num_classes = probs.shape[1]
            labels_list = list(range(num_classes))
            clipped = np.clip(probs, 1e-15, 1.0 - 1e-15)
            clipped = clipped / clipped.sum(axis=1, keepdims=True)
            return float(log_loss(y_true, clipped, labels=labels_list))
        except Exception as exc:
            raise MetricCalculationError(f"Failed to calculate cross-entropy loss: {exc}") from exc

    def compute_summary_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Computes overall accuracy, loss, macro and weighted averages."""
        if len(y_true) != len(y_pred):
            raise MetricCalculationError(f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})")
        if len(y_true) == 0:
            raise MetricCalculationError("Cannot calculate metrics on empty label sequences.")

        labels_list = list(range(self.num_classes))
        acc = float(accuracy_score(y_true, y_pred))

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels_list, average="macro", zero_division=0.0
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels_list, average="weighted", zero_division=0.0
        )

        loss_val = (
            self.compute_categorical_loss(y_true, probabilities)
            if probabilities is not None
            else 0.0
        )

        return {
            "accuracy": round(acc, 4),
            "test_loss": round(loss_val, 4),
            "macro_avg": {
                "precision": round(float(macro_p), 4),
                "recall": round(float(macro_r), 4),
                "f1_score": round(float(macro_f1), 4),
            },
            "weighted_avg": {
                "precision": round(float(weighted_p), 4),
                "recall": round(float(weighted_r), 4),
                "f1_score": round(float(weighted_f1), 4),
            },
        }

    def compute_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Computes per-class precision, recall, F1-score, support, and OvR accuracy."""
        labels_list = list(range(self.num_classes))
        precisions, recalls, f1s, supports = precision_recall_fscore_support(
            y_true, y_pred, labels=labels_list, average=None, zero_division=0.0
        )

        per_class: Dict[str, Dict[str, Any]] = {}
        for cid, cname in enumerate(self.class_names):
            tp = int(np.sum((y_true == cid) & (y_pred == cid)))
            tn = int(np.sum((y_true != cid) & (y_pred != cid)))
            ovr_acc = float((tp + tn) / len(y_true)) if len(y_true) > 0 else 0.0

            per_class[cname] = {
                "class_id": cid,
                "precision": round(float(precisions[cid]), 4),
                "recall": round(float(recalls[cid]), 4),
                "f1_score": round(float(f1s[cid]), 4),
                "support": int(supports[cid]),
                "accuracy": round(ovr_acc, 4),
            }

        return per_class

    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        """Generates scikit-learn classification report as a dictionary."""
        labels_list = list(range(self.num_classes))
        return classification_report(
            y_true,
            y_pred,
            labels=labels_list,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0.0,
        )

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
    ) -> EvaluationSummaryMetrics:
        """Computes comprehensive multiclass evaluation metrics returning EvaluationSummaryMetrics dataclass."""
        summary = self.compute_summary_metrics(y_true, y_pred, probabilities)
        per_class_raw = self.compute_per_class_metrics(y_true, y_pred, probabilities)

        per_class_dataclasses = {
            cname: PerClassMetric(
                class_name=cname,
                class_id=data["class_id"],
                precision=data["precision"],
                recall=data["recall"],
                f1_score=data["f1_score"],
                support=data["support"],
                accuracy=data["accuracy"],
            )
            for cname, data in per_class_raw.items()
        }

        return EvaluationSummaryMetrics(
            test_accuracy=summary["accuracy"],
            test_loss=summary["test_loss"],
            macro_precision=summary["macro_avg"]["precision"],
            macro_recall=summary["macro_avg"]["recall"],
            macro_f1=summary["macro_avg"]["f1_score"],
            weighted_precision=summary["weighted_avg"]["precision"],
            weighted_recall=summary["weighted_avg"]["recall"],
            weighted_f1=summary["weighted_avg"]["f1_score"],
            per_class_metrics=per_class_dataclasses,
            num_samples=len(y_true),
            num_classes=self.num_classes,
        )
