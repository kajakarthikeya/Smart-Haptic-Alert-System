"""
Prediction Analysis, Confidence Assessment, and Error Breakdown Subsystem.

Analyzes sample-level inference outputs, exports detailed predictions.csv,
computes confidence metrics, and categorizes confusion patterns.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from app.ai.evaluation.exceptions import ReportGenerationError
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


@dataclass(frozen=True)
class SamplePrediction:
    """Detailed prediction record for a single evaluation test sample."""
    sample_id: str
    sample_index: int
    actual_id: int
    actual_class: str
    predicted_id: int
    predicted_class: str
    confidence: float
    is_correct: bool
    probabilities: List[float]

    @property
    def true_label_id(self) -> int:
        return self.actual_id

    @property
    def true_class(self) -> str:
        return self.actual_class

    @property
    def predicted_label_id(self) -> int:
        return self.predicted_id


# Export alias
PredictionRecord = SamplePrediction


class PredictionAnalyzer:
    """Analyzes test set predictions, confidence statistics, and misclassification errors."""

    def __init__(
        self,
        class_names: Optional[Sequence[str]] = None,
        low_confidence_threshold: Optional[float] = None,
        config: Optional[Any] = None,
    ) -> None:
        """Initializes analyzer with class names, confidence thresholds, or config.

        Args:
            class_names: Ordered list of class strings matching integer IDs.
            low_confidence_threshold: Threshold below which a prediction is considered low-confidence.
            config: Optional EvaluationConfig instance.
        """
        self.config = config or getattr(settings, "evaluation", None)
        if class_names is not None:
            self.class_names = list(class_names)
        elif hasattr(settings, "dataset") and hasattr(settings.dataset, "target_classes"):
            self.class_names = list(settings.dataset.target_classes)
        else:
            self.class_names = ["ambulance", "car_horn", "fire_alarm", "doorbell", "dog_bark"]

        self.num_classes = len(self.class_names)

        if low_confidence_threshold is not None:
            self.low_confidence_threshold = float(low_confidence_threshold)
        elif self.config and hasattr(self.config, "low_confidence_threshold"):
            self.low_confidence_threshold = float(self.config.low_confidence_threshold)
        else:
            self.low_confidence_threshold = 0.50

    def analyze_predictions(
        self,
        y_true: np.ndarray,
        y_pred_or_probs: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
    ) -> List[SamplePrediction]:
        """Processes raw prediction probabilities and ground truth into structured analyses.

        Args:
            y_true: 1D array of actual integer labels.
            y_pred_or_probs: Either 1D predicted labels or 2D probability matrix.
            probabilities: 2D probability matrix if 3 arguments are passed.

        Returns:
            List of SamplePrediction dataclass instances.
        """
        y_true = np.asarray(y_true, dtype=int)
        if probabilities is not None:
            probs = np.asarray(probabilities, dtype=float)
            y_pred = np.asarray(y_pred_or_probs, dtype=int)
        else:
            probs = np.asarray(y_pred_or_probs, dtype=float)
            y_pred = np.argmax(probs, axis=-1).astype(int)

        num_samples = len(y_true)
        records: List[SamplePrediction] = []

        for idx in range(num_samples):
            actual_id = int(y_true[idx])
            pred_id = int(y_pred[idx])
            sample_probs = probs[idx]
            conf = float(sample_probs[pred_id])

            actual_cls = self.class_names[actual_id] if actual_id < len(self.class_names) else str(actual_id)
            pred_cls = self.class_names[pred_id] if pred_id < len(self.class_names) else str(pred_id)
            is_correct = bool(actual_id == pred_id)

            rec = SamplePrediction(
                sample_id=f"test_sample_{idx:03d}",
                sample_index=idx,
                actual_id=actual_id,
                actual_class=actual_cls,
                predicted_id=pred_id,
                predicted_class=pred_cls,
                confidence=round(conf, 4),
                is_correct=is_correct,
                probabilities=[round(float(p), 4) for p in sample_probs],
            )
            records.append(rec)

        return records

    def compute_confidence_summary(
        self,
        records: List[SamplePrediction],
    ) -> Dict[str, Any]:
        """Computes statistical confidence summary across all, correct, and misclassified predictions."""
        if not records:
            return {
                "total_samples": 0,
                "correct_count": 0,
                "misclassified_count": 0,
                "overall_mean": 0.0,
                "overall_std": 0.0,
                "overall_min": 0.0,
                "overall_max": 0.0,
                "correct_mean": 0.0,
                "misclassified_mean": 0.0,
                "low_confidence_threshold": self.low_confidence_threshold,
                "low_confidence_count": 0,
            }

        confs = np.array([r.confidence for r in records], dtype=float)
        correct_confs = np.array([r.confidence for r in records if r.is_correct], dtype=float)
        wrong_confs = np.array([r.confidence for r in records if not r.is_correct], dtype=float)
        low_confs = [r for r in records if r.confidence < self.low_confidence_threshold]

        return {
            "total_samples": len(records),
            "correct_count": len(correct_confs),
            "misclassified_count": len(wrong_confs),
            "overall_mean": round(float(np.mean(confs)), 4),
            "overall_std": round(float(np.std(confs)), 4),
            "overall_min": round(float(np.min(confs)), 4),
            "overall_max": round(float(np.max(confs)), 4),
            "correct_mean": round(float(np.mean(correct_confs)), 4) if len(correct_confs) > 0 else 0.0,
            "misclassified_mean": round(float(np.mean(wrong_confs)), 4) if len(wrong_confs) > 0 else 0.0,
            "low_confidence_threshold": self.low_confidence_threshold,
            "low_confidence_count": len(low_confs),
        }

    def find_misclassifications(
        self,
        records: List[SamplePrediction],
    ) -> List[Dict[str, Any]]:
        """Filters and formats all misclassified samples."""
        misclassified = []
        for r in records:
            if not r.is_correct:
                top_probs = {
                    self.class_names[i]: float(r.probabilities[i])
                    for i in range(min(len(self.class_names), len(r.probabilities)))
                }
                misclassified.append({
                    "sample_id": r.sample_id,
                    "sample_index": r.sample_index,
                    "true_class": r.actual_class,
                    "true_label_id": r.actual_id,
                    "predicted_class": r.predicted_class,
                    "predicted_label_id": r.predicted_id,
                    "confidence": r.confidence,
                    "true_class_probability": float(r.probabilities[r.actual_id]) if r.actual_id < len(r.probabilities) else 0.0,
                    "top_probabilities": top_probs,
                })
        return misclassified

    def find_low_confidence(
        self,
        records: List[SamplePrediction],
        threshold: Optional[float] = None,
    ) -> List[SamplePrediction]:
        """Returns predictions with confidence lower than threshold."""
        thresh = threshold if threshold is not None else self.low_confidence_threshold
        return [r for r in records if r.confidence < thresh]

    def export_predictions_csv(
        self,
        records: List[SamplePrediction],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Exports sample predictions to CSV file format.

        Args:
            records: List of SamplePrediction records.
            output_path: Optional destination CSV filepath.

        Returns:
            Path to exported CSV file.

        Raises:
            ReportGenerationError: If file writing fails.
        """
        if output_path is not None:
            target = Path(output_path)
        elif self.config and hasattr(self.config, "output_dir"):
            target = Path(self.config.output_dir) / getattr(self.config, "predictions_csv_filename", "predictions.csv")
        else:
            target = Path("app/outputs/model_evaluation/predictions.csv")

        target.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "sample_id",
            "sample_index",
            "true_label_id",
            "true_class",
            "predicted_label_id",
            "predicted_class",
            "confidence",
            "is_correct",
        ] + [f"prob_{cls}" for cls in self.class_names]

        try:
            with open(target, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in records:
                    row: Dict[str, Any] = {
                        "sample_id": r.sample_id,
                        "sample_index": r.sample_index,
                        "true_label_id": r.actual_id,
                        "true_class": r.actual_class,
                        "predicted_label_id": r.predicted_id,
                        "predicted_class": r.predicted_class,
                        "confidence": r.confidence,
                        "is_correct": r.is_correct,
                    }
                    for cid, cname in enumerate(self.class_names):
                        row[f"prob_{cname}"] = r.probabilities[cid] if cid < len(r.probabilities) else 0.0
                    writer.writerow(row)

            logger.info("Exported prediction records CSV to: %s", target)
            return target
        except Exception as exc:
            raise ReportGenerationError(f"Failed to export predictions CSV to '{target}': {str(exc)}") from exc
