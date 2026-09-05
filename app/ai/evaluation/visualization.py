"""
Visualizations for AI Model Evaluation in the Smart Haptic Alert System.

This module generates publication-grade diagnostic plots for model performance:
1. Raw confusion matrix heatmap (confusion_matrix.png)
2. Normalized confusion matrix heatmap (normalized_confusion_matrix.png)
3. Per-class precision, recall, and F1-score comparison bar chart (metrics_comparison.png)
4. Confidence score distribution histogram with correct/incorrect breakdown (confidence_distribution.png)
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt

from config import Config, EvaluationConfig
from app.ai.evaluation.exceptions import VisualizationError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationVisualizer:
    """Generates and saves diagnostic evaluation plots."""

    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.config = config or getattr(Config, "evaluation", None)
        self.class_names = list(class_names) if class_names else [
            "ambulance", "car_horn", "fire_alarm", "doorbell", "dog_bark"
        ]
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_confusion_matrix(
        self,
        matrix: np.ndarray,
        normalized: bool = False,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Plots a confusion matrix heatmap (raw counts or recall-normalized).

        Args:
            matrix: 2D numpy array of shape (num_classes, num_classes).
            normalized: If True, formats cells as percentages/proportions.
            filename: Target filename; defaults from config if not provided.

        Returns:
            Path to the saved PNG image.
        """
        try:
            matrix = np.asarray(matrix, dtype=float if normalized else int)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise VisualizationError(
                    f"Confusion matrix must be square 2D array, got shape {matrix.shape}"
                )

            num_classes = matrix.shape[0]
            labels = self.class_names[:num_classes]

            fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
            cmap = plt.cm.Blues

            im = ax.imshow(matrix, interpolation="nearest", cmap=cmap)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=10)

            title = (
                "Normalized Confusion Matrix (Recall / Class Accuracy)"
                if normalized
                else "Evaluation Confusion Matrix (Raw Counts)"
            )
            ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

            tick_marks = np.arange(num_classes)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
            ax.set_yticklabels(labels, fontsize=10)

            ax.set_ylabel("True Ground Truth Label", fontsize=11, fontweight="semibold")
            ax.set_xlabel("Predicted Label", fontsize=11, fontweight="semibold")

            # Annotate cell numbers
            thresh = matrix.max() / 2.0 if matrix.max() > 0 else 1.0
            for i in range(num_classes):
                for j in range(num_classes):
                    val = matrix[i, j]
                    text_val = f"{val:.2f}" if normalized else f"{int(val)}"
                    color = "white" if val > thresh else "black"
                    ax.text(
                        j, i, text_val,
                        ha="center", va="center",
                        color=color, fontsize=11, fontweight="bold"
                    )

            plt.tight_layout()

            default_name = (
                self.config.normalized_confusion_matrix_filename
                if normalized
                else self.config.confusion_matrix_filename
            )
            target_path = self.output_dir / (filename or default_name)
            fig.savefig(target_path, bbox_inches="tight")
            plt.close(fig)

            logger.info("Saved confusion matrix plot to: %s", target_path)
            return target_path

        except Exception as exc:
            if not isinstance(exc, VisualizationError):
                raise VisualizationError(f"Failed to plot confusion matrix: {exc}") from exc
            raise

    def plot_metrics_comparison(
        self,
        per_class_metrics: Dict[str, Dict[str, float]],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Plots a grouped bar chart comparing Precision, Recall, and F1-score across classes.

        Args:
            per_class_metrics: Dictionary mapping class names to metric dicts.
            filename: Target filename; defaults from config.

        Returns:
            Path to saved PNG image.
        """
        try:
            classes = [c for c in self.class_names if c in per_class_metrics]
            if not classes:
                classes = list(per_class_metrics.keys())

            precisions = [per_class_metrics[c].get("precision", 0.0) for c in classes]
            recalls = [per_class_metrics[c].get("recall", 0.0) for c in classes]
            f1s = [per_class_metrics[c].get("f1_score", 0.0) for c in classes]

            x = np.arange(len(classes))
            width = 0.25

            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

            rects1 = ax.bar(x - width, precisions, width, label="Precision", color="#3b82f6", alpha=0.9)
            rects2 = ax.bar(x, recalls, width, label="Recall", color="#10b981", alpha=0.9)
            rects3 = ax.bar(x + width, f1s, width, label="F1-Score", color="#8b5cf6", alpha=0.9)

            ax.set_ylabel("Score", fontsize=11, fontweight="semibold")
            ax.set_title("Per-Class Classification Metrics Comparison", fontsize=13, fontweight="bold", pad=14)
            ax.set_xticks(x)
            ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=10)
            ax.set_ylim(0.0, 1.15)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.legend(loc="upper right", framealpha=0.9)

            # Value labels above bars
            def _autolabel(rects):
                for rect in rects:
                    h = rect.get_height()
                    ax.annotate(
                        f"{h:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=8, fontweight="semibold"
                    )

            _autolabel(rects1)
            _autolabel(rects2)
            _autolabel(rects3)

            plt.tight_layout()

            target_path = self.output_dir / (filename or self.config.metrics_comparison_filename)
            fig.savefig(target_path, bbox_inches="tight")
            plt.close(fig)

            logger.info("Saved metrics comparison plot to: %s", target_path)
            return target_path

        except Exception as exc:
            raise VisualizationError(f"Failed to plot metrics comparison: {exc}") from exc

    def plot_confidence_distribution(
        self,
        confidences: Sequence[float],
        is_correct_list: Sequence[bool],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Plots prediction confidence distribution, distinguishing correct vs misclassified samples.

        Args:
            confidences: Array of winning softmax probabilities [0, 1].
            is_correct_list: Boolean array indicating whether prediction matched ground truth.
            filename: Target filename; defaults from config.

        Returns:
            Path to saved PNG image.
        """
        try:
            confs = np.asarray(confidences, dtype=float)
            corrects = np.asarray(is_correct_list, dtype=bool)

            fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

            bins = np.linspace(0.0, 1.0, 11)

            correct_confs = confs[corrects]
            wrong_confs = confs[~corrects]

            ax.hist(
                [correct_confs, wrong_confs],
                bins=bins,
                label=[f"Correct ({len(correct_confs)})", f"Misclassified ({len(wrong_confs)})"],
                color=["#10b981", "#ef4444"],
                stacked=True,
                edgecolor="black",
                alpha=0.85,
            )

            # Threshold line
            thresh = self.config.low_confidence_threshold
            ax.axvline(
                thresh,
                color="#f59e0b",
                linestyle="--",
                linewidth=1.5,
                label=f"Low Confidence Threshold ({thresh:.2f})",
            )

            ax.set_xlabel("Prediction Confidence (Softmax Probability)", fontsize=11, fontweight="semibold")
            ax.set_ylabel("Sample Count", fontsize=11, fontweight="semibold")
            ax.set_title("Prediction Confidence Score Distribution", fontsize=13, fontweight="bold", pad=14)
            ax.set_xlim(0.0, 1.05)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.legend(loc="upper left", framealpha=0.9)

            plt.tight_layout()

            target_path = self.output_dir / (filename or self.config.confidence_distribution_filename)
            fig.savefig(target_path, bbox_inches="tight")
            plt.close(fig)

            logger.info("Saved confidence distribution plot to: %s", target_path)
            return target_path

        except Exception as exc:
            raise VisualizationError(f"Failed to plot confidence distribution: {exc}") from exc
