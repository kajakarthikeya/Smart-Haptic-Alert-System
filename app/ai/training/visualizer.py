"""Training Metrics Visualization Subsystem.

Renders high-resolution loss and accuracy progression curves over training epochs
for model performance evaluation and overfitting diagnosis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless server execution
import matplotlib.pyplot as plt

from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


class TrainingVisualizer:
    """Renders and saves training and validation performance charts."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """Initializes visualizer with output directory.

        Args:
            output_dir: Directory where generated plots will be saved.
        """
        self.output_dir = Path(output_dir or settings.training.training_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Curated aesthetic color palette
        self.palette = {
            "train_acc": "#2563EB",  # Royal Blue
            "val_acc": "#059669",    # Emerald Green
            "train_loss": "#DC2626", # Deep Red
            "val_loss": "#D97706",   # Amber / Orange
            "grid": "#E2E8F0",       # Slate 200
            "text": "#1E293B",       # Slate 800
        }

    def plot_accuracy_curve(
        self,
        history: Dict[str, List[float]],
        filename: str = "accuracy_curve.png",
    ) -> Path:
        """Generates Training vs Validation Accuracy vs Epoch curve.

        Args:
            history: Dictionary containing 'accuracy' and 'val_accuracy' sequences.
            filename: Target image filename.

        Returns:
            Path to saved image file.
        """
        output_path = self.output_dir / filename
        train_acc = history.get("accuracy", [])
        val_acc = history.get("val_accuracy", [])
        epochs = range(1, len(train_acc) + 1)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.plot(epochs, train_acc, color=self.palette["train_acc"], marker="o", linewidth=2.2, label="Training Accuracy")
        if val_acc:
            ax.plot(epochs, val_acc, color=self.palette["val_acc"], marker="s", linestyle="--", linewidth=2.2, label="Validation Accuracy")

        ax.set_title("Training & Validation Accuracy vs Epoch", fontsize=13, fontweight="bold", color=self.palette["text"], pad=12)
        ax.set_xlabel("Epoch", fontsize=11, fontweight="medium", color=self.palette["text"])
        ax.set_ylabel("Accuracy", fontsize=11, fontweight="medium", color=self.palette["text"])
        ax.grid(True, linestyle="--", alpha=0.6, color=self.palette["grid"])
        ax.set_ylim([0.0, 1.05])
        ax.legend(frameon=True, facecolor="white", edgecolor=self.palette["grid"], fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close(fig)

        logger.info(f"Saved accuracy curve plot to: {output_path}")
        return output_path

    def plot_loss_curve(
        self,
        history: Dict[str, List[float]],
        filename: str = "loss_curve.png",
    ) -> Path:
        """Generates Training vs Validation Loss vs Epoch curve.

        Args:
            history: Dictionary containing 'loss' and 'val_loss' sequences.
            filename: Target image filename.

        Returns:
            Path to saved image file.
        """
        output_path = self.output_dir / filename
        train_loss = history.get("loss", [])
        val_loss = history.get("val_loss", [])
        epochs = range(1, len(train_loss) + 1)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.plot(epochs, train_loss, color=self.palette["train_loss"], marker="o", linewidth=2.2, label="Training Loss")
        if val_loss:
            ax.plot(epochs, val_loss, color=self.palette["val_loss"], marker="s", linestyle="--", linewidth=2.2, label="Validation Loss")

        ax.set_title("Training & Validation Loss vs Epoch", fontsize=13, fontweight="bold", color=self.palette["text"], pad=12)
        ax.set_xlabel("Epoch", fontsize=11, fontweight="medium", color=self.palette["text"])
        ax.set_ylabel("Loss", fontsize=11, fontweight="medium", color=self.palette["text"])
        ax.grid(True, linestyle="--", alpha=0.6, color=self.palette["grid"])
        ax.legend(frameon=True, facecolor="white", edgecolor=self.palette["grid"], fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close(fig)

        logger.info(f"Saved loss curve plot to: {output_path}")
        return output_path

    def plot_combined_metrics(
        self,
        history: Dict[str, List[float]],
        filename: str = "training_metrics.png",
    ) -> Path:
        """Generates side-by-side composite loss and accuracy figures.

        Args:
            history: Training history dictionary with accuracy, loss, val_accuracy, val_loss.
            filename: Target image filename.

        Returns:
            Path to saved composite image file.
        """
        output_path = self.output_dir / filename
        epochs = range(1, len(history.get("accuracy", [])) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        # Subplot 1: Accuracy
        ax1.plot(epochs, history.get("accuracy", []), color=self.palette["train_acc"], marker="o", linewidth=2.0, label="Train Accuracy")
        if "val_accuracy" in history:
            ax1.plot(epochs, history.get("val_accuracy", []), color=self.palette["val_acc"], marker="s", linestyle="--", linewidth=2.0, label="Val Accuracy")
        ax1.set_title("Model Accuracy Across Epochs", fontsize=12, fontweight="bold", color=self.palette["text"])
        ax1.set_xlabel("Epoch", fontsize=10)
        ax1.set_ylabel("Accuracy", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.6, color=self.palette["grid"])
        ax1.set_ylim([0.0, 1.05])
        ax1.legend(frameon=True, facecolor="white", edgecolor=self.palette["grid"])

        # Subplot 2: Loss
        ax1_loss = history.get("loss", [])
        ax2.plot(epochs, ax1_loss, color=self.palette["train_loss"], marker="o", linewidth=2.0, label="Train Loss")
        if "val_loss" in history:
            ax2.plot(epochs, history.get("val_loss", []), color=self.palette["val_loss"], marker="s", linestyle="--", linewidth=2.0, label="Val Loss")
        ax2.set_title("Model Loss Across Epochs", fontsize=12, fontweight="bold", color=self.palette["text"])
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.set_ylabel("Loss", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.6, color=self.palette["grid"])
        ax2.legend(frameon=True, facecolor="white", edgecolor=self.palette["grid"])

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close(fig)

        logger.info(f"Saved combined training metrics plot to: {output_path}")
        return output_path
