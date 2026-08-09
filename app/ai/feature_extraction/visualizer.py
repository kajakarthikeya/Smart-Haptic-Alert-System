"""Feature Visualization Utilities.

Generates and exports feature heatmaps (MFCC, Mel Spectrogram) and class distribution charts
for analysis and documentation.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np

from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


class FeatureVisualizer:
    """Renders and saves plots for acoustic features and dataset distributions."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """Initializes output visualization directory.

        Args:
            output_dir: Target directory for plots (default: app/outputs/feature_visualizations).
        """
        self.output_dir = Path(output_dir or settings.feature_extraction.visualization_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_mel_spectrogram(
        self,
        mel_spectrogram: np.ndarray,
        sample_rate: int = 22050,
        title: str = "Mel Spectrogram (dB)",
        filename: str = "mel_spectrogram.png",
    ) -> Path:
        """Renders and exports a Mel Spectrogram heatmap.

        Args:
            mel_spectrogram: 2D numpy array of shape (n_mels, time_steps).
            sample_rate: Sampling frequency.
            title: Plot header title.
            filename: Target file name.

        Returns:
            Path of saved plot PNG file.
        """
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spectrogram, aspect="auto", origin="lower", cmap="viridis")
        plt.colorbar(format="%+2.0f dB")
        plt.title(title)
        plt.xlabel("Time Frames")
        plt.ylabel("Mel Frequency Bands")
        plt.tight_layout()

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300)
        plt.close()
        logger.info(f"Saved Mel Spectrogram visualization to {filepath}")
        return filepath

    def plot_mfcc(
        self,
        mfcc: np.ndarray,
        title: str = "MFCC Coefficients",
        filename: str = "mfcc_heatmap.png",
    ) -> Path:
        """Renders and exports an MFCC heatmap.

        Args:
            mfcc: 2D numpy array of shape (n_mfcc, time_steps).
            title: Plot header title.
            filename: Target file name.

        Returns:
            Path of saved plot PNG file.
        """
        plt.figure(figsize=(10, 4))
        plt.imshow(mfcc, aspect="auto", origin="lower", cmap="coolwarm")
        plt.colorbar()
        plt.title(title)
        plt.xlabel("Time Frames")
        plt.ylabel("MFCC Coefficients")
        plt.tight_layout()

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300)
        plt.close()
        logger.info(f"Saved MFCC visualization to {filepath}")
        return filepath

    def plot_class_distribution(
        self,
        class_counts: Dict[str, int],
        title: str = "Dataset Class Distribution",
        filename: str = "class_distribution.png",
    ) -> Path:
        """Renders and exports a class distribution bar chart.

        Args:
            class_counts: Mapping from class names to sample counts.
            title: Plot header title.
            filename: Target file name.

        Returns:
            Path of saved plot PNG file.
        """
        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        plt.figure(figsize=(8, 5))
        bars = plt.bar(classes, counts, color="#4C72B0", edgecolor="black", alpha=0.85)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Sound Class", fontsize=12)
        plt.ylabel("Number of Samples", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Annotate bars with exact numbers
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300)
        plt.close()
        logger.info(f"Saved class distribution chart to {filepath}")
        return filepath
