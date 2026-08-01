"""Model Training Pipeline Interface & Implementation."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union
from app.ai.models.base_model import BaseSoundClassifier
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BaseTrainer(ABC):
    """Abstract Base Class for Model Trainers."""

    @abstractmethod
    def train(self, dataset_path: Union[str, Path], epochs: int = 10) -> Dict[str, Any]:
        """Runs the model training loop.

        Args:
            dataset_path: Location of dataset.
            epochs: Number of training iterations.

        Returns:
            Dictionary containing training metrics history.
        """
        pass

    @abstractmethod
    def export_model(self, export_path: Union[str, Path]) -> bool:
        """Exports trained model weights/quantized binary artifact.

        Args:
            export_path: Destination path.

        Returns:
            True if export succeeded.
        """
        pass


class ModelTrainer(BaseTrainer):
    """Starter Model Trainer implementation for deep learning models."""

    def __init__(self, model: BaseSoundClassifier) -> None:
        """Initializes trainer with a target model instance.

        Args:
            model: Instance of BaseSoundClassifier.
        """
        self._model = model
        logger.info(f"ModelTrainer initialized with target model: {type(model).__name__}")

    def train(self, dataset_path: Union[str, Path], epochs: int = 10) -> Dict[str, Any]:
        """Placeholder for training execution loop.

        Args:
            dataset_path: Dataset path.
            epochs: Training epochs.

        Returns:
            Mock history metrics dictionary.
        """
        logger.info(f"Initiating training loop for {epochs} epochs on dataset {dataset_path}...")
        # Structural placeholder
        return {"loss": [0.5, 0.2], "accuracy": [0.80, 0.92]}

    def export_model(self, export_path: Union[str, Path]) -> bool:
        """Placeholder for TFLite / ONNX model serialization.

        Args:
            export_path: Target path.

        Returns:
            True.
        """
        target = Path(export_path)
        logger.info(f"Exporting model artifact to: {target}")
        return True
