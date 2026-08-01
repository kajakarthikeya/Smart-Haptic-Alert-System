"""Base Model Interface definition."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Tuple, Union


class BaseSoundClassifier(ABC):
    """Abstract Base Class defining the contract for all Environmental Sound Classifiers."""

    @abstractmethod
    def load_model(self, model_path: Union[str, Path]) -> bool:
        """Loads model weights/artifacts from file.

        Args:
            model_path: Path to serialized model file (.tflite, .h5, .onnx, etc.).

        Returns:
            True if loaded successfully, False otherwise.
        """
        pass

    @abstractmethod
    def predict(self, features: Any) -> Tuple[str, float, List[float]]:
        """Performs classification inference on extracted features.

        Args:
            features: Preprocessed feature tensor.

        Returns:
            Tuple of (predicted_class_label: str, confidence_score: float, all_probabilities: List[float]).
        """
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Returns True if model weights are loaded and ready for inference."""
        pass
