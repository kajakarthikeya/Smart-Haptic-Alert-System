"""Model Factory for environmental sound classifiers."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union
from app.ai.models.base_model import BaseSoundClassifier
from app.utils.logger import get_logger

logger = get_logger(__name__)


class StarterMockClassifier(BaseSoundClassifier):
    """Starter Mock Model implementation used during architecture initialization."""

    def __init__(self) -> None:
        self._loaded = True
        self._labels = [
            "fire_alarm",
            "baby_crying",
            "doorbell",
            "door_knock",
            "car_horn",
            "siren",
            "speech",
            "ambient_noise",
        ]

    def load_model(self, model_path: Union[str, Path]) -> bool:
        logger.info(f"Mock classifier loaded successfully from {model_path}")
        self._loaded = True
        return True

    def predict(self, features: Any) -> Tuple[str, float, List[float]]:
        # Starter mock prediction: returns doorbell with 0.85 confidence
        probs = [0.05, 0.05, 0.85, 0.01, 0.01, 0.01, 0.01, 0.01]
        return "doorbell", 0.85, probs

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class ModelFactory:
    """Factory creating instances of BaseSoundClassifier models."""

    _registry: Dict[str, Type[BaseSoundClassifier]] = {
        "mock": StarterMockClassifier,
        "starter": StarterMockClassifier,
    }

    @classmethod
    def register_model(cls, name: str, model_cls: Type[BaseSoundClassifier]) -> None:
        """Registers a new model architecture class with the factory.

        Args:
            name: Identifier name for model format/architecture.
            model_cls: Model class inheriting from BaseSoundClassifier.
        """
        cls._registry[name.lower()] = model_cls
        logger.info(f"Registered model architecture '{name}' in ModelFactory")

    @classmethod
    def create_model(cls, model_type: str = "starter") -> BaseSoundClassifier:
        """Instantiates a sound classifier by type name.

        Args:
            model_type: Name of registered model architecture.

        Returns:
            Instance of BaseSoundClassifier.
        """
        key = model_type.lower()
        if key not in cls._registry:
            logger.warning(f"Model type '{model_type}' not found in registry. Falling back to 'starter'.")
            key = "starter"
        
        model_cls = cls._registry[key]
        return model_cls()
