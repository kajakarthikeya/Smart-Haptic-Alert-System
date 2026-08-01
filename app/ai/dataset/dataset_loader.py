"""Dataset Ingestion and Loader Interface for Environmental Audio."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BaseDatasetLoader(ABC):
    """Abstract Base Class for Dataset Loaders."""

    @abstractmethod
    def load_dataset(self, data_dir: Union[str, Path]) -> Tuple[Any, Any]:
        """Loads and returns training features and label vectors.

        Args:
            data_dir: Path to directory containing audio samples and metadata.

        Returns:
            Tuple of (features, labels).
        """
        pass

    @abstractmethod
    def get_class_names(self) -> List[str]:
        """Returns list of target class labels."""
        pass


class AudioDatasetLoader(BaseDatasetLoader):
    """Starter Dataset Loader for Environmental Sound Datasets (e.g. ESC-50, UrbanSound8K)."""

    def __init__(self, sample_rate: int = 16000) -> None:
        """Initializes dataset loader settings.

        Args:
            sample_rate: Target sampling rate.
        """
        self._sample_rate = sample_rate
        self._class_names = [
            "fire_alarm",
            "baby_crying",
            "doorbell",
            "door_knock",
            "car_horn",
            "siren",
            "speech",
            "ambient_noise",
        ]
        logger.info(f"AudioDatasetLoader initialized (sample_rate={self._sample_rate}Hz)")

    def load_dataset(self, data_dir: Union[str, Path]) -> Tuple[List[Any], List[int]]:
        """Placeholder method for dataset loading in future training phase.

        Args:
            data_dir: Root dataset path.

        Returns:
            Placeholder tuple of empty feature list and label list.
        """
        data_path = Path(data_dir)
        logger.info(f"Loading environmental audio dataset from: {data_path}")
        # Structural placeholder
        return [], []

    def get_class_names(self) -> List[str]:
        """Returns target sound labels."""
        return self._class_names
