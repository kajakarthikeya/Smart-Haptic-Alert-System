"""Dataset Directory Manager for automated folder structure creation and maintenance."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from config import settings
from app.utils.logger import get_logger
from app.ai.dataset.exceptions import DatasetNotFoundError

logger = get_logger(__name__)


class DatasetDirectoryManager:
    """Manages creation, verification, and directory structure initialization for datasets."""

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        test_audio_dir: Optional[Path] = None,
        target_classes: Optional[Tuple[str, ...]] = None,
    ) -> None:
        """Initializes DatasetDirectoryManager with paths from settings or explicit arguments.

        Args:
            raw_dir: Directory for raw audio files.
            processed_dir: Directory for preprocessed data.
            test_audio_dir: Directory for testing audio samples.
            target_classes: Sequence of target sound class labels.
        """
        self._raw_dir = raw_dir or settings.dataset.raw_dir
        self._processed_dir = processed_dir or settings.dataset.processed_dir
        self._test_audio_dir = test_audio_dir or settings.dataset.test_audio_dir
        self._target_classes = target_classes or settings.dataset.target_classes
        logger.info("DatasetDirectoryManager initialized.")

    @property
    def raw_dir(self) -> Path:
        return self._raw_dir

    @property
    def processed_dir(self) -> Path:
        return self._processed_dir

    @property
    def test_audio_dir(self) -> Path:
        return self._test_audio_dir

    @property
    def target_classes(self) -> Tuple[str, ...]:
        return self._target_classes

    def initialize_directories(self) -> Dict[str, Path]:
        """Creates the dataset folder hierarchy if it does not exist.

        Folder Structure Created:
        dataset/
            raw/
                ambulance/
                car_horn/
                fire_alarm/
                doorbell/
                dog_bark/
            processed/
            test_audio/

        Returns:
            Dictionary mapping folder key -> Path object.
        """
        created_paths: Dict[str, Path] = {}

        # 1. Create root subdirectories
        for dir_name, path in [
            ("raw", self._raw_dir),
            ("processed", self._processed_dir),
            ("test_audio", self._test_audio_dir),
        ]:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created dataset directory: {path}")
            else:
                logger.debug(f"Dataset directory already exists: {path}")
            created_paths[dir_name] = path

        # 2. Create raw class subdirectories
        for class_label in self._target_classes:
            class_path = self._raw_dir / class_label
            if not class_path.exists():
                class_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created class directory for '{class_label}': {class_path}")
            else:
                logger.debug(f"Class directory already exists for '{class_label}': {class_path}")
            created_paths[f"raw_{class_label}"] = class_path

        logger.info("Dataset directory hierarchy successfully initialized.")
        return created_paths

    def verify_structure(self) -> Dict[str, bool]:
        """Verifies presence of required dataset directories.

        Returns:
            Dictionary mapping path name -> status boolean.
        """
        status: Dict[str, bool] = {
            "raw": self._raw_dir.exists(),
            "processed": self._processed_dir.exists(),
            "test_audio": self._test_audio_dir.exists(),
        }

        for class_label in self._target_classes:
            class_path = self._raw_dir / class_label
            status[f"raw_{class_label}"] = class_path.exists()

        missing = [k for k, exists in status.items() if not exists]
        if missing:
            logger.warning(f"Missing required dataset directories: {missing}")
        else:
            logger.info("Dataset directory structure verification passed.")

        return status

    def get_class_directories(self) -> Dict[str, Path]:
        """Returns mapping of target class name -> Path object under raw_dir."""
        return {label: self._raw_dir / label for label in self._target_classes}
