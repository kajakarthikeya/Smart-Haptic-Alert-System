"""Label Encoder Module for Sound Classes.

Provides bidirectional mapping between string class names and integer labels,
persisting mappings to structured JSON files for model training and real-time inference.
"""

import json
from pathlib import Path
from typing import Dict, List, Union

from app.ai.feature_extraction.exceptions import LabelEncodingError
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


class LabelEncoder:
    """Encodes categorical class names into numeric labels and decodes predictions."""

    def __init__(self, target_classes: Union[List[str], tuple] = None) -> None:
        """Initializes label mappings.

        Args:
            target_classes: Iterable of sound class names. Defaults to config settings.
        """
        classes = list(target_classes or settings.dataset.target_classes)
        # Sort classes to guarantee deterministic mapping unless pre-configured
        self._classes: List[str] = list(classes)
        self._class_to_id: Dict[str, int] = {name: idx for idx, name in enumerate(self._classes)}
        self._id_to_class: Dict[int, str] = {idx: name for idx, name in enumerate(self._classes)}

    @property
    def classes(self) -> List[str]:
        """List of target class names in indexed order."""
        return list(self._classes)

    @property
    def num_classes(self) -> int:
        """Total number of registered classes."""
        return len(self._classes)

    def encode(self, class_name: str) -> int:
        """Converts a class name string into its integer label.

        Args:
            class_name: Target sound class name.

        Returns:
            Integer label ID.

        Raises:
            LabelEncodingError: If class_name is not in registered classes.
        """
        clean_name = class_name.strip().lower()
        if clean_name not in self._class_to_id:
            raise LabelEncodingError(
                label=class_name,
                reason=f"Unknown class label '{class_name}'. Registered classes: {self._classes}",
            )
        return self._class_to_id[clean_name]

    def decode(self, label_id: int) -> str:
        """Converts an integer label ID back to its string class name.

        Args:
            label_id: Integer label ID.

        Returns:
            Class name string.

        Raises:
            LabelEncodingError: If label_id is not registered.
        """
        if label_id not in self._id_to_class:
            raise LabelEncodingError(
                label=str(label_id),
                reason=f"Unknown label ID {label_id}. Registered IDs: {list(self._id_to_class.keys())}",
            )
        return self._id_to_class[label_id]

    def get_mapping(self) -> Dict[str, int]:
        """Returns a copy of the class-to-id dictionary mapping."""
        return dict(self._class_to_id)

    def save_mapping(self, filepath: Path) -> Path:
        """Saves class mapping to a JSON file.

        Args:
            filepath: Target output file path (e.g., class_names.json).

        Returns:
            Path object of saved mapping file.
        """
        dest_path = Path(filepath)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "class_to_id": self._class_to_id,
            "id_to_class": {str(k): v for k, v in self._id_to_class.items()},
            "classes": self._classes,
            "num_classes": self.num_classes,
        }
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Class label mapping saved to {dest_path}")
        return dest_path

    @classmethod
    def load_mapping(cls, filepath: Path) -> "LabelEncoder":
        """Loads class mapping from a saved JSON file.

        Args:
            filepath: Path to class_names.json.

        Returns:
            Configured LabelEncoder instance.

        Raises:
            LabelEncodingError: If file cannot be read or is invalid.
        """
        source_path = Path(filepath)
        if not source_path.exists():
            raise LabelEncodingError(
                label=str(filepath), reason=f"Mapping file not found at '{filepath}'."
            )
        try:
            with open(source_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            encoder = cls(target_classes=data["classes"])
            return encoder
        except Exception as exc:
            raise LabelEncodingError(
                label=str(filepath), reason=f"Failed to load class mapping JSON: {str(exc)}"
            )
