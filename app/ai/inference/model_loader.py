"""
Model Loading and Validation Subsystem for Real-Time Inference.

Loads the best trained Phase 5 CNN model once at application startup, validates
input/output shapes against metadata, and binds canonical class mappings.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from config import Config, InferenceConfig, settings
from app.ai.models.cnn_classifier import CNNSoundClassifier
from app.ai.inference.exceptions import ModelLoadingError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceModelLoader:
    """Loads and validates the trained sound classification model artifact and metadata."""

    def __init__(
        self,
        config: Optional[InferenceConfig] = None,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        class_mapping_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.config = config or getattr(settings, "inference", None)

        def_mod = (
            self.config.model_path
            if self.config
            else settings.training.model_save_dir / settings.model.best_model_filename
        )
        def_meta = (
            self.config.metadata_path
            if self.config
            else settings.training.model_save_dir / settings.model.metadata_filename
        )
        def_map = (
            self.config.class_mapping_path
            if self.config
            else settings.feature_extraction.features_dir / "class_names.json"
        )

        self.model_path = Path(model_path or def_mod)
        self.metadata_path = Path(metadata_path or def_meta)
        self.class_mapping_path = Path(class_mapping_path or def_map)

        self._classifier: Optional[CNNSoundClassifier] = None
        self._class_names: List[str] = []
        self._class_to_id: Dict[str, int] = {}
        self._id_to_class: Dict[int, str] = {}
        self._metadata: Dict[str, Any] = {}
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Returns True if model and metadata have been loaded and validated."""
        return self._is_loaded

    @property
    def classifier(self) -> CNNSoundClassifier:
        """Returns the loaded classifier instance."""
        if not self._is_loaded or self._classifier is None:
            raise ModelLoadingError("Model has not been loaded. Call load() first.")
        return self._classifier

    @property
    def class_names(self) -> List[str]:
        """Returns canonical list of class names."""
        return list(self._class_names)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Returns model training metadata dictionary."""
        return dict(self._metadata)

    def load(self) -> CNNSoundClassifier:
        """
        Loads and validates model weights, metadata, and class mappings.
        Executed ONCE at startup to avoid repeated I/O overhead.
        """
        if self._is_loaded and self._classifier is not None:
            logger.debug("Model already loaded; reusing existing instance.")
            return self._classifier

        logger.info("Loading inference model from: %s", self.model_path)

        # 1. Verify file existence
        if not self.model_path.exists():
            raise ModelLoadingError(f"Trained model artifact not found at: {self.model_path}")

        # 2. Load class mappings
        self._load_class_mapping()

        # 3. Load metadata if available
        self._load_metadata()

        # 4. Instantiate and load classifier
        try:
            expected_input_shape = (184, 173, 1)
            num_classes = len(self._class_names)

            self._classifier = CNNSoundClassifier(
                input_shape=expected_input_shape,
                num_classes=num_classes,
                class_labels=self._class_names,
            )

            load_success = self._classifier.load_model(self.model_path)
            if not load_success or not self._classifier.is_loaded:
                raise ModelLoadingError(f"Failed to load model weights from: {self.model_path}")

            # Validate input shape & output layer
            keras_model = getattr(self._classifier, "model", None)
            if keras_model is not None:
                actual_input = keras_model.input_shape
                # Input shape format is (None, 184, 173, 1)
                if actual_input[1:] != expected_input_shape:
                    raise ModelLoadingError(
                        f"Model input shape mismatch: expected {expected_input_shape}, got {actual_input[1:]}"
                    )

                actual_output = keras_model.output_shape
                if actual_output[-1] != num_classes:
                    raise ModelLoadingError(
                        f"Model output classes mismatch: expected {num_classes}, got {actual_output[-1]}"
                    )

            self._is_loaded = True
            logger.info(
                "Model loaded successfully: %s (%d classes: %s)",
                self.model_path.name,
                len(self._class_names),
                self._class_names,
            )
            return self._classifier

        except Exception as exc:
            self._is_loaded = False
            self._classifier = None
            if isinstance(exc, ModelLoadingError):
                raise
            raise ModelLoadingError(f"Unexpected error loading inference model: {exc}") from exc

    def _load_class_mapping(self) -> None:
        """Loads and verifies class names mapping JSON."""
        if not self.class_mapping_path.exists():
            # Fallback to standard 5 target classes
            logger.warning(
                "Class mapping file not found at %s. Using default 5 classes.",
                self.class_mapping_path,
            )
            self._class_names = ["ambulance", "car_horn", "fire_alarm", "doorbell", "dog_bark"]
            self._class_to_id = {c: i for i, c in enumerate(self._class_names)}
            self._id_to_class = {i: c for i, c in enumerate(self._class_names)}
            return

        try:
            with open(self.class_mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._class_to_id = {str(k): int(v) for k, v in data.get("class_to_id", {}).items()}
            self._id_to_class = {int(k): str(v) for k, v in data.get("id_to_class", {}).items()}
            self._class_names = [self._id_to_class[i] for i in range(len(self._id_to_class))]

            # Verify target classes
            target_classes = {"ambulance", "car_horn", "fire_alarm", "doorbell", "dog_bark"}
            if not target_classes.issubset(set(self._class_names)):
                raise ModelLoadingError(
                    f"Class mapping missing required classes. Found: {self._class_names}, Required: {target_classes}"
                )

        except Exception as exc:
            if isinstance(exc, ModelLoadingError):
                raise
            raise ModelLoadingError(f"Failed to read class mapping '{self.class_mapping_path}': {exc}") from exc

    def _load_metadata(self) -> None:
        """Loads model training metadata if present."""
        if not self.metadata_path.exists():
            logger.warning("Model metadata file not found at: %s", self.metadata_path)
            return

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
            logger.debug("Loaded model metadata: %s", self.metadata_path)
        except Exception as exc:
            logger.warning("Failed to parse model metadata: %s", exc)
