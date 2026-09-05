"""CNN-based Environmental Sound Classifier Implementation.

Implements a 2D Convolutional Neural Network specialized for time-frequency
acoustic feature representations (composite 184x173 Mel/MFCC/Chroma/Spectral maps).
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Set default Keras backend to torch before importing Keras
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras import layers
import numpy as np

from app.ai.models.base_model import BaseSoundClassifier
from app.ai.models.exceptions import ModelSaveError, ModelTrainingError
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


class CNNSoundClassifier(BaseSoundClassifier):
    """2D Convolutional Neural Network Sound Classifier for wearable alerts."""

    def __init__(
        self,
        input_shape: Optional[Tuple[int, int, int]] = None,
        num_classes: Optional[int] = None,
        class_labels: Optional[List[str]] = None,
        learning_rate: Optional[float] = None,
        dropout_rate: Optional[float] = None,
    ) -> None:
        """Initializes CNN Sound Classifier.

        Args:
            input_shape: Input tensor dimensions (H, W, C), defaults to (184, 173, 1).
            num_classes: Number of target sound categories (defaults to 5).
            class_labels: List of class name strings in index order.
            learning_rate: Initial Adam learning rate.
            dropout_rate: Dropout regularization probability.
        """
        self.input_shape = input_shape or settings.model.input_shape
        self.num_classes = num_classes or settings.model.num_classes
        self.class_labels = class_labels or list(settings.dataset.target_classes)
        self.learning_rate = learning_rate or settings.training.learning_rate
        self.dropout_rate = dropout_rate or settings.training.dropout_rate

        self._model: Optional[keras.Model] = None
        self._loaded: bool = False

    @property
    def model(self) -> keras.Model:
        """Returns underlying Keras Model, building it if not yet created."""
        if self._model is None:
            self._model = self.build_model()
        return self._model

    @property
    def is_loaded(self) -> bool:
        """Returns True if model weights are loaded and ready for inference."""
        return self._loaded

    def build_model(self) -> keras.Model:
        """Constructs and compiles the 2D CNN architecture.

        Architecture Overview:
        - Input: (184, 173, 1) acoustic feature map
        - Block 1: Conv2D(32, 3x3) + BatchNorm + ReLU + MaxPool2D(2x2) + Dropout(0.25)
        - Block 2: Conv2D(64, 3x3) + BatchNorm + ReLU + MaxPool2D(2x2) + Dropout(0.25)
        - Block 3: Conv2D(128, 3x3) + BatchNorm + ReLU + MaxPool2D(2x2) + Dropout(0.30)
        - GlobalAveragePooling2D: Extracts robust spatial/temporal features
        - Dense Head: Dense(128) + BatchNorm + ReLU + Dropout(0.50)
        - Classification: Dense(num_classes, softmax)

        Returns:
            Compiled Keras Sequential Model.
        """
        try:
            model = keras.Sequential([
                # Input Layer
                layers.Input(shape=self.input_shape, name="acoustic_input"),

                # Conv Block 1
                layers.Conv2D(32, (3, 3), padding="same", name="conv1"),
                layers.BatchNormalization(name="bn1"),
                layers.Activation("relu", name="relu1"),
                layers.MaxPooling2D((2, 2), name="pool1"),
                layers.Dropout(self.dropout_rate, name="drop1"),

                # Conv Block 2
                layers.Conv2D(64, (3, 3), padding="same", name="conv2"),
                layers.BatchNormalization(name="bn2"),
                layers.Activation("relu", name="relu2"),
                layers.MaxPooling2D((2, 2), name="pool2"),
                layers.Dropout(self.dropout_rate, name="drop2"),

                # Conv Block 3
                layers.Conv2D(128, (3, 3), padding="same", name="conv3"),
                layers.BatchNormalization(name="bn3"),
                layers.Activation("relu", name="relu3"),
                layers.MaxPooling2D((2, 2), name="pool3"),
                layers.Dropout(self.dropout_rate + 0.05, name="drop3"),

                # Global Pooling & Dense Head
                layers.GlobalAveragePooling2D(name="global_avg_pool"),
                layers.Dense(128, name="dense1"),
                layers.BatchNormalization(name="bn4"),
                layers.Activation("relu", name="relu4"),
                layers.Dropout(0.5, name="drop4"),

                # Output Layer
                layers.Dense(self.num_classes, activation="softmax", name="class_output"),
            ], name="SoundClassifierCNN")

            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
            model.compile(
                optimizer=optimizer,
                loss=settings.training.loss_function,
                metrics=["accuracy"],
            )

            logger.info(f"Built CNN Sound Classifier architecture. Total parameters: {model.count_params():,}")
            return model
        except Exception as exc:
            raise ModelTrainingError(f"Failed to build CNN model: {str(exc)}") from exc

    def save_model(self, model_path: Union[str, Path]) -> bool:
        """Serializes trained Keras model weights and architecture to disk.

        Args:
            model_path: Destination path (.keras).

        Returns:
            True if saving succeeded.

        Raises:
            ModelSaveError: If model cannot be saved.
        """
        target = Path(model_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.model.save(str(target))
            logger.info(f"Successfully saved trained CNN model to: {target}")
            return True
        except Exception as exc:
            raise ModelSaveError(f"Failed to save model to '{target}': {str(exc)}") from exc

    def load_model(self, model_path: Union[str, Path]) -> bool:
        """Loads serialized model weights from file.

        Args:
            model_path: Path to serialized .keras model file.

        Returns:
            True if loaded successfully.
        """
        target = Path(model_path)
        if not target.exists():
            logger.error(f"Model file not found at: {target}")
            self._loaded = False
            return False

        try:
            self._model = keras.models.load_model(str(target))
            self._loaded = True
            logger.info(f"Loaded trained CNN model from: {target}")
            return True
        except Exception as exc:
            logger.error(f"Failed to load model from '{target}': {str(exc)}")
            self._loaded = False
            return False

    def predict(self, features: Any) -> Tuple[str, float, List[float]]:
        """Performs classification inference on an input feature map.

        Args:
            features: 2D array (184, 173) or 3D array (184, 173, 1) or 4D batch (1, 184, 173, 1).

        Returns:
            Tuple of (predicted_class_label: str, confidence_score: float, all_probabilities: List[float]).
        """
        arr = np.asarray(features, dtype=np.float32)

        # Standardize dimension to (1, 184, 173, 1)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=(0, -1))
        elif arr.ndim == 3:
            if arr.shape[-1] == 1:
                arr = np.expand_dims(arr, axis=0)
            else:
                arr = np.expand_dims(arr, axis=-1)
        elif arr.ndim == 4:
            pass
        else:
            raise ValueError(f"Expected input array with 2, 3, or 4 dimensions, got {arr.shape}")

        probs_batch = self.model.predict(arr, verbose=0)
        probs = probs_batch[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        if pred_idx < len(self.class_labels):
            pred_class = self.class_labels[pred_idx]
        else:
            pred_class = f"class_{pred_idx}"

        all_probs = [float(p) for p in probs]
        return pred_class, confidence, all_probs
