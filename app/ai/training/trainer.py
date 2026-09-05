"""Model Training Subsystem Implementation.

Executes 2D CNN sound classifier training with validation loops, early stopping,
checkpointing, learning rate reduction on plateau, class weight support,
and automated metadata & history persistence.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Set default Keras backend to torch
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from app.ai.models.base_model import BaseSoundClassifier
from app.ai.models.cnn_classifier import CNNSoundClassifier
from app.ai.training.data_loader import TrainingDataset
from app.ai.training.exceptions import ModelSaveError, ModelTrainingError
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


class BaseTrainer(ABC):
    """Abstract Base Class for Model Trainers."""

    @abstractmethod
    def train(self, dataset: Any, epochs: Optional[int] = None) -> Dict[str, Any]:
        """Runs the model training loop.

        Args:
            dataset: Training dataset container.
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
    """Deep Learning Model Trainer orchestrating CNN training lifecycle."""

    def __init__(
        self,
        model: Optional[CNNSoundClassifier] = None,
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        random_seed: Optional[int] = None,
        save_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initializes ModelTrainer with configurations.

        Args:
            model: CNNSoundClassifier instance. If None, instantiates a new one.
            batch_size: Mini-batch size.
            epochs: Max training epochs.
            learning_rate: Initial optimizer learning rate.
            random_seed: Random seed for reproducibility.
            save_dir: Directory to save model checkpoints.
            output_dir: Directory to save training history and logs.
        """
        self.batch_size = batch_size or settings.training.batch_size
        self.epochs = epochs or settings.training.epochs
        self.learning_rate = learning_rate or settings.training.learning_rate
        self.random_seed = random_seed if random_seed is not None else settings.training.random_seed
        self.save_dir = Path(save_dir or settings.training.model_save_dir)
        self.output_dir = Path(output_dir or settings.training.training_output_dir)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._set_seed(self.random_seed)

        self.model = model or CNNSoundClassifier(learning_rate=self.learning_rate)
        self.history: Dict[str, List[float]] = {}
        self.best_model_path = self.save_dir / settings.model.best_model_filename
        self.final_model_path = self.save_dir / settings.model.final_model_filename
        self.metadata_path = self.save_dir / settings.model.metadata_filename

        logger.info(
            f"Initialized ModelTrainer (Epochs={self.epochs}, BatchSize={self.batch_size}, "
            f"LR={self.learning_rate}, Seed={self.random_seed})"
        )

    def _set_seed(self, seed: int) -> None:
        """Enforces deterministic random seed across runtimes."""
        random.seed(seed)
        np.random.seed(seed)
        keras.utils.set_random_seed(seed)
        logger.debug(f"Applied random seed: {seed}")

    def train(
        self,
        dataset: TrainingDataset,
        epochs: Optional[int] = None,
        use_class_weights: bool = True,
    ) -> Dict[str, Any]:
        """Executes the complete training loop with validation and callbacks.

        Args:
            dataset: Validated TrainingDataset container.
            epochs: Epoch count override.
            use_class_weights: If True, applies balanced class weights during fitting.

        Returns:
            Dictionary containing metrics history across all epochs.

        Raises:
            ModelTrainingError: If training fails during execution.
        """
        total_epochs = epochs or self.epochs
        logger.info(
            f"Starting model training for {total_epochs} epochs on {len(dataset.X_train)} training samples "
            f"and {len(dataset.X_val)} validation samples..."
        )

        # Build callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=settings.training.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(self.best_model_path),
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=settings.training.reduce_lr_patience,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        class_weights_dict = dataset.class_weights if use_class_weights else None

        try:
            fit_history = self.model.model.fit(
                x=dataset.X_train,
                y=dataset.y_train,
                batch_size=min(self.batch_size, len(dataset.X_train)),
                epochs=total_epochs,
                validation_data=(dataset.X_val, dataset.y_val),
                class_weight=class_weights_dict,
                callbacks=callbacks,
                verbose=1,
                shuffle=True,
            )

            # Convert history metrics to serializable Python floats
            self.history = {
                metric: [float(val) for val in values]
                for metric, values in fit_history.history.items()
            }

            # Save the final model state as well
            self.export_model(self.final_model_path)

            # Ensure best model exists (if ModelCheckpoint didn't trigger, save current)
            if not self.best_model_path.exists():
                self.export_model(self.best_model_path)

            # Save training history JSON
            self.save_training_history()

            # Save model metadata JSON
            self.save_model_metadata(dataset)

            final_train_acc = self.history.get("accuracy", [0.0])[-1]
            best_val_acc = max(self.history.get("val_accuracy", [0.0]))
            logger.info(
                f"Model training completed: Final Train Acc={final_train_acc:.4f}, "
                f"Best Val Acc={best_val_acc:.4f}"
            )

            return self.history

        except Exception as exc:
            logger.error(f"Training loop encountered a fatal error: {str(exc)}")
            raise ModelTrainingError(f"Model training failed: {str(exc)}") from exc

    def export_model(self, export_path: Union[str, Path]) -> bool:
        """Exports trained model weights to destination path.

        Args:
            export_path: Destination path (.keras).

        Returns:
            True if export succeeded.
        """
        return self.model.save_model(export_path)

    def save_training_history(
        self, filename: str = "training_history.json"
    ) -> Path:
        """Saves epoch metric progression to machine-readable JSON format.

        Args:
            filename: Target JSON filename.

        Returns:
            Path to saved JSON file.

        Raises:
            ModelSaveError: If saving history fails.
        """
        output_file = self.output_dir / filename
        try:
            payload = {
                "metrics": self.history,
                "epochs_trained": len(self.history.get("loss", [])),
                "best_val_accuracy": float(max(self.history.get("val_accuracy", [0.0]))),
                "final_train_accuracy": float(self.history.get("accuracy", [0.0])[-1]) if self.history.get("accuracy") else 0.0,
                "final_train_loss": float(self.history.get("loss", [0.0])[-1]) if self.history.get("loss") else 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info(f"Saved training history to: {output_file}")
            return output_file
        except Exception as exc:
            raise ModelSaveError(f"Failed to save training history: {str(exc)}") from exc

    def save_model_metadata(
        self, dataset: TrainingDataset, filename: Optional[str] = None
    ) -> Path:
        """Generates and exports comprehensive model metadata JSON.

        Args:
            dataset: TrainingDataset used during training.
            filename: Target JSON filename (defaults to config setting).

        Returns:
            Path to exported metadata file.

        Raises:
            ModelSaveError: If saving metadata fails.
        """
        output_file = self.save_dir / (filename or settings.model.metadata_filename)
        try:
            metadata = {
                "model_name": settings.model.model_name,
                "model_version": settings.model.model_version,
                "architecture": "2D Convolutional Neural Network (CNN)",
                "total_parameters": int(self.model.model.count_params()),
                "number_of_classes": dataset.num_classes,
                "class_mapping": dataset.class_to_id,
                "id_to_class": {str(k): v for k, v in dataset.id_to_class.items()},
                "input_shape": list(dataset.input_shape),
                "feature_type": "Composite Time-Frequency Matrix (Mel, MFCC, Chroma, Spectral)",
                "dataset_summary": {
                    "train_samples": len(dataset.X_train),
                    "val_samples": len(dataset.X_val),
                    "test_samples": len(dataset.X_test),
                    "total_samples": len(dataset.X_train) + len(dataset.X_val) + len(dataset.X_test),
                    "class_weights": {str(k): float(v) for k, v in (dataset.class_weights or {}).items()},
                },
                "training_hyperparameters": {
                    "batch_size": self.batch_size,
                    "epochs_requested": self.epochs,
                    "epochs_completed": len(self.history.get("loss", [])),
                    "learning_rate": self.learning_rate,
                    "optimizer": settings.training.optimizer,
                    "loss_function": settings.training.loss_function,
                    "dropout_rate": self.model.dropout_rate,
                    "random_seed": self.random_seed,
                },
                "training_performance": {
                    "final_train_accuracy": float(self.history.get("accuracy", [0.0])[-1]) if self.history.get("accuracy") else 0.0,
                    "best_val_accuracy": float(max(self.history.get("val_accuracy", [0.0]))),
                    "final_train_loss": float(self.history.get("loss", [0.0])[-1]) if self.history.get("loss") else 0.0,
                    "final_val_loss": float(self.history.get("val_loss", [0.0])[-1]) if self.history.get("val_loss") else 0.0,
                },
                "artifacts": {
                    "best_model": str(self.best_model_path.name),
                    "final_model": str(self.final_model_path.name),
                },
                "training_date": datetime.now(timezone.utc).isoformat(),
                "framework": f"Keras {keras.__version__} (Backend: {keras.backend.backend()})",
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Exported model metadata report to: {output_file}")
            return output_file
        except Exception as exc:
            raise ModelSaveError(f"Failed to save model metadata: {str(exc)}") from exc
