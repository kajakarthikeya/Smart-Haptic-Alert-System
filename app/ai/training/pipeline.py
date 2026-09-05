"""Automated AI Model Training Pipeline Subsystem.

Coordinates the end-to-end training process: loading features, validating data integrity,
compiling the CNN architecture, executing training with callbacks and class weights,
saving best and final models, generating metadata & history, producing visualization curves,
and performing post-training inference verification.
"""

from pathlib import Path
import time
from typing import Any, Dict, Optional, Union

import numpy as np

from app.ai.models.cnn_classifier import CNNSoundClassifier
from app.ai.training.data_loader import TrainingDataLoader, TrainingDataset
from app.ai.training.exceptions import ModelTrainingError
from app.ai.training.trainer import ModelTrainer
from app.ai.training.visualizer import TrainingVisualizer
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


class TrainingPipeline:
    """Automated training pipeline coordinating end-to-end model training."""

    def __init__(
        self,
        features_dir: Optional[Union[str, Path]] = None,
        model_save_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initializes pipeline components and configuration parameters.

        Args:
            features_dir: Directory containing extracted features and mappings.
            model_save_dir: Destination directory for trained model artifacts.
            output_dir: Destination directory for graphs and training history.
            epochs: Maximum training epochs.
            batch_size: Mini-batch size.
            learning_rate: Optimizer learning rate.
            random_seed: Random seed for deterministic reproducibility.
        """
        self.features_dir = Path(features_dir or settings.feature_extraction.features_dir)
        self.model_save_dir = Path(model_save_dir or settings.training.model_save_dir)
        self.output_dir = Path(output_dir or settings.training.training_output_dir)
        self.epochs = epochs or settings.training.epochs
        self.batch_size = batch_size or settings.training.batch_size
        self.learning_rate = learning_rate or settings.training.learning_rate
        self.random_seed = random_seed if random_seed is not None else settings.training.random_seed

        self.data_loader = TrainingDataLoader(features_dir=self.features_dir)
        self.visualizer = TrainingVisualizer(output_dir=self.output_dir)

    def run(self, generate_visualizations: bool = True) -> Dict[str, Any]:
        """Runs the complete AI model training workflow.

        Args:
            generate_visualizations: Flag to generate and save training curves.

        Returns:
            Dictionary containing training summary metrics and verification results.

        Raises:
            ModelTrainingError: If any stage fails during pipeline execution.
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("STARTING AI MODEL TRAINING PIPELINE (PHASE 5)")
        logger.info("=" * 60)

        # 1. Load and validate dataset
        logger.info("Step 1/6: Loading and validating extracted features...")
        dataset: TrainingDataset = self.data_loader.load_dataset(add_channel_dim=True)

        # 2. Build CNN model architecture
        logger.info("Step 2/6: Initializing CNN Sound Classifier architecture...")
        classifier = CNNSoundClassifier(
            input_shape=dataset.input_shape,
            num_classes=dataset.num_classes,
            class_labels=[dataset.id_to_class[i] for i in range(dataset.num_classes)],
            learning_rate=self.learning_rate,
            dropout_rate=settings.training.dropout_rate,
        )

        # 3. Initialize Trainer
        trainer = ModelTrainer(
            model=classifier,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            random_seed=self.random_seed,
            save_dir=self.model_save_dir,
            output_dir=self.output_dir,
        )

        # 4. Execute training loop
        logger.info(f"Step 3/6: Training CNN model for up to {self.epochs} epochs...")
        history = trainer.train(dataset, epochs=self.epochs, use_class_weights=True)

        # 5. Generate visualization curves
        plots: Dict[str, str] = {}
        if generate_visualizations:
            logger.info("Step 4/6: Generating training and validation performance charts...")
            try:
                acc_plot = self.visualizer.plot_accuracy_curve(history, "accuracy_curve.png")
                loss_plot = self.visualizer.plot_loss_curve(history, "loss_curve.png")
                combined_plot = self.visualizer.plot_combined_metrics(history, "training_metrics.png")
                plots = {
                    "accuracy_curve": str(acc_plot),
                    "loss_curve": str(loss_plot),
                    "combined_metrics": str(combined_plot),
                }
            except Exception as vis_exc:
                logger.warning(f"Failed to generate training charts: {str(vis_exc)}")

        # 6. Post-training verification
        logger.info("Step 5/6: Executing post-training model verification...")
        verification_result = self._verify_trained_model(trainer.best_model_path, dataset)

        elapsed_time = time.time() - start_time
        logger.info(f"Step 6/6: Training pipeline completed successfully in {elapsed_time:.2f} seconds.")
        logger.info("=" * 60)

        return {
            "status": "success",
            "elapsed_seconds": round(elapsed_time, 2),
            "epochs_completed": len(history.get("loss", [])),
            "final_train_accuracy": float(history.get("accuracy", [0.0])[-1]) if history.get("accuracy") else 0.0,
            "best_val_accuracy": float(max(history.get("val_accuracy", [0.0]))),
            "final_train_loss": float(history.get("loss", [0.0])[-1]) if history.get("loss") else 0.0,
            "best_model_path": str(trainer.best_model_path),
            "final_model_path": str(trainer.final_model_path),
            "metadata_path": str(trainer.metadata_path),
            "history_path": str(trainer.output_dir / "training_history.json"),
            "plots": plots,
            "verification": verification_result,
        }

    def _verify_trained_model(
        self, best_model_path: Path, dataset: TrainingDataset
    ) -> Dict[str, Any]:
        """Loads the saved best model and verifies single sample inference.

        Args:
            best_model_path: Path to serialized best .keras model.
            dataset: Validated dataset containing test samples.

        Returns:
            Dictionary containing verification predictions and status.
        """
        verify_classifier = CNNSoundClassifier(
            input_shape=dataset.input_shape,
            num_classes=dataset.num_classes,
            class_labels=[dataset.id_to_class[i] for i in range(dataset.num_classes)],
        )

        load_success = verify_classifier.load_model(best_model_path)
        if not load_success:
            raise ModelTrainingError(f"Verification failed: Could not load saved best model from {best_model_path}")

        # Pick first test sample
        test_sample = dataset.X_test[0]
        actual_label_id = int(dataset.y_test[0])
        actual_class = dataset.id_to_class.get(actual_label_id, f"class_{actual_label_id}")

        predicted_class, confidence, probabilities = verify_classifier.predict(test_sample)

        # Validate predicted class belongs to target classes
        valid_classes = set(dataset.class_to_id.keys())
        is_valid_class = predicted_class in valid_classes

        logger.info(
            f"Verification sample prediction: Actual='{actual_class}', "
            f"Predicted='{predicted_class}', Confidence={confidence:.4f}, Valid={is_valid_class}"
        )

        return {
            "load_successful": True,
            "test_sample_index": 0,
            "actual_class": actual_class,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": [round(p, 4) for p in probabilities],
            "belongs_to_target_classes": is_valid_class,
        }
