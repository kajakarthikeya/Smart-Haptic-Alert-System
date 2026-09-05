"""Unit Test Suite for AI Model Training Subsystem (Phase 5)."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from app.ai.models.base_model import BaseSoundClassifier
from app.ai.models.cnn_classifier import CNNSoundClassifier
from app.ai.models.model_factory import ModelFactory
from app.ai.training.data_loader import TrainingDataLoader, TrainingDataset
from app.ai.training.exceptions import (
    ConfigurationError,
    InvalidFeatureData,
    InvalidLabelData,
    ModelSaveError,
    ModelTrainingError,
)
from app.ai.training.pipeline import TrainingPipeline
from app.ai.training.trainer import ModelTrainer
from app.ai.training.visualizer import TrainingVisualizer
from config import settings


class TestModelTraining(unittest.TestCase):
    """Unit tests for Phase 5 AI Model Training Subsystem."""

    def setUp(self) -> None:
        """Sets up temporary workspace, synthetic feature data, and mock class mappings."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        self.num_classes = 5
        self.rows = 184
        self.cols = 173

        self.class_to_id = {
            "ambulance": 0,
            "car_horn": 1,
            "fire_alarm": 2,
            "doorbell": 3,
            "dog_bark": 4,
        }
        self.id_to_class = {str(v): k for k, v in self.class_to_id.items()}

        # Create valid class mapping file
        self.mapping_file = self.test_dir / "class_names.json"
        with open(self.mapping_file, "w", encoding="utf-8") as f:
            json.dump({
                "class_to_id": self.class_to_id,
                "id_to_class": self.id_to_class,
                "classes": list(self.class_to_id.keys()),
                "num_classes": self.num_classes,
            }, f, indent=2)

        # Create synthetic dataset splits (10 train, 5 val, 5 test)
        self.n_train = 10
        self.n_val = 5
        self.n_test = 5

        rng = np.random.default_rng(42)
        self.X_train = rng.standard_normal((self.n_train, self.rows, self.cols)).astype(np.float32)
        self.y_train = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=int)

        self.X_val = rng.standard_normal((self.n_val, self.rows, self.cols)).astype(np.float32)
        self.y_val = np.array([0, 1, 2, 3, 4], dtype=int)

        self.X_test = rng.standard_normal((self.n_test, self.rows, self.cols)).astype(np.float32)
        self.y_test = np.array([0, 1, 2, 3, 4], dtype=int)

        self.splits_file = self.test_dir / "dataset_splits.npz"
        np.savez_compressed(
            self.splits_file,
            X_composite_train=self.X_train,
            y_train=self.y_train,
            X_composite_val=self.X_val,
            y_val=self.y_val,
            X_composite_test=self.X_test,
            y_test=self.y_test,
        )

        self.data_loader = TrainingDataLoader(
            features_dir=self.test_dir,
            expected_rows=self.rows,
            expected_cols=self.cols,
        )

    def tearDown(self) -> None:
        """Cleans up temporary directory."""
        self.temp_dir.cleanup()

    def test_data_loading_success_and_channel_expansion(self) -> None:
        """Tests that TrainingDataLoader loads splits and adds channel dimension."""
        dataset = self.data_loader.load_dataset(add_channel_dim=True)

        self.assertIsInstance(dataset, TrainingDataset)
        self.assertEqual(dataset.X_train.shape, (self.n_train, self.rows, self.cols, 1))
        self.assertEqual(dataset.X_val.shape, (self.n_val, self.rows, self.cols, 1))
        self.assertEqual(dataset.X_test.shape, (self.n_test, self.rows, self.cols, 1))
        self.assertEqual(len(dataset.y_train), self.n_train)
        self.assertEqual(dataset.num_classes, 5)
        self.assertEqual(dataset.input_shape, (self.rows, self.cols, 1))

    def test_missing_splits_file_raises_configuration_error(self) -> None:
        """Tests that missing dataset_splits.npz raises ConfigurationError."""
        loader = TrainingDataLoader(features_dir=self.test_dir / "nonexistent")
        with self.assertRaises(ConfigurationError):
            loader.load_dataset()

    def test_nan_detection_raises_invalid_feature_data(self) -> None:
        """Tests that NaN values in features trigger InvalidFeatureData."""
        bad_X = self.X_train.copy()
        bad_X[0, 5, 5] = np.nan
        corrupt_splits = self.test_dir / "corrupt_nan.npz"
        np.savez_compressed(
            corrupt_splits,
            X_composite_train=bad_X,
            y_train=self.y_train,
            X_composite_val=self.X_val,
            y_val=self.y_val,
            X_composite_test=self.X_test,
            y_test=self.y_test,
        )

        with self.assertRaises(InvalidFeatureData) as ctx:
            self.data_loader.load_dataset(splits_filename="corrupt_nan.npz")
        self.assertIn("NaN values", str(ctx.exception))

    def test_inf_detection_raises_invalid_feature_data(self) -> None:
        """Tests that infinite values in features trigger InvalidFeatureData."""
        bad_X = self.X_train.copy()
        bad_X[1, 10, 10] = np.inf
        corrupt_splits = self.test_dir / "corrupt_inf.npz"
        np.savez_compressed(
            corrupt_splits,
            X_composite_train=bad_X,
            y_train=self.y_train,
            X_composite_val=self.X_val,
            y_val=self.y_val,
            X_composite_test=self.X_test,
            y_test=self.y_test,
        )

        with self.assertRaises(InvalidFeatureData) as ctx:
            self.data_loader.load_dataset(splits_filename="corrupt_inf.npz")
        self.assertIn("infinite values", str(ctx.exception))

    def test_invalid_feature_dimensions_raise_error(self) -> None:
        """Tests that wrong feature shape raises InvalidFeatureData."""
        bad_X = np.zeros((self.n_train, 100, 100), dtype=np.float32)
        corrupt_splits = self.test_dir / "corrupt_dim.npz"
        np.savez_compressed(
            corrupt_splits,
            X_composite_train=bad_X,
            y_train=self.y_train,
            X_composite_val=self.X_val,
            y_val=self.y_val,
            X_composite_test=self.X_test,
            y_test=self.y_test,
        )

        with self.assertRaises(InvalidFeatureData) as ctx:
            self.data_loader.load_dataset(splits_filename="corrupt_dim.npz")
        self.assertIn("shape mismatch", str(ctx.exception))

    def test_label_length_mismatch_raises_invalid_label_data(self) -> None:
        """Tests that mismatch between feature samples and label length raises InvalidLabelData."""
        corrupt_splits = self.test_dir / "corrupt_labels_len.npz"
        np.savez_compressed(
            corrupt_splits,
            X_composite_train=self.X_train,
            y_train=np.array([0, 1]),  # Only 2 labels for 10 samples
            X_composite_val=self.X_val,
            y_val=self.y_val,
            X_composite_test=self.X_test,
            y_test=self.y_test,
        )

        with self.assertRaises(InvalidLabelData) as ctx:
            self.data_loader.load_dataset(splits_filename="corrupt_labels_len.npz")
        self.assertIn("count mismatch", str(ctx.exception))

    def test_out_of_bounds_label_id_raises_invalid_label_data(self) -> None:
        """Tests that class label ID >= num_classes raises InvalidLabelData."""
        bad_y = self.y_train.copy()
        bad_y[0] = 99  # Valid range is [0, 4]
        corrupt_splits = self.test_dir / "corrupt_labels_val.npz"
        np.savez_compressed(
            corrupt_splits,
            X_composite_train=self.X_train,
            y_train=bad_y,
            X_composite_val=self.X_val,
            y_val=self.y_val,
            X_composite_test=self.X_test,
            y_test=self.y_test,
        )

        with self.assertRaises(InvalidLabelData) as ctx:
            self.data_loader.load_dataset(splits_filename="corrupt_labels_val.npz")
        self.assertIn("out-of-bounds", str(ctx.exception))

    def test_class_weights_calculation(self) -> None:
        """Tests that balanced class weights are correctly computed for balanced and imbalanced data."""
        # Balanced: 2 of each class (0..4) -> all weights should be 1.0
        balanced_y = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        weights = self.data_loader.calculate_class_weights(balanced_y, num_classes=5)
        for c in range(5):
            self.assertAlmostEqual(weights[c], 1.0, places=2)

        # Imbalanced: class 0 has 8 samples, class 1..4 have 1 sample each (total = 12)
        imbalanced_y = np.array([0] * 8 + [1, 2, 3, 4])
        imb_weights = self.data_loader.calculate_class_weights(imbalanced_y, num_classes=5)
        # Class 0 weight: 12 / (5 * 8) = 0.3
        self.assertAlmostEqual(imb_weights[0], 0.3, places=2)
        # Class 1 weight: 12 / (5 * 1) = 2.4
        self.assertAlmostEqual(imb_weights[1], 2.4, places=2)

    def test_cnn_classifier_creation_and_shapes(self) -> None:
        """Tests that CNNSoundClassifier constructs valid model matching input/output specs."""
        classifier = CNNSoundClassifier(
            input_shape=(self.rows, self.cols, 1),
            num_classes=5,
            class_labels=["ambulance", "car_horn", "fire_alarm", "doorbell", "dog_bark"],
        )

        model = classifier.model
        self.assertEqual(model.input_shape, (None, self.rows, self.cols, 1))
        self.assertEqual(model.output_shape, (None, 5))
        self.assertGreater(model.count_params(), 1000)

    def test_cnn_classifier_prediction_output(self) -> None:
        """Tests that forward pass produces valid softmax probability distribution."""
        classifier = CNNSoundClassifier(
            input_shape=(self.rows, self.cols, 1),
            num_classes=5,
        )

        sample = np.random.randn(self.rows, self.cols, 1).astype(np.float32)
        pred_class, confidence, probs = classifier.predict(sample)

        self.assertIn(pred_class, ["ambulance", "car_horn", "fire_alarm", "doorbell", "dog_bark"])
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertEqual(len(probs), 5)
        self.assertAlmostEqual(sum(probs), 1.0, places=4)

    def test_cnn_model_saving_and_loading(self) -> None:
        """Tests saving trained CNN model to .keras and loading it back."""
        classifier = CNNSoundClassifier(
            input_shape=(self.rows, self.cols, 1),
            num_classes=5,
        )

        model_path = self.test_dir / "test_model.keras"
        save_success = classifier.save_model(model_path)
        self.assertTrue(save_success)
        self.assertTrue(model_path.exists())

        loaded_classifier = CNNSoundClassifier(
            input_shape=(self.rows, self.cols, 1),
            num_classes=5,
        )
        load_success = loaded_classifier.load_model(model_path)
        self.assertTrue(load_success)
        self.assertTrue(loaded_classifier.is_loaded)

        sample = np.random.randn(self.rows, self.cols, 1).astype(np.float32)
        p1, c1, _ = classifier.predict(sample)
        p2, c2, _ = loaded_classifier.predict(sample)
        self.assertEqual(p1, p2)
        self.assertAlmostEqual(c1, c2, places=4)

    def test_model_factory_registration(self) -> None:
        """Tests that ModelFactory instantiates CNNSoundClassifier for 'cnn' and 'sound_classifier'."""
        m_cnn = ModelFactory.create_model("cnn")
        self.assertIsInstance(m_cnn, CNNSoundClassifier)

        m_sc = ModelFactory.create_model("sound_classifier")
        self.assertIsInstance(m_sc, CNNSoundClassifier)

    def test_training_visualizer_generates_plots(self) -> None:
        """Tests that TrainingVisualizer renders and saves plots to disk."""
        vis = TrainingVisualizer(output_dir=self.test_dir / "plots")
        mock_history = {
            "accuracy": [0.4, 0.6, 0.8],
            "val_accuracy": [0.3, 0.5, 0.7],
            "loss": [1.5, 0.9, 0.4],
            "val_loss": [1.7, 1.1, 0.6],
        }

        acc_path = vis.plot_accuracy_curve(mock_history, "acc.png")
        loss_path = vis.plot_loss_curve(mock_history, "loss.png")
        comb_path = vis.plot_combined_metrics(mock_history, "comb.png")

        self.assertTrue(acc_path.exists())
        self.assertTrue(loss_path.exists())
        self.assertTrue(comb_path.exists())
        self.assertGreater(acc_path.stat().st_size, 1000)

    def test_trainer_execution_and_metadata_export(self) -> None:
        """Tests that ModelTrainer executes training and saves metadata & history."""
        dataset = self.data_loader.load_dataset(add_channel_dim=True)

        trainer = ModelTrainer(
            epochs=2,
            batch_size=4,
            save_dir=self.test_dir / "models",
            output_dir=self.test_dir / "outputs",
            random_seed=42,
        )

        history = trainer.train(dataset, epochs=2)

        self.assertIn("accuracy", history)
        self.assertIn("loss", history)
        self.assertEqual(len(history["loss"]), 2)

        # Verify artifacts
        self.assertTrue(trainer.best_model_path.exists())
        self.assertTrue(trainer.final_model_path.exists())
        self.assertTrue(trainer.metadata_path.exists())
        self.assertTrue((trainer.output_dir / "training_history.json").exists())

        # Verify metadata JSON contents
        with open(trainer.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            self.assertEqual(meta["number_of_classes"], 5)
            self.assertEqual(meta["dataset_summary"]["train_samples"], self.n_train)
            self.assertIn("training_performance", meta)

    def test_training_pipeline_end_to_end(self) -> None:
        """Tests complete TrainingPipeline execution on synthetic dataset."""
        pipeline = TrainingPipeline(
            features_dir=self.test_dir,
            model_save_dir=self.test_dir / "pipeline_models",
            output_dir=self.test_dir / "pipeline_outputs",
            epochs=2,
            batch_size=4,
            random_seed=42,
        )

        result = pipeline.run(generate_visualizations=True)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["epochs_completed"], 2)
        self.assertTrue(Path(result["best_model_path"]).exists())
        self.assertTrue(Path(result["metadata_path"]).exists())
        self.assertTrue(Path(result["history_path"]).exists())

        # Check post-training verification
        verification = result["verification"]
        self.assertTrue(verification["load_successful"])
        self.assertTrue(verification["belongs_to_target_classes"])
        self.assertIn(verification["predicted_class"], list(self.class_to_id.keys()))


if __name__ == "__main__":
    unittest.main()
