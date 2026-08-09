"""Unit Test Suite for Audio Feature Extraction Subsystem (Phase 4)."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import scipy.io.wavfile as wavfile

from app.ai.feature_extraction.dataset_splitter import StratifiedDatasetSplitter
from app.ai.feature_extraction.exceptions import (
    FeatureExtractionError,
    FeatureShapeError,
    FeatureStorageError,
    InvalidFeatureError,
    LabelEncodingError,
)
from app.ai.feature_extraction.feature_extractor import FeatureExtractor
from app.ai.feature_extraction.label_encoder import LabelEncoder
from app.ai.feature_extraction.normalizer import FeatureNormalizer
from app.ai.feature_extraction.pipeline import FeatureExtractionPipeline
from app.ai.feature_extraction.storage import FeatureStorageManager
from app.ai.feature_extraction.visualizer import FeatureVisualizer


class TestFeatureExtraction(unittest.TestCase):
    """Test suite verifying acoustic feature extraction, label encoding, scaling, and storage."""

    def setUp(self) -> None:
        """Sets up synthetic 22,050 Hz 4.0-second audio signals and temporary test workspace."""
        self.sample_rate = 22050
        self.duration_sec = 4.0
        self.num_samples = int(self.sample_rate * self.duration_sec)  # 88,200 samples

        # Generate synthetic sine wave signal (440 Hz A tone)
        t = np.linspace(0, self.duration_sec, self.num_samples, endpoint=False)
        self.sine_waveform = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        self.extractor = FeatureExtractor(
            sample_rate=self.sample_rate,
            n_mfcc=40,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            n_chroma=12,
        )

        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        """Cleans up temporary directory."""
        self.temp_dir.cleanup()

    def test_mfcc_extraction_shape_and_values(self) -> None:
        """Tests that MFCC extraction yields valid arrays of shape (n_mfcc, time_steps)."""
        mfcc = self.extractor.extract_mfcc(self.sine_waveform)
        self.assertIsInstance(mfcc, np.ndarray)
        self.assertEqual(mfcc.shape[0], 40)
        self.assertTrue(np.isfinite(mfcc).all())

    def test_mel_spectrogram_extraction_shape_and_values(self) -> None:
        """Tests Log-Mel Spectrogram extraction yields valid arrays of shape (n_mels, time_steps)."""
        mel_spec = self.extractor.extract_mel_spectrogram(self.sine_waveform)
        self.assertIsInstance(mel_spec, np.ndarray)
        self.assertEqual(mel_spec.shape[0], 128)
        self.assertTrue(np.isfinite(mel_spec).all())

    def test_additional_features_extraction(self) -> None:
        """Tests ZCR, Spectral Centroid, Bandwidth, Rolloff, and Chroma feature shapes."""
        zcr = self.extractor.extract_zcr(self.sine_waveform)
        centroid = self.extractor.extract_spectral_centroid(self.sine_waveform)
        bandwidth = self.extractor.extract_spectral_bandwidth(self.sine_waveform)
        rolloff = self.extractor.extract_spectral_rolloff(self.sine_waveform)
        chroma = self.extractor.extract_chroma(self.sine_waveform)

        self.assertEqual(zcr.shape[0], 1)
        self.assertEqual(centroid.shape[0], 1)
        self.assertEqual(bandwidth.shape[0], 1)
        self.assertEqual(rolloff.shape[0], 1)
        self.assertEqual(chroma.shape[0], 12)

        # All features share the same time steps length
        expected_steps = mfcc_steps = zcr.shape[1]
        self.assertEqual(centroid.shape[1], expected_steps)
        self.assertEqual(bandwidth.shape[1], expected_steps)
        self.assertEqual(rolloff.shape[1], expected_steps)
        self.assertEqual(chroma.shape[1], expected_steps)

    def test_composite_matrix_and_vector(self) -> None:
        """Tests stacked 2D composite feature matrix and 1D summary vector extraction."""
        matrix = self.extractor.extract_composite_matrix(self.sine_waveform)
        vector = self.extractor.extract_summary_vector(self.sine_waveform)

        # 128 + 40 + 1 + 1 + 1 + 1 + 12 = 184 total feature rows
        self.assertEqual(matrix.shape[0], 184)
        self.assertEqual(vector.shape[0], 184 * 2)  # mean + std

    def test_invalid_audio_signal_validation(self) -> None:
        """Verifies that empty waveforms or arrays containing NaN/Inf raise InvalidFeatureError."""
        with self.assertRaises(InvalidFeatureError):
            self.extractor.extract_mfcc(np.array([]))

        bad_waveform = self.sine_waveform.copy()
        bad_waveform[10] = np.nan
        with self.assertRaises(InvalidFeatureError):
            self.extractor.extract_mfcc(bad_waveform)

    def test_label_encoder(self) -> None:
        """Tests LabelEncoder string-to-int encoding, int-to-string decoding, and persistence."""
        classes = ["ambulance", "car_horn", "fire_alarm", "doorbell", "dog_bark"]
        encoder = LabelEncoder(target_classes=classes)

        self.assertEqual(encoder.num_classes, 5)
        self.assertEqual(encoder.encode("ambulance"), 0)
        self.assertEqual(encoder.decode(1), "car_horn")

        # Unknown label handling
        with self.assertRaises(LabelEncodingError):
            encoder.encode("unknown_sound")

        with self.assertRaises(LabelEncodingError):
            encoder.decode(99)

        # Persistence test
        json_path = self.test_path / "class_names.json"
        encoder.save_mapping(json_path)
        self.assertTrue(json_path.exists())

        loaded_encoder = LabelEncoder.load_mapping(json_path)
        self.assertEqual(loaded_encoder.classes, encoder.classes)
        self.assertEqual(loaded_encoder.encode("fire_alarm"), 2)

    def test_feature_normalizer(self) -> None:
        """Tests Z-score feature normalization fitting, transformation, and parameter serialization."""
        data = np.random.randn(20, 10).astype(np.float32) * 5.0 + 10.0
        normalizer = FeatureNormalizer(normalization_type="z_score")

        normalized = normalizer.fit_transform(data)
        self.assertTrue(normalizer.is_fitted)

        # Normalized training mean should be approximately 0.0, std approx 1.0
        self.assertAlmostEqual(float(np.mean(normalized)), 0.0, places=3)
        self.assertAlmostEqual(float(np.std(normalized)), 1.0, places=1)

        # JSON parameter persistence
        params_path = self.test_path / "scaler_params.json"
        normalizer.save_params(params_path)

        loaded_normalizer = FeatureNormalizer.load_params(params_path)
        transformed_test = loaded_normalizer.transform(data)
        np.testing.assert_allclose(normalized, transformed_test, rtol=1e-5)

    def test_stratified_dataset_splitter(self) -> None:
        """Tests stratified train/val/test splitting maintaining sample ratios and reproducibility."""
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.array([0] * 20 + [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20)

        splitter = StratifiedDatasetSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
        splits = splitter.split(X, y)

        self.assertEqual(len(splits["X_train"]), 70)
        self.assertEqual(len(splits["X_val"]), 15)
        self.assertEqual(len(splits["X_test"]), 15)

        # Reproducibility check with same seed
        splits2 = splitter.split(X, y)
        np.testing.assert_array_equal(splits["y_train"], splits2["y_train"])

    def test_feature_storage_and_metadata(self) -> None:
        """Tests saving and loading .npz archives and feature_metadata.json files."""
        storage = FeatureStorageManager(output_dir=self.test_path)

        dummy_splits = {
            "X_train": np.ones((10, 5)),
            "y_train": np.zeros(10),
            "X_val": np.ones((2, 5)),
            "y_val": np.zeros(2),
            "X_test": np.ones((2, 5)),
            "y_test": np.zeros(2),
        }
        npz_file = storage.save_dataset_splits(dummy_splits, "test_splits.npz")
        self.assertTrue(npz_file.exists())

        loaded_splits = storage.load_dataset_splits("test_splits.npz")
        self.assertEqual(loaded_splits["X_train"].shape, (10, 5))

        meta_file = storage.save_feature_metadata(
            num_samples=14,
            feature_dimensions={"composite": (184, 173)},
            class_distribution={"ambulance": 14},
        )
        self.assertTrue(meta_file.exists())

        metadata = storage.load_feature_metadata()
        self.assertEqual(metadata["number_of_samples"], 14)
        self.assertIn("ambulance", metadata["class_distribution"])

    def test_end_to_end_batch_pipeline(self) -> None:
        """Tests full FeatureExtractionPipeline batch processing on synthetic WAV files."""
        processed_dir = self.test_path / "processed"
        features_dir = self.test_path / "features"
        classes = ["ambulance", "car_horn", "fire_alarm", "doorbell", "dog_bark"]

        # Create dummy processed class directories with WAV files
        for cls in classes:
            cls_dir = processed_dir / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(3):  # 3 files per class = 15 total clips
                wav_path = cls_dir / f"clip_{i}.wav"
                scaled_signal = (self.sine_waveform * 32767).astype(np.int16)
                wavfile.write(wav_path, self.sample_rate, scaled_signal)

        pipeline = FeatureExtractionPipeline(processed_dir=processed_dir, output_dir=features_dir)
        summary = pipeline.run(generate_visualizations=True)

        self.assertEqual(summary["total_audio_files"], 15)
        self.assertEqual(summary["successfully_processed"], 15)
        self.assertEqual(summary["failed_files_count"], 0)

        # Verify output files exist
        self.assertTrue((features_dir / "dataset_splits.npz").exists())
        self.assertTrue((features_dir / "class_names.json").exists())
        self.assertTrue((features_dir / "scaler_params.json").exists())
        self.assertTrue((features_dir / "feature_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
