"""Unit tests for Phase 2 Dataset Management module."""

import json
from pathlib import Path
import tempfile
import unittest
import wave
import struct

from config import settings
from app.ai.dataset import (
    DatasetDirectoryManager,
    AudioDatasetLoader,
    DatasetValidator,
    DatasetStatisticsCalculator,
    DatasetExplorer,
    DatasetNotFoundError,
    CorruptedAudioError,
    ValidationSeverity,
)


def create_dummy_wav(file_path: Path, duration_sec: float = 1.0, sample_rate: int = 16000) -> Path:
    """Helper utility creating a valid dummy PCM WAV file for testing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_sec * sample_rate)
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        # Write silent PCM data
        wf.writeframes(b"\x00\x00" * num_samples)
    return file_path


class TestDatasetManagement(unittest.TestCase):
    """Test suite for dataset directory management, loading, validation, statistics, and exploration."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.raw_dir = self.root_path / "raw"
        self.processed_dir = self.root_path / "processed"
        self.test_dir = self.root_path / "test_audio"
        self.target_classes = ("ambulance", "car_horn", "fire_alarm", "doorbell", "dog_bark")

        self.dir_mgr = DatasetDirectoryManager(
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
            test_audio_dir=self.test_dir,
            target_classes=self.target_classes,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_directory_manager_creation(self) -> None:
        """Verifies automatic dataset folder hierarchy creation."""
        paths = self.dir_mgr.initialize_directories()
        self.assertTrue(self.raw_dir.exists())
        self.assertTrue(self.processed_dir.exists())
        self.assertTrue(self.test_dir.exists())

        for cls_name in self.target_classes:
            self.assertTrue((self.raw_dir / cls_name).exists())

        status = self.dir_mgr.verify_structure()
        self.assertTrue(all(status.values()))

    def test_dataset_loader_and_manifest(self) -> None:
        """Verifies loading valid WAV audio files into a DatasetManifest."""
        self.dir_mgr.initialize_directories()

        # Create dummy WAV files for doorbell and ambulance
        f1 = create_dummy_wav(self.raw_dir / "doorbell" / "doorbell_01.wav", duration_sec=1.5)
        f2 = create_dummy_wav(self.raw_dir / "ambulance" / "siren_01.wav", duration_sec=2.0)

        loader = AudioDatasetLoader(target_classes=self.target_classes)
        manifest = loader.load_dataset(self.raw_dir)

        self.assertEqual(manifest.total_count, 2)
        self.assertIn("doorbell", manifest.class_counts)
        self.assertIn("ambulance", manifest.class_counts)
        self.assertEqual(manifest.class_counts["doorbell"], 1)
        self.assertEqual(manifest.class_counts["ambulance"], 1)

    def test_dataset_validator_checks(self) -> None:
        """Verifies validator detects empty folders, unsupported formats, duplicates, and invalid filenames."""
        self.dir_mgr.initialize_directories()

        # Create 1 valid file
        f1 = create_dummy_wav(self.raw_dir / "doorbell" / "doorbell_01.wav")
        # Create duplicate file in car_horn
        f2 = create_dummy_wav(self.raw_dir / "car_horn" / "horn_copy.wav")
        # Create file with unsupported extension
        txt_file = self.raw_dir / "fire_alarm" / "info.txt"
        txt_file.write_text("not an audio file")
        # Create file with invalid filename space
        bad_name_file = create_dummy_wav(self.raw_dir / "dog_bark" / "dog bark 01.wav")

        validator = DatasetValidator(target_classes=self.target_classes)
        report = validator.validate_directory(self.raw_dir)

        self.assertEqual(report.total_files_checked, 4)
        self.assertTrue(len(report.duplicate_files) > 0)
        self.assertIn(str(txt_file.relative_to(self.raw_dir)), report.unsupported_format_files)
        self.assertIn(str(bad_name_file.relative_to(self.raw_dir)), report.invalid_filename_files)

    def test_statistics_calculator(self) -> None:
        """Verifies computation of statistics and JSON export."""
        self.dir_mgr.initialize_directories()

        create_dummy_wav(self.raw_dir / "doorbell" / "chime1.wav", duration_sec=1.0)
        create_dummy_wav(self.raw_dir / "doorbell" / "chime2.wav", duration_sec=3.0)

        loader = AudioDatasetLoader(target_classes=self.target_classes)
        manifest = loader.load_dataset(self.raw_dir)

        calc = DatasetStatisticsCalculator(output_dir=self.root_path)
        stats = calc.compute_statistics(manifest)

        self.assertEqual(stats.total_files, 2)
        self.assertEqual(stats.duration_min_sec, 1.0)
        self.assertEqual(stats.duration_max_sec, 3.0)
        self.assertEqual(stats.duration_avg_sec, 2.0)

        json_path = calc.export_json_report(stats, "test_stats.json")
        self.assertTrue(json_path.exists())

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["total_files"], 2)

    def test_dataset_explorer(self) -> None:
        """Verifies DatasetExplorer search and preview functionalities."""
        self.dir_mgr.initialize_directories()

        f1 = create_dummy_wav(self.raw_dir / "ambulance" / "amb_01.wav", duration_sec=1.0)
        f2 = create_dummy_wav(self.raw_dir / "ambulance" / "amb_02.wav", duration_sec=4.0)
        f3 = create_dummy_wav(self.raw_dir / "fire_alarm" / "alarm_01.wav", duration_sec=2.0)

        loader = AudioDatasetLoader(target_classes=self.target_classes)
        manifest = loader.load_dataset(self.raw_dir)

        explorer = DatasetExplorer(manifest)
        self.assertEqual(explorer.count_files("ambulance"), 2)
        self.assertEqual(explorer.count_files("fire_alarm"), 1)

        # Preview samples
        samples = explorer.preview_samples("ambulance", limit=1)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].label, "ambulance")

        # Search with duration filter
        short_samples = explorer.search_files(max_duration_sec=2.5)
        self.assertEqual(len(short_samples), 2)  # amb_01 (1s) & alarm_01 (2s)


if __name__ == "__main__":
    unittest.main()
