"""Unit tests for Phase 3 Audio Preprocessing Subsystem."""

import json
from pathlib import Path
import tempfile
import unittest
import wave

from config import settings
from app.ai.preprocessing import (
    AudioLoader,
    AudioStandardizer,
    SilenceProcessor,
    NoiseReducer,
    LengthStandardizer,
    MetadataGenerator,
    PreprocessingPipeline,
    RawAudioData,
    AudioLoadError,
    UnsupportedFormatError,
)


def create_sample_wav(file_path: Path, duration_sec: float = 2.0, sample_rate: int = 16000, channels: int = 1) -> Path:
    """Helper utility creating a sample PCM WAV file for testing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_sec * sample_rate * channels)
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        # Write non-zero test samples
        samples = [int(16000.0 * (i % 100 / 100.0)) for i in range(num_samples)]
        import struct
        raw_bytes = struct.pack(f"<{num_samples}h", *samples)
        wf.writeframes(raw_bytes)
    return file_path


class TestAudioPreprocessing(unittest.TestCase):
    """Test suite for audio loading, standardization, silence trimming, noise reduction, and length standardization."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.raw_dir = self.root_path / "raw"
        self.processed_dir = self.root_path / "processed"
        self.target_classes = ("ambulance", "car_horn", "fire_alarm", "doorbell", "dog_bark")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_audio_loader_valid_file(self) -> None:
        """Verifies loading WAV audio signal into RawAudioData."""
        file_path = create_sample_wav(self.raw_dir / "doorbell" / "chime.wav", duration_sec=1.5, sample_rate=16000)
        loader = AudioLoader()
        raw_data = loader.load_audio(file_path, class_label="doorbell")

        self.assertIsInstance(raw_data, RawAudioData)
        self.assertEqual(raw_data.sample_rate, 16000)
        self.assertEqual(raw_data.channels, 1)
        self.assertEqual(raw_data.class_label, "doorbell")
        self.assertTrue(len(raw_data.waveform) > 0)

    def test_audio_standardizer(self) -> None:
        """Verifies mono conversion, 22050 Hz resampling, and peak normalization."""
        standardizer = AudioStandardizer(target_sample_rate=22050, target_channels=1)

        # Test mono conversion (stereo 2 channels)
        stereo_wave = [0.2, 0.4, 0.6, 0.8]  # 2 frames
        mono_wave = standardizer.convert_to_mono(stereo_wave, channels=2)
        self.assertEqual(len(mono_wave), 2)
        self.assertAlmostEqual(mono_wave[0], 0.3)
        self.assertAlmostEqual(mono_wave[1], 0.7)

        # Test resampling (16000 Hz to 22050 Hz)
        input_wave = [0.1 * i for i in range(16000)]
        resampled = standardizer.resample(input_wave, orig_sr=16000, target_sr=22050)
        self.assertEqual(len(resampled), 22050)

        # Test peak normalization
        raw_wave = [0.0, 0.5, -0.2]
        normalized = standardizer.normalize_peak_amplitude(raw_wave, target_peak=0.95)
        self.assertAlmostEqual(max(abs(v) for v in normalized), 0.95)

    def test_silence_processor(self) -> None:
        """Verifies trimming leading and trailing silence."""
        silence_proc = SilenceProcessor(threshold_db=-40.0)

        # Create signal with leading zeros, audio pulse, and trailing zeros
        silent_prefix = [0.0] * 1000
        audio_signal = [0.5, -0.5] * 2000
        silent_suffix = [0.0] * 1000
        full_wave = silent_prefix + audio_signal + silent_suffix

        trimmed = silence_proc.trim_silence(full_wave, frame_length=256)
        self.assertTrue(len(trimmed) < len(full_wave))
        self.assertTrue(len(trimmed) >= len(audio_signal))

    def test_noise_reducer(self) -> None:
        """Verifies optional background noise reduction filter."""
        reducer_enabled = NoiseReducer(enabled=True)
        reducer_disabled = NoiseReducer(enabled=False)

        signal = [0.1, 0.9, 0.2, 0.8, 0.1]
        out_enabled = reducer_enabled.reduce_noise(signal)
        out_disabled = reducer_disabled.reduce_noise(signal)

        self.assertNotEqual(out_enabled, signal)
        self.assertEqual(out_disabled, signal)

    def test_length_standardizer(self) -> None:
        """Verifies fixed 4.0s (88,200 samples at 22050Hz) length trimming and zero-padding."""
        length_std = LengthStandardizer(target_duration_sec=4.0, target_sample_rate=22050)
        self.assertEqual(length_std.target_samples, 88200)

        # Test padding short signal (2.0s = 44,100 samples)
        short_wave = [0.5] * 44100
        padded = length_std.standardize_length(short_wave)
        self.assertEqual(len(padded), 88200)
        self.assertEqual(padded[:44100], short_wave)
        self.assertEqual(padded[44100:], [0.0] * 44100)

        # Test trimming long signal (5.0s = 110,250 samples)
        long_wave = [0.5] * 110250
        trimmed = length_std.standardize_length(long_wave)
        self.assertEqual(len(trimmed), 88200)

    def test_preprocessing_pipeline_batch(self) -> None:
        """Verifies end-to-end batch processing pipeline across raw dataset folders."""
        # Create sample raw dataset files
        create_sample_wav(self.raw_dir / "ambulance" / "amb1.wav", duration_sec=1.5)
        create_sample_wav(self.raw_dir / "doorbell" / "chime1.wav", duration_sec=5.0)

        pipeline = PreprocessingPipeline(
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
        )

        summary = pipeline.process_dataset(overwrite=True)

        self.assertEqual(summary.total_files, 2)
        self.assertEqual(summary.processed_count, 2)
        self.assertEqual(summary.error_count, 0)

        # Verify output files created under processed_dir
        amb_out = self.processed_dir / "ambulance" / "amb1.wav"
        chime_out = self.processed_dir / "doorbell" / "chime1.wav"
        self.assertTrue(amb_out.exists())
        self.assertTrue(chime_out.exists())

        # Verify metadata JSON report exported
        json_meta_path = self.processed_dir / "preprocessed_metadata.json"
        self.assertTrue(json_meta_path.exists())

        with open(json_meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["total_files"], 2)


if __name__ == "__main__":
    unittest.main()
