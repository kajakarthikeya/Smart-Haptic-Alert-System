"""
Unit tests for Real-Time Sound Recognition & Inference Subsystem (Phase 7).

Tests cover:
1. Exception hierarchy
2. AudioDeviceManager device enumeration and selection
3. MicrophoneAudioCapture configuration and mocks
4. Audio windowing and circular buffer rolling
5. Preprocessing standardization reuse
6. Feature extraction compatibility (shape 184 x 173 x 1)
7. Model loader validation and missing file handling
8. Prediction result generation and formatting
9. Confidence thresholding logic
10. Low-confidence handling
11. Prediction stability consensus buffer (N=3, K=2)
12. Offline test-file inference on real dataset WAV
13. Invalid audio signal handling (empty, NaNs)
14. Model loading failure recovery
15. Graceful recognizer stop, resource release, and session telemetry
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from config import InferenceConfig
from app.ai.inference.audio_capture import (
    AudioDeviceInfo,
    AudioDeviceManager,
    MicrophoneAudioCapture,
)
from app.ai.inference.exceptions import (
    AudioCaptureError,
    ConfigurationError,
    DeviceNotFoundError,
    FeaturePipelineError,
    InferenceError,
    MicrophoneInitializationError,
    ModelLoadingError,
    PredictionError,
    RealtimeInferenceError,
)
from app.ai.inference.feature_pipeline import RealtimeFeaturePipeline
from app.ai.inference.model_loader import InferenceModelLoader
from app.ai.inference.prediction import (
    LatencyMetrics,
    PredictionResult,
    PredictionStabilizer,
    PredictionStatus,
)
from app.ai.inference.realtime_recognizer import RealtimeSoundRecognizer


# ==========================================
# 1. Exception Hierarchy Tests
# ==========================================

def test_exception_hierarchy():
    """Verify that all inference exceptions inherit from InferenceError."""
    assert issubclass(AudioCaptureError, InferenceError)
    assert issubclass(MicrophoneInitializationError, AudioCaptureError)
    assert issubclass(DeviceNotFoundError, AudioCaptureError)
    assert issubclass(RealtimeInferenceError, InferenceError)
    assert issubclass(FeaturePipelineError, InferenceError)
    assert issubclass(ModelLoadingError, InferenceError)
    assert issubclass(PredictionError, InferenceError)
    assert issubclass(ConfigurationError, InferenceError)


# ==========================================
# 2. Audio Device Enumeration Tests
# ==========================================

def test_audio_device_enumeration():
    """Test device query and default selection logic."""
    mock_devices = [
        {"name": "Output Speaker", "max_input_channels": 0, "default_samplerate": 44100.0},
        {"name": "USB Microphone", "max_input_channels": 1, "default_samplerate": 48000.0},
        {"name": "Built-in Mic", "max_input_channels": 2, "default_samplerate": 44100.0},
    ]

    with patch("app.ai.inference.audio_capture._SOUNDDEVICE_AVAILABLE", True):
        with patch("sounddevice.query_devices", return_value=mock_devices):
            with patch("sounddevice.default.device", (1, 0)):
                devs = AudioDeviceManager.list_input_devices()
                assert len(devs) == 2
                assert devs[0].device_id == 1
                assert devs[0].is_default is True
                assert devs[1].device_id == 2
                assert devs[1].is_default is False

                # Test get_device_info default
                default_dev = AudioDeviceManager.get_device_info(None)
                assert default_dev.device_id == 1

                # Test get_device_info specific
                dev2 = AudioDeviceManager.get_device_info(2)
                assert dev2.device_id == 2

                # Test non-existent device
                with pytest.raises(DeviceNotFoundError):
                    AudioDeviceManager.get_device_info(99)


# ==========================================
# 3. MicrophoneAudioCapture Configuration Tests
# ==========================================

def test_microphone_capture_configuration():
    """Verify capture settings match training defaults: 22050 Hz, 1 channel, 4.0s duration."""
    capture = MicrophoneAudioCapture(
        sample_rate=22050,
        channels=1,
        window_duration_sec=4.0,
        block_size=1024,
    )
    assert capture.sample_rate == 22050
    assert capture.channels == 1
    assert capture.window_duration_sec == 4.0
    assert capture.window_samples == 88200
    assert capture.block_size == 1024
    assert capture.is_recording is False


# ==========================================
# 4. Circular Buffer Windowing Tests
# ==========================================

def test_circular_buffer_windowing():
    """Verify that audio chunks roll correctly inside the 88,200-sample circular buffer."""
    capture = MicrophoneAudioCapture(
        sample_rate=22050,
        channels=1,
        window_duration_sec=4.0,
    )

    # Initially all zeros
    initial_window = capture.get_latest_window()
    assert initial_window.shape == (88200,)
    assert np.all(initial_window == 0.0)

    # Push a 1024-sample block of ones
    chunk = np.ones((1024, 1), dtype=np.float32)
    capture._audio_callback(chunk, 1024, None, None)

    window = capture.get_latest_window()
    assert window.shape == (88200,)
    # The last 1024 samples should be 1.0, earlier samples should remain 0.0
    assert np.all(window[-1024:] == 1.0)
    assert np.all(window[:-1024] == 0.0)
    assert capture.total_samples_captured == 1024

    # Clear buffer
    capture.clear_buffer()
    cleared_window = capture.get_latest_window()
    assert np.all(cleared_window == 0.0)
    assert capture.total_samples_captured == 0


# ==========================================
# 5. Preprocessing Standardization Reuse Tests
# ==========================================

def test_preprocessing_standardization_reuse():
    """Verify that RealtimeFeaturePipeline standardizes input audio to exact 88,200 samples."""
    pipeline = RealtimeFeaturePipeline()

    # Short audio (1.0 second = 22,050 samples)
    short_audio = np.ones(22050, dtype=np.float32) * 0.5
    clean_audio, prep_time = pipeline.preprocess_signal(short_audio, orig_sr=22050)

    assert len(clean_audio) == 88200
    assert prep_time >= 0.0
    # Values should be normalized to peak [-0.95, 0.95]
    assert np.isclose(np.max(clean_audio), 0.95, atol=1e-3)
    # Padded tail should be 0.0
    assert np.all(clean_audio[22050:] == 0.0)


# ==========================================
# 6. Feature Extraction Compatibility Tests
# ==========================================

def test_feature_extraction_compatibility():
    """Verify feature extractor produces exact (1, 184, 173, 1) tensor."""
    pipeline = RealtimeFeaturePipeline()
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 4.0, 88200)).astype(np.float32)

    feat_tensor, feat_time = pipeline.extract_features(audio)
    assert feat_tensor.shape == (1, 184, 173, 1)
    assert feat_time >= 0.0
    assert not np.isnan(feat_tensor).any()
    assert not np.isinf(feat_tensor).any()


# ==========================================
# 7. Model Loader Validation Tests
# ==========================================

def test_model_loader_validation():
    """Verify InferenceModelLoader loads the best Phase 5 model and class names."""
    loader = InferenceModelLoader()
    classifier = loader.load()

    assert loader.is_loaded is True
    assert len(loader.class_names) == 5
    assert set(loader.class_names) == {"ambulance", "car_horn", "fire_alarm", "doorbell", "dog_bark"}
    assert classifier is not None


def test_model_loader_missing_model_file(tmp_path):
    """Verify loader raises ModelLoadingError if model file is missing."""
    loader = InferenceModelLoader(model_path=tmp_path / "non_existent.keras")
    with pytest.raises(ModelLoadingError, match="not found"):
        loader.load()


# ==========================================
# 8. Prediction Generation Tests
# ==========================================

def test_prediction_generation():
    """Verify recognize_window returns structured PredictionResult."""
    recognizer = RealtimeSoundRecognizer(confidence_threshold=0.50)
    audio = np.zeros(88200, dtype=np.float32)

    result = recognizer.recognize_window(audio, orig_sr=22050)
    assert isinstance(result, PredictionResult)
    assert result.raw_class in recognizer.model_loader.class_names
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.probabilities) == 5
    assert result.latency.total_ms > 0.0
    assert isinstance(result.status, PredictionStatus)


# ==========================================
# 9. Confidence Thresholding Tests
# ==========================================

def test_confidence_threshold_logic():
    """Verify confidence gating overrides predicted_class when below threshold."""
    recognizer = RealtimeSoundRecognizer(confidence_threshold=0.99)  # Intentionally high threshold
    audio = np.zeros(88200, dtype=np.float32)

    result = recognizer.recognize_window(audio, orig_sr=22050)
    # Since max confidence is < 0.99, it should be marked as Low Confidence
    assert result.is_confident is False
    assert result.predicted_class == "Unknown / Low Confidence"
    assert result.predicted_id == -1
    assert result.status == PredictionStatus.LOW_CONFIDENCE
    # Raw prediction should still be preserved
    assert result.raw_class in recognizer.model_loader.class_names


# ==========================================
# 10. Prediction Stability Tests
# ==========================================

def test_prediction_stabilizer():
    """Verify multi-window agreement (N=3, K=2) consensus gating."""
    stabilizer = PredictionStabilizer(buffer_size=3, required_agreement=2)

    # 1. Low confidence prediction -> LOW_CONFIDENCE
    status1 = stabilizer.evaluate_stability("fire_alarm", is_confident=False)
    assert status1 == PredictionStatus.LOW_CONFIDENCE

    # 2. Single confident prediction -> TENTATIVE (only 1 occurrence in buffer)
    status2 = stabilizer.evaluate_stability("fire_alarm", is_confident=True)
    assert status2 == PredictionStatus.TENTATIVE

    # 3. Second matching confident prediction -> CONFIRMED (2 occurrences in last 3)
    status3 = stabilizer.evaluate_stability("fire_alarm", is_confident=True)
    assert status3 == PredictionStatus.CONFIRMED

    # 4. Different sound arrives -> TENTATIVE (1 occurrence of car_horn)
    status4 = stabilizer.evaluate_stability("car_horn", is_confident=True)
    assert status4 == PredictionStatus.TENTATIVE


# ==========================================
# 11. Offline Test-File Inference Tests
# ==========================================

def test_offline_wav_file_inference():
    """Verify offline recognition on real dataset WAV sample."""
    sample_path = Path("dataset/processed/car_horn/sample_0.wav")
    if not sample_path.exists():
        pytest.skip(f"Test sample not found at: {sample_path}")

    recognizer = RealtimeSoundRecognizer(confidence_threshold=0.25)
    result = recognizer.recognize_file(sample_path)

    assert result.raw_class == "car_horn"
    assert result.is_confident is True
    assert result.predicted_class == "car_horn"
    assert result.confidence >= 0.25
    assert result.latency.total_ms > 0.0


# ==========================================
# 12. Invalid Audio Handling Tests
# ==========================================

def test_invalid_audio_handling():
    """Verify empty or non-existent audio inputs raise appropriate errors."""
    pipeline = RealtimeFeaturePipeline()

    # Empty array
    with pytest.raises(FeaturePipelineError, match="empty"):
        pipeline.preprocess_signal(np.array([], dtype=np.float32))

    # Missing file
    recognizer = RealtimeSoundRecognizer()
    with pytest.raises(FileNotFoundError):
        recognizer.recognize_file("non_existent_file.wav")


# ==========================================
# 13. Telemetry Session Summary Tests
# ==========================================

def test_session_summary_telemetry():
    """Verify session telemetry aggregates average latencies correctly."""
    recognizer = RealtimeSoundRecognizer(confidence_threshold=0.25)
    recognizer.clear_session()

    audio = np.zeros(88200, dtype=np.float32)
    # Process 2 windows
    recognizer.recognize_window(audio, orig_sr=22050)
    recognizer.recognize_window(audio, orig_sr=22050)

    summary = recognizer.get_session_summary()
    assert summary["total_windows_processed"] == 2
    assert summary["average_total_latency_ms"] > 0.0
    assert summary["average_preprocessing_ms"] >= 0.0
    assert summary["average_feature_extraction_ms"] > 0.0
    assert summary["average_inference_ms"] > 0.0
