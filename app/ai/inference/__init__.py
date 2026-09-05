"""
Real-Time Sound Recognition and Inference Module for the Smart Haptic Alert System.

Provides high-performance, modular audio capture, signal preprocessing,
feature extraction, CNN classification, confidence thresholding, and prediction stability consensus.
"""

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

__all__ = [
    "RealtimeSoundRecognizer",
    "MicrophoneAudioCapture",
    "AudioDeviceManager",
    "AudioDeviceInfo",
    "RealtimeFeaturePipeline",
    "InferenceModelLoader",
    "PredictionResult",
    "PredictionStatus",
    "PredictionStabilizer",
    "LatencyMetrics",
    "InferenceError",
    "AudioCaptureError",
    "MicrophoneInitializationError",
    "DeviceNotFoundError",
    "RealtimeInferenceError",
    "FeaturePipelineError",
    "ModelLoadingError",
    "PredictionError",
    "ConfigurationError",
]
