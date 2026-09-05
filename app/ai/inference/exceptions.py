"""Domain Exceptions for Real-Time Sound Recognition & Inference Subsystem."""


class InferenceError(Exception):
    """Base exception for all errors originating within the real-time inference subsystem."""
    pass


class AudioCaptureError(InferenceError):
    """Raised when audio capture from the microphone stream fails or encounters an error."""
    pass


class MicrophoneInitializationError(AudioCaptureError):
    """Raised when failing to open, configure, or initialize the audio input device."""
    pass


class DeviceNotFoundError(AudioCaptureError):
    """Raised when a requested audio input device index or name cannot be located on the host."""
    pass


class RealtimeInferenceError(InferenceError):
    """Raised when errors occur during real-time streaming recognition execution."""
    pass


class FeaturePipelineError(InferenceError):
    """Raised when real-time signal preprocessing or acoustic feature extraction fails."""
    pass


class ModelLoadingError(InferenceError):
    """Raised when loading or validating the serialized model or metadata fails."""
    pass


class PredictionError(InferenceError):
    """Raised when classification inference or confidence evaluation encounters a failure."""
    pass


class ConfigurationError(InferenceError):
    """Raised when inference configuration parameters are invalid or contradictory."""
    pass
