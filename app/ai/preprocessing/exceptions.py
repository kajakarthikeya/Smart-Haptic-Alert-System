"""Custom Exceptions for Audio Preprocessing Subsystem."""

class PreprocessingError(Exception):
    """Base exception for audio preprocessing subsystem errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class AudioLoadError(PreprocessingError):
    """Raised when an audio file cannot be loaded or opened."""

    def __init__(self, file_path: str, reason: str) -> None:
        message = f"Failed to load audio file '{file_path}': {reason}"
        super().__init__(message)
        self.file_path = file_path
        self.reason = reason


class UnsupportedFormatError(PreprocessingError):
    """Raised when an unsupported audio format is provided for preprocessing."""

    def __init__(self, format_ext: str) -> None:
        message = f"Audio format '{format_ext}' is not supported for preprocessing."
        super().__init__(message)
        self.format_ext = format_ext


class ProcessingError(PreprocessingError):
    """Raised when an error occurs during audio signal processing algorithms."""

    def __init__(self, stage: str, reason: str) -> None:
        message = f"Audio processing error during [{stage}]: {reason}"
        super().__init__(message)
        self.stage = stage
        self.reason = reason


class CorruptedAudioError(PreprocessingError):
    """Raised when audio payload or header is corrupted or unreadable."""

    def __init__(self, file_path: str, reason: str) -> None:
        message = f"Corrupted audio file detected at '{file_path}': {reason}"
        super().__init__(message)
        self.file_path = file_path
        self.reason = reason
