"""Custom Exceptions for Dataset Management Subsystem."""

class DatasetError(Exception):
    """Base exception for all dataset domain errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class DatasetNotFoundError(DatasetError):
    """Raised when the specified dataset directory or file does not exist."""
    pass


class MissingClassError(DatasetError):
    """Raised when one or more required sound class directories or samples are missing."""
    pass


class CorruptedAudioError(DatasetError):
    """Raised when an audio file cannot be read, header is damaged, or file is truncated."""

    def __init__(self, file_path: str, reason: str) -> None:
        message = f"Corrupted audio file detected at '{file_path}': {reason}"
        super().__init__(message)
        self.file_path = file_path
        self.reason = reason


class InvalidDatasetError(DatasetError):
    """Raised when dataset validation checks fail critical threshold requirements."""
    pass
