"""Custom Exception Classes for Audio Feature Extraction Subsystem."""


class FeatureExtractionError(Exception):
    """Base exception for all feature extraction subsystem errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidFeatureError(FeatureExtractionError):
    """Raised when extracted audio features contain missing, empty, NaN, or Infinite values."""

    def __init__(self, feature_name: str, reason: str) -> None:
        message = f"Invalid feature detected for '{feature_name}': {reason}"
        super().__init__(message)
        self.feature_name = feature_name
        self.reason = reason


class FeatureShapeError(FeatureExtractionError):
    """Raised when feature matrix/vector shape violates configured expectations."""

    def __init__(self, expected_shape: tuple, actual_shape: tuple) -> None:
        message = f"Feature shape mismatch: expected {expected_shape}, got {actual_shape}."
        super().__init__(message)
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape


class LabelEncodingError(FeatureExtractionError):
    """Raised when encoding sound class names to numerical IDs or decoding fails."""

    def __init__(self, label: str, reason: str) -> None:
        message = f"Label encoding failure for '{label}': {reason}"
        super().__init__(message)
        self.label = label
        self.reason = reason


class FeatureStorageError(FeatureExtractionError):
    """Raised when reading from or writing feature arrays, metadata, or maps fails."""

    def __init__(self, path: str, reason: str) -> None:
        message = f"Feature storage operation failed at '{path}': {reason}"
        super().__init__(message)
        self.path = path
        self.reason = reason
