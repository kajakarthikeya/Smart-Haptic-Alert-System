"""Domain Exceptions for AI Model Training Subsystem."""


class TrainingSubsystemError(Exception):
    """Base exception for all errors originating within the model training subsystem."""
    pass


class InvalidFeatureData(TrainingSubsystemError):
    """Raised when training feature arrays are corrupt, empty, contain NaNs/Infs, or have invalid shapes."""
    pass


class InvalidLabelData(TrainingSubsystemError):
    """Raised when labels are out of range, have invalid formats, or do not match feature counts."""
    pass


class ModelTrainingError(TrainingSubsystemError):
    """Raised when an error occurs during model initialization, compilation, or the training loop."""
    pass


class ModelSaveError(TrainingSubsystemError):
    """Raised when saving, serializing, or exporting model artifacts fails."""
    pass


class ConfigurationError(TrainingSubsystemError):
    """Raised when training configuration parameters are missing or invalid."""
    pass
