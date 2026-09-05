"""Domain Exceptions for AI Models Subsystem."""


class ModelError(Exception):
    """Base exception for all AI model errors."""
    pass


class ModelSaveError(ModelError):
    """Raised when saving or serializing model weights/architecture fails."""
    pass


class ModelLoadError(ModelError):
    """Raised when loading or deserializing model weights fails."""
    pass


class ModelTrainingError(ModelError):
    """Raised when model building or training fails."""
    pass
