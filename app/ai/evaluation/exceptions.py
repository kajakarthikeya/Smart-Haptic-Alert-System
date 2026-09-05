"""Domain Exceptions for AI Model Evaluation Subsystem."""


class EvaluationError(Exception):
    """Base exception for all errors originating within the model evaluation subsystem."""
    pass


class InvalidEvaluationData(EvaluationError):
    """Raised when evaluation test features/labels are corrupted, empty, contain NaNs/Infs, or have invalid dimensions."""
    pass


class ModelLoadError(EvaluationError):
    """Raised when loading or deserializing the trained model artifact fails."""
    pass


class PredictionError(EvaluationError):
    """Raised when inference or prediction generation on test features encounters an error."""
    pass


class MetricCalculationError(EvaluationError):
    """Raised when computing classification metrics, confusion matrices, or statistical aggregations fails."""
    pass


class ReportGenerationError(EvaluationError):
    """Raised when generating or exporting evaluation reports, CSVs, or JSON summaries fails."""
    pass


class VisualizationError(EvaluationError):
    """Raised when generating or saving evaluation plots and visual figures fails."""
    pass
