"""
AI Model Evaluation module for the Smart Haptic Alert System.

This module provides end-to-end evaluation capabilities for trained environmental sound
classification models on unseen test datasets.
"""

from app.ai.evaluation.confusion_matrix import ConfusionMatrixGenerator
from app.ai.evaluation.data_loader import EvaluationData, EvaluationDataLoader
from app.ai.evaluation.evaluator import ModelEvaluator
from app.ai.evaluation.exceptions import (
    EvaluationError,
    InvalidEvaluationData,
    MetricCalculationError,
    ModelLoadError,
    PredictionError,
    ReportGenerationError,
    VisualizationError,
)
from app.ai.evaluation.metrics import EvaluationMetricsCalculator
from app.ai.evaluation.prediction_analyzer import PredictionAnalyzer, PredictionRecord
from app.ai.evaluation.report_generator import EvaluationReportGenerator
from app.ai.evaluation.visualization import EvaluationVisualizer

__all__ = [
    "ModelEvaluator",
    "EvaluationDataLoader",
    "EvaluationData",
    "EvaluationMetricsCalculator",
    "ConfusionMatrixGenerator",
    "PredictionAnalyzer",
    "PredictionRecord",
    "EvaluationVisualizer",
    "EvaluationReportGenerator",
    "EvaluationError",
    "InvalidEvaluationData",
    "ModelLoadError",
    "PredictionError",
    "MetricCalculationError",
    "ReportGenerationError",
    "VisualizationError",
]
