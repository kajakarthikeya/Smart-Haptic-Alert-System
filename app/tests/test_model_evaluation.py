"""
Unit tests for AI Model Evaluation module (Phase 6).

Tests cover:
- Exception hierarchy
- EvaluationDataLoader data integrity and validation
- EvaluationMetricsCalculator precision, recall, F1, and cross-entropy loss
- ConfusionMatrixGenerator raw and normalized calculations
- PredictionAnalyzer prediction records, confidence stats, error filtering, and CSV export
- EvaluationVisualizer plotting and file creation
- EvaluationReportGenerator JSON and text exports
- ModelEvaluator end-to-end pipeline execution
"""

import json
from pathlib import Path
from typing import Any
import numpy as np
import pytest

from config import EvaluationConfig
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


# ==========================================
# 1. Exception Hierarchy Tests
# ==========================================

def test_exception_hierarchy():
    """Verify that all domain evaluation exceptions inherit from EvaluationError."""
    assert issubclass(InvalidEvaluationData, EvaluationError)
    assert issubclass(ModelLoadError, EvaluationError)
    assert issubclass(PredictionError, EvaluationError)
    assert issubclass(MetricCalculationError, EvaluationError)
    assert issubclass(ReportGenerationError, EvaluationError)
    assert issubclass(VisualizationError, EvaluationError)


# ==========================================
# 2. EvaluationDataLoader Tests
# ==========================================

def test_data_loader_missing_features_file(tmp_path):
    """DataLoader should raise InvalidEvaluationData if features file does not exist."""
    config = EvaluationConfig(
        features_path=str(tmp_path / "non_existent.npz"),
        output_dir=str(tmp_path / "eval_out"),
    )
    loader = EvaluationDataLoader(config=config)
    with pytest.raises(InvalidEvaluationData, match="Test features file not found"):
        loader.load()


def test_data_loader_missing_keys(tmp_path):
    """DataLoader should raise InvalidEvaluationData if keys are missing from npz."""
    bad_npz = tmp_path / "bad_splits.npz"
    np.savez(bad_npz, dummy_key=np.zeros((5, 10)))

    config = EvaluationConfig(
        features_path=str(bad_npz),
        output_dir=str(tmp_path / "eval_out"),
    )
    loader = EvaluationDataLoader(config=config)
    with pytest.raises(InvalidEvaluationData, match="missing 'X_composite_test' or 'y_test' keys"):
        loader.load()


def test_data_loader_mismatched_sample_count(tmp_path):
    """DataLoader should raise InvalidEvaluationData when X and y lengths mismatch."""
    mismatch_npz = tmp_path / "mismatch.npz"
    np.savez(
        mismatch_npz,
        X_composite_test=np.zeros((5, 184, 173)),
        y_test=np.array([0, 1, 2]),
    )
    config = EvaluationConfig(
        features_path=str(mismatch_npz),
        output_dir=str(tmp_path / "eval_out"),
    )
    loader = EvaluationDataLoader(config=config)
    with pytest.raises(InvalidEvaluationData, match="Sample count mismatch"):
        loader.load()


# ==========================================
# 3. Metrics Calculator Tests
# ==========================================

def test_metrics_calculator_perfect_classification():
    """Verify metrics calculator output on perfect predictions."""
    calculator = EvaluationMetricsCalculator()
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4])
    y_probs = np.eye(5)

    summary = calculator.compute_summary_metrics(y_true, y_pred, y_probs)
    assert summary["accuracy"] == 1.0
    assert summary["macro_avg"]["f1_score"] == 1.0
    assert summary["weighted_avg"]["precision"] == 1.0

    per_class = calculator.compute_per_class_metrics(y_true, y_pred, y_probs)
    for c_name in ["ambulance", "car_horn", "fire_alarm", "doorbell", "dog_bark"]:
        assert per_class[c_name]["precision"] == 1.0
        assert per_class[c_name]["recall"] == 1.0
        assert per_class[c_name]["f1_score"] == 1.0
        assert per_class[c_name]["support"] == 1


def test_metrics_calculator_with_misclassifications():
    """Verify metrics calculation when predictions contain misclassifications."""
    calculator = EvaluationMetricsCalculator()
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 0])  # dog_bark (4) misclassified as ambulance (0)

    summary = calculator.compute_summary_metrics(y_true, y_pred)
    assert summary["accuracy"] == 0.8  # 4 / 5 = 80%

    per_class = calculator.compute_per_class_metrics(y_true, y_pred)
    assert per_class["dog_bark"]["recall"] == 0.0
    assert per_class["dog_bark"]["precision"] == 0.0
    assert per_class["ambulance"]["recall"] == 1.0
    assert per_class["ambulance"]["precision"] == 0.5  # 1 true positive / 2 predicted


def test_metrics_calculator_loss():
    """Verify categorical cross-entropy loss computation."""
    calculator = EvaluationMetricsCalculator()
    y_true = np.array([0, 1])
    # Confident correct predictions
    y_probs = np.array([
        [0.99, 0.01],
        [0.01, 0.99],
    ])
    loss = calculator.compute_categorical_loss(y_true, y_probs)
    assert 0.0 < loss < 0.05

    # Invalid probability shape should raise MetricCalculationError
    with pytest.raises(MetricCalculationError):
        calculator.compute_categorical_loss(y_true, np.array([0.5, 0.5]))


# ==========================================
# 4. Confusion Matrix Generator Tests
# ==========================================

def test_confusion_matrix_generator():
    """Test raw and normalized confusion matrix calculations."""
    generator = ConfusionMatrixGenerator()
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 0])

    cm_data = generator.generate(y_true, y_pred)
    raw = cm_data["raw_matrix"]
    norm = cm_data["normalized_matrix"]

    assert raw.shape == (5, 5)
    assert norm.shape == (5, 5)

    # ambulance row: true 0, pred 0 -> 1 count
    assert raw[0, 0] == 1
    # dog_bark row (idx 4): true 4, pred 0 -> 1 count at (4, 0)
    assert raw[4, 0] == 1
    assert raw[4, 4] == 0

    # Normalized row 0 (ambulance) should sum to 1.0
    assert np.isclose(np.sum(norm[0]), 1.0)
    # Normalized row 4 (dog_bark) should have 1.0 at (4, 0) and 0.0 elsewhere
    assert np.isclose(norm[4, 0], 1.0)
    assert np.isclose(norm[4, 4], 0.0)


# ==========================================
# 5. Prediction Analyzer Tests
# ==========================================

def test_prediction_analyzer(tmp_path):
    """Test prediction analyzer records, statistics, and CSV export."""
    config = EvaluationConfig(output_dir=str(tmp_path / "eval_out"))
    analyzer = PredictionAnalyzer(config=config)

    y_true = np.array([2, 0, 4, 1, 3])
    y_pred = np.array([2, 0, 0, 1, 3])
    y_probs = np.array([
        [0.01, 0.01, 0.95, 0.01, 0.02],
        [0.85, 0.05, 0.05, 0.02, 0.03],
        [0.60, 0.10, 0.10, 0.10, 0.10],  # dog_bark predicted as ambulance with 0.60 conf
        [0.02, 0.92, 0.02, 0.02, 0.02],
        [0.05, 0.05, 0.05, 0.80, 0.05],
    ])

    records = analyzer.analyze_predictions(y_true, y_pred, y_probs)
    assert len(records) == 5

    # Check 3rd record (index 2, dog_bark -> ambulance)
    rec2 = records[2]
    assert rec2.true_class == "dog_bark"
    assert rec2.predicted_class == "ambulance"
    assert rec2.is_correct is False
    assert np.isclose(rec2.confidence, 0.60)

    # Check summary statistics
    summary = analyzer.compute_confidence_summary(records)
    assert summary["total_samples"] == 5
    assert summary["correct_count"] == 4
    assert summary["misclassified_count"] == 1
    assert summary["overall_min"] == pytest.approx(0.60, abs=1e-3)
    assert summary["overall_max"] == pytest.approx(0.95, abs=1e-3)

    # Check misclassifications filtering
    misclassified = analyzer.find_misclassifications(records)
    assert len(misclassified) == 1
    assert misclassified[0]["true_class"] == "dog_bark"
    assert misclassified[0]["predicted_class"] == "ambulance"

    # Check CSV export
    csv_path = analyzer.export_predictions_csv(records)
    assert csv_path.exists()
    content = csv_path.read_text(encoding="utf-8")
    assert "sample_index,true_label_id,true_class" in content
    assert "dog_bark" in content


# ==========================================
# 6. Evaluation Visualizer Tests
# ==========================================

def test_evaluation_visualizer(tmp_path):
    """Test plot generation for confusion matrices, metric comparisons, and confidence distribution."""
    config = EvaluationConfig(output_dir=str(tmp_path / "plots"))
    visualizer = EvaluationVisualizer(config=config)

    # 1. Confusion matrices
    cm = np.eye(5)
    raw_path = visualizer.plot_confusion_matrix(cm, normalized=False)
    assert raw_path.exists()
    assert raw_path.stat().st_size > 0

    norm_path = visualizer.plot_confusion_matrix(cm, normalized=True)
    assert norm_path.exists()
    assert norm_path.stat().st_size > 0

    # 2. Metrics comparison
    per_class = {
        c: {"precision": 0.8, "recall": 0.9, "f1_score": 0.85}
        for c in ["ambulance", "car_horn", "fire_alarm", "doorbell", "dog_bark"]
    }
    metrics_path = visualizer.plot_metrics_comparison(per_class)
    assert metrics_path.exists()
    assert metrics_path.stat().st_size > 0

    # 3. Confidence distribution
    conf_path = visualizer.plot_confidence_distribution([0.9, 0.85, 0.45], [True, True, False])
    assert conf_path.exists()
    assert conf_path.stat().st_size > 0


# ==========================================
# 7. Evaluation Report Generator Tests
# ==========================================

def test_report_generator(tmp_path):
    """Test generation of classification_report.json, evaluation_metrics.json, and evaluation_report.txt."""
    config = EvaluationConfig(output_dir=str(tmp_path / "reports"))
    generator = EvaluationReportGenerator(config=config)

    # 1. Classification report JSON
    clf_dict = {
        "ambulance": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 1},
        "accuracy": 1.0,
    }
    clf_path = generator.export_classification_report_json(clf_dict)
    assert clf_path.exists()
    with open(clf_path, "r", encoding="utf-8") as f:
        loaded_clf = json.load(f)
    assert loaded_clf["accuracy"] == 1.0

    # 2. Evaluation metrics JSON
    metrics_payload = {
        "overall_metrics": {"accuracy": 0.8, "test_loss": 0.35},
        "per_class_metrics": {},
    }
    metrics_path = generator.export_metrics_summary_json(metrics_payload)
    assert metrics_path.exists()
    with open(metrics_path, "r", encoding="utf-8") as f:
        loaded_metrics = json.load(f)
    assert loaded_metrics["overall_metrics"]["accuracy"] == 0.8

    # 3. Human readable report
    txt_path = generator.generate_human_readable_report(
        overall_metrics={"accuracy": 0.8, "test_loss": 0.35},
        per_class_metrics={"ambulance": {"precision": 1.0, "recall": 1.0, "f1_score": 1.0, "accuracy": 1.0, "support": 1}},
        confusion_matrix_data={"classes": ["ambulance"], "raw_matrix": [[1]]},
        confidence_summary={"overall_mean": 0.85, "low_confidence_count": 0},
        misclassifications=[],
        model_info={"architecture": "CNNSoundClassifier", "total_parameters": 111237},
        dataset_info={"test_samples_count": 5, "feature_shape": [5, 184, 173, 1]},
    )
    assert txt_path.exists()
    content = txt_path.read_text(encoding="utf-8")
    assert "SMART HAPTIC ALERT SYSTEM - AI MODEL EVALUATION REPORT" in content
    assert "Overall Test Accuracy : 80.00%" in content


# ==========================================
# 8. Master ModelEvaluator Pipeline Integration Test
# ==========================================

def test_master_model_evaluator_end_to_end(tmp_path):
    """Verify ModelEvaluator executes successfully on existing Phase 4/5 outputs."""
    # Ensure real Phase 4 & Phase 5 outputs exist before running
    splits_file = Path("app/ai/features/dataset_splits.npz")
    model_file = Path("app/ai/models/sound_classifier_best.keras")

    if not splits_file.exists() or not model_file.exists():
        pytest.skip("Test features or trained model not found in workspace.")

    eval_out = tmp_path / "eval_pipeline_out"
    config = EvaluationConfig(output_dir=str(eval_out))
    evaluator = ModelEvaluator(config=config)

    results = evaluator.evaluate()

    # Validate output dictionary schema
    assert "test_accuracy" in results
    assert "test_loss" in results
    assert "per_class_metrics" in results
    assert "confusion_matrix" in results
    assert "confidence_summary" in results
    assert "artifacts" in results

    assert 0.0 <= results["test_accuracy"] <= 1.0
    assert results["test_loss"] >= 0.0

    # Validate all 8 artifacts were created on disk
    artifacts = results["artifacts"]
    for key, file_path in artifacts.items():
        p = Path(file_path)
        assert p.exists(), f"Artifact '{key}' at {file_path} was not created!"
        assert p.stat().st_size > 0, f"Artifact '{key}' at {file_path} is empty!"
