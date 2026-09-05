# AI Model Evaluation Module (`app/ai/evaluation`)

## Overview

The `app/ai/evaluation` module provides a comprehensive, publication-grade evaluation framework for the Smart Haptic Alert System. It evaluates trained environmental sound classification models (e.g. `CNNSoundClassifier`) against unseen test datasets (`dataset_splits.npz`) to measure accuracy, discriminative power, calibration, and safety readiness for assistive wearable devices.

---

## Architecture & Responsibilities

The evaluation pipeline is decomposed into dedicated, single-responsibility components:

| Component | File | Responsibility |
| :--- | :--- | :--- |
| **ModelEvaluator** | `evaluator.py` | Master orchestrator; runs full evaluation workflow and coordinates sub-components. |
| **EvaluationDataLoader** | `data_loader.py` | Loads, reshapes, and validates unseen test features, ground-truth labels, and model weights. |
| **EvaluationMetricsCalculator** | `metrics.py` | Computes test loss, overall accuracy, per-class metrics, and macro/weighted summaries. |
| **ConfusionMatrixGenerator** | `confusion_matrix.py` | Computes 5x5 raw count matrix and normalized (recall/class accuracy) matrix. |
| **PredictionAnalyzer** | `prediction_analyzer.py` | Analyzes confidence scores, isolates misclassifications, flags low-confidence predictions, and exports CSV. |
| **EvaluationVisualizer** | `visualization.py` | Renders publication-grade diagnostic plots (confusion matrices, bar charts, confidence histograms). |
| **EvaluationReportGenerator** | `report_generator.py` | Formats and outputs structured JSON reports and human-readable text evaluation summaries. |
| **Exceptions** | `exceptions.py` | Domain-specific exception hierarchy inheriting from `EvaluationError`. |

---

## Target Sound Classes

The evaluator explicitly tracks and verifies the five system sound classes:

0. **Ambulance** (Safety Priority: Emergency Alarm)
1. **Car Horn** (Safety Priority: Traffic Hazard)
2. **Fire Alarm** (Safety Priority: Life Safety Critical)
3. **Doorbell** (Safety Priority: Environmental Alert)
4. **Dog Bark** (Safety Priority: Environmental Warning)

---

## Generated Artifacts

All evaluation artifacts are exported to `app/outputs/model_evaluation/`:

- `predictions.csv`: Granular per-sample prediction records including true/pred labels, confidence, correct status, and softmax probabilities.
- `confusion_matrix.png`: Heatmap showing raw sample counts across true vs. predicted classes.
- `normalized_confusion_matrix.png`: Normalized confusion matrix showing per-class recall / accuracy rates.
- `metrics_comparison.png`: Grouped bar chart comparing Precision, Recall, and F1-score across all 5 sound classes.
- `confidence_distribution.png`: Histogram depicting model confidence score distribution across correct and misclassified predictions.
- `classification_report.json`: Scikit-learn compliant classification report dictionary.
- `evaluation_metrics.json`: Complete serializable evaluation payload containing all metrics, dataset metadata, and confidence summaries.
- `evaluation_report.txt`: Human-readable executive evaluation report detailing performance, error analysis, and deployment recommendations.

---

## Usage Example

```python
from app.ai.evaluation.evaluator import ModelEvaluator

# Initialize evaluator with default configuration (or pass custom EvaluationConfig)
evaluator = ModelEvaluator()

# Execute complete evaluation pipeline
results = evaluator.evaluate()

print(f"Test Accuracy: {results['test_accuracy'] * 100:.2f}%")
print(f"Test Loss: {results['test_loss']:.4f}")
print(f"Artifacts generated at: {results['artifacts']}")
```
