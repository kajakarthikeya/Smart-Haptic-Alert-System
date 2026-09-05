"""
Master Evaluator Orchestrator for the Smart Haptic Alert System.

This module coordinates:
1. Loading evaluation data and trained model.
2. Inference on unseen test split.
3. Metric calculations (Accuracy, Loss, Precision, Recall, F1, Per-class metrics).
4. Confusion matrix generation (raw and normalized).
5. Prediction and confidence error analysis.
6. Diagnostic plot visualization.
7. Report generation (JSON + TXT + CSV).
"""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import numpy as np

from config import Config, EvaluationConfig
from app.ai.evaluation.confusion_matrix import ConfusionMatrixGenerator
from app.ai.evaluation.data_loader import EvaluationDataLoader
from app.ai.evaluation.exceptions import EvaluationError, PredictionError
from app.ai.evaluation.metrics import EvaluationMetricsCalculator
from app.ai.evaluation.prediction_analyzer import PredictionAnalyzer
from app.ai.evaluation.report_generator import EvaluationReportGenerator
from app.ai.evaluation.visualization import EvaluationVisualizer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Orchestrator for comprehensive offline evaluation of trained sound classification models."""

    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        data_loader: Optional[EvaluationDataLoader] = None,
        metrics_calculator: Optional[EvaluationMetricsCalculator] = None,
        confusion_generator: Optional[ConfusionMatrixGenerator] = None,
        prediction_analyzer: Optional[PredictionAnalyzer] = None,
        visualizer: Optional[EvaluationVisualizer] = None,
        report_generator: Optional[EvaluationReportGenerator] = None,
    ) -> None:
        self.config = config or getattr(Config, "evaluation", getattr(Config, "EVALUATION", None))
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data_loader = data_loader or EvaluationDataLoader(config=self.config)
        self.metrics_calculator = metrics_calculator or EvaluationMetricsCalculator(config=self.config)
        self.confusion_generator = confusion_generator or ConfusionMatrixGenerator(config=self.config)
        self.prediction_analyzer = prediction_analyzer or PredictionAnalyzer(config=self.config)
        self.visualizer = visualizer or EvaluationVisualizer(config=self.config)
        self.report_generator = report_generator or EvaluationReportGenerator(config=self.config)

    def evaluate(
        self,
        features_path: Optional[str] = None,
        model_path: Optional[str] = None,
        class_names_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes the full evaluation pipeline and outputs all artifacts.

        Returns:
            Dictionary containing evaluation metrics, summaries, and generated artifact paths.
        """
        logger.info("Starting Phase 6: AI Model Evaluation pipeline...")

        # 1. Load Evaluation Data & Model
        eval_data = self.data_loader.load(
            features_path=features_path,
            model_path=model_path,
            class_names_path=class_names_path,
        )
        class_names = eval_data.class_names
        X_test = eval_data.X_test
        y_test = eval_data.y_test
        model = eval_data.model

        logger.info(
            "Evaluation data successfully loaded: %d test samples, input shape %s, %d target classes",
            eval_data.num_samples,
            X_test.shape,
            len(class_names),
        )

        # 2. Run Inference
        try:
            logger.info("Running model inference on unseen test set...")
            keras_model = getattr(model, "model", model)
            if keras_model is not None and hasattr(keras_model, "predict"):
                y_probs = keras_model.predict(X_test, verbose=0)
            elif hasattr(model, "predict"):
                preds = [model.predict(sample)[2] for sample in X_test]
                y_probs = np.array(preds, dtype=np.float32)
            else:
                raise PredictionError("Model object does not support predict()")

            y_probs = np.asarray(y_probs, dtype=np.float32)
            y_pred = np.argmax(y_probs, axis=-1).astype(int)
        except Exception as exc:
            raise PredictionError(f"Inference failed on test dataset: {exc}") from exc

        # 3. Compute Metrics
        logger.info("Computing evaluation metrics...")
        overall_metrics = self.metrics_calculator.compute_summary_metrics(y_test, y_pred, y_probs)
        per_class_metrics = self.metrics_calculator.compute_per_class_metrics(y_test, y_pred, y_probs)
        sklearn_report = self.metrics_calculator.generate_classification_report(y_test, y_pred)

        test_loss = self.metrics_calculator.compute_categorical_loss(y_test, y_probs)
        overall_metrics["test_loss"] = test_loss

        # 4. Confusion Matrix Generation
        logger.info("Generating raw and normalized confusion matrices...")
        cm_data = self.confusion_generator.generate(y_test, y_pred)
        raw_cm = cm_data["raw_matrix"]
        norm_cm = cm_data["normalized_matrix"]

        # 5. Prediction & Confidence Analysis
        logger.info("Analyzing predictions, confidence scores, and error patterns...")
        records = self.prediction_analyzer.analyze_predictions(y_test, y_pred, y_probs)
        conf_summary = self.prediction_analyzer.compute_confidence_summary(records)
        misclassifications = self.prediction_analyzer.find_misclassifications(records)

        # Export predictions.csv
        csv_path = self.prediction_analyzer.export_predictions_csv(records)

        # 6. Visualizations
        logger.info("Generating diagnostic plots...")
        raw_cm_img = self.visualizer.plot_confusion_matrix(raw_cm, normalized=False)
        norm_cm_img = self.visualizer.plot_confusion_matrix(norm_cm, normalized=True)
        metrics_img = self.visualizer.plot_metrics_comparison(per_class_metrics)

        confidences = [r.confidence for r in records]
        is_corrects = [r.is_correct for r in records]
        conf_dist_img = self.visualizer.plot_confidence_distribution(confidences, is_corrects)

        # 7. Model & Dataset Info Payloads
        keras_model = getattr(model, "model", model)
        model_info = {
            "model_path": str(model_path or getattr(self.config, "model_path", "")),
            "architecture": type(model).__name__,
            "total_parameters": getattr(keras_model, "count_params", lambda: None)(),
            "input_shape": list(keras_model.input_shape) if hasattr(keras_model, "input_shape") else None,
        }
        dataset_info = {
            "test_samples_count": len(y_test),
            "feature_shape": list(X_test.shape),
            "classes": class_names,
            "ground_truth_counts": {
                class_names[idx]: int(np.sum(y_test == idx)) for idx in range(len(class_names))
            },
        }

        # 8. Report Generation
        logger.info("Exporting JSON and human-readable text evaluation reports...")
        clf_json_path = self.report_generator.export_classification_report_json(sklearn_report)

        metrics_payload = {
            "overall_metrics": overall_metrics,
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm_data,
            "confidence_summary": conf_summary,
            "misclassifications_count": len(misclassifications),
            "model_info": model_info,
            "dataset_info": dataset_info,
        }
        metrics_json_path = self.report_generator.export_metrics_summary_json(metrics_payload)

        txt_report_path = self.report_generator.generate_human_readable_report(
            overall_metrics=overall_metrics,
            per_class_metrics=per_class_metrics,
            confusion_matrix_data=cm_data,
            confidence_summary=conf_summary,
            misclassifications=misclassifications,
            model_info=model_info,
            dataset_info=dataset_info,
        )

        artifacts = {
            "predictions_csv": str(csv_path),
            "confusion_matrix_png": str(raw_cm_img),
            "normalized_confusion_matrix_png": str(norm_cm_img),
            "metrics_comparison_png": str(metrics_img),
            "confidence_distribution_png": str(conf_dist_img),
            "classification_report_json": str(clf_json_path),
            "evaluation_metrics_json": str(metrics_json_path),
            "evaluation_report_txt": str(txt_report_path),
        }

        results = {
            "test_accuracy": overall_metrics["accuracy"],
            "test_loss": test_loss,
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm_data,
            "confidence_summary": conf_summary,
            "misclassifications": misclassifications,
            "artifacts": artifacts,
        }

        logger.info(
            "Phase 6 AI Model Evaluation completed successfully! Overall Accuracy: %.2f%%, Categorical Loss: %.4f",
            results["test_accuracy"] * 100,
            results["test_loss"],
        )
        return results
