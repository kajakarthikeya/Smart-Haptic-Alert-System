"""
Report generation for AI Model Evaluation in the Smart Haptic Alert System.

This module formats and exports comprehensive evaluation summaries into:
1. classification_report.json (scikit-learn compliant structured dictionary)
2. evaluation_metrics.json (complete serializable evaluation metrics bundle)
3. evaluation_report.txt (human-readable executive diagnostic report)
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from config import Config, EvaluationConfig
from app.ai.evaluation.exceptions import ReportGenerationError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationReportGenerator:
    """Generates structured JSON reports and human-readable text evaluation summaries."""

    def __init__(self, config: Optional[EvaluationConfig] = None) -> None:
        self.config = config or getattr(Config, "evaluation", None)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_classification_report_json(
        self,
        report_dict: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Saves scikit-learn classification report dictionary as JSON.

        Args:
            report_dict: Nested dictionary returned by classification_report(output_dict=True).
            filename: Target file name; defaults to classification_report.json.

        Returns:
            Path to saved JSON file.
        """
        try:
            target_path = self.output_dir / (filename or self.config.classification_report_json_filename)
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(report_dict, f, indent=2)
            logger.info("Saved classification report JSON to: %s", target_path)
            return target_path
        except Exception as exc:
            raise ReportGenerationError(f"Failed to export classification report JSON: {exc}") from exc

    def export_metrics_summary_json(
        self,
        metrics_payload: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Saves the comprehensive evaluation metrics payload to JSON.

        Args:
            metrics_payload: Dictionary containing accuracy, loss, macro/weighted averages, etc.
            filename: Target file name; defaults to evaluation_metrics.json.

        Returns:
            Path to saved JSON file.
        """
        try:
            target_path = self.output_dir / (filename or self.config.evaluation_metrics_json_filename)
            payload = dict(metrics_payload)
            payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

            # Convert any non-serializable objects (like numpy types)
            def _clean(val: Any) -> Any:
                if isinstance(val, np.ndarray):
                    return val.tolist()
                if hasattr(val, "item"):
                    return val.item()
                if hasattr(val, "tolist"):
                    return val.tolist()
                if isinstance(val, dict):
                    return {k: _clean(v) for k, v in val.items()}
                if isinstance(val, (list, tuple, set)):
                    return [_clean(x) for x in val]
                return val

            cleaned_payload = _clean(payload)

            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_payload, f, indent=2)
            logger.info("Saved evaluation metrics JSON to: %s", target_path)
            return target_path
        except Exception as exc:
            raise ReportGenerationError(f"Failed to export evaluation metrics JSON: {exc}") from exc

    def generate_human_readable_report(
        self,
        overall_metrics: Dict[str, Any],
        per_class_metrics: Dict[str, Dict[str, float]],
        confusion_matrix_data: Dict[str, Any],
        confidence_summary: Dict[str, Any],
        misclassifications: List[Dict[str, Any]],
        model_info: Dict[str, Any],
        dataset_info: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Generates a comprehensive, human-readable text report summarizing model performance.

        Args:
            overall_metrics: Accuracy, loss, macro/weighted averages.
            per_class_metrics: Precision, recall, F1, support, accuracy per class.
            confusion_matrix_data: Raw and normalized confusion matrices.
            confidence_summary: Mean/std/min/max confidence across samples and subsets.
            misclassifications: List of misclassified sample dictionaries.
            model_info: Model architecture, parameter count, weights path.
            dataset_info: Split dimensions, sample count, class distribution.
            filename: Target file name; defaults to evaluation_report.txt.

        Returns:
            Path to saved text file.
        """
        try:
            target_path = self.output_dir / (filename or self.config.evaluation_report_txt_filename)

            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            lines: List[str] = []

            def w(line: str = ""):
                lines.append(line)

            w("=" * 80)
            w("          SMART HAPTIC ALERT SYSTEM - AI MODEL EVALUATION REPORT")
            w("=" * 80)
            w(f"Timestamp          : {timestamp}")
            w(f"Evaluation Phase   : Phase 6 (AI Model Evaluation)")
            w(f"Target Domain      : Assistive Wearable Sound Classifier")
            w(f"Model Path         : {model_info.get('model_path', 'N/A')}")
            w(f"Model Architecture : {model_info.get('architecture', 'CNNSoundClassifier (2D CNN)')}")
            w(f"Total Parameters   : {model_info.get('total_parameters', 'N/A'):,}" if isinstance(model_info.get('total_parameters'), int) else f"Total Parameters   : {model_info.get('total_parameters', 'N/A')}")
            w(f"Feature Dimension  : {dataset_info.get('feature_shape', 'N/A')} (Composite Mel-Spectrogram + Delta + Delta-Delta)")
            w(f"Test Split Size    : {dataset_info.get('test_samples_count', 'N/A')} unseen test samples")
            w("=" * 80)
            w()

            # 1. Executive Summary
            w("1. EXECUTIVE SUMMARY")
            w("-" * 80)
            test_acc = overall_metrics.get("accuracy", 0.0)
            test_loss = overall_metrics.get("test_loss", 0.0)
            macro_f1 = overall_metrics.get("macro_avg", {}).get("f1_score", 0.0)
            weighted_f1 = overall_metrics.get("weighted_avg", {}).get("f1_score", 0.0)

            w(f"  • Overall Test Accuracy : {test_acc * 100:.2f}% ({dataset_info.get('test_samples_count', 0) - len(misclassifications)} / {dataset_info.get('test_samples_count', 0)} correct)")
            w(f"  • Categorical Test Loss : {test_loss:.4f}")
            w(f"  • Macro Avg F1-Score    : {macro_f1:.4f}")
            w(f"  • Weighted Avg F1-Score : {weighted_f1:.4f}")
            w()

            # 2. Per-Class Performance
            w("2. PER-CLASS PERFORMANCE BREAKDOWN")
            w("-" * 80)
            w(f"{'Class':<14} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Class Acc':<10} | {'Support':<8}")
            w("-" * 80)
            for c_name, c_data in per_class_metrics.items():
                p = c_data.get("precision", 0.0)
                r = c_data.get("recall", 0.0)
                f1 = c_data.get("f1_score", 0.0)
                acc = c_data.get("accuracy", 0.0)
                sup = c_data.get("support", 0)
                w(f"{c_name:<14} | {p:>9.4f}  | {r:>9.4f}  | {f1:>9.4f}  | {acc:>9.4f}  | {sup:>8}")

            w("-" * 80)
            w()

            # 3. Confusion Matrix Analysis
            w("3. CONFUSION MATRIX ANALYSIS")
            w("-" * 80)
            classes = confusion_matrix_data.get("classes", list(per_class_metrics.keys()))
            raw_matrix = confusion_matrix_data.get("raw_matrix", [])

            # Header
            header_str = f"{'True \\ Pred':<14} | " + " | ".join(f"{c[:8]:>8}" for c in classes)
            w(header_str)
            w("-" * len(header_str))

            for i, c_name in enumerate(classes):
                row_vals = raw_matrix[i] if i < len(raw_matrix) else []
                row_str = f"{c_name:<14} | " + " | ".join(f"{int(v):>8}" for v in row_vals)
                w(row_str)

            w("-" * len(header_str))
            w()

            # 4. Confidence Score Statistics
            w("4. PREDICTION CONFIDENCE ANALYSIS")
            w("-" * 80)
            w(f"  • Mean Confidence (All Samples)     : {confidence_summary.get('overall_mean', 0.0):.4f}")
            w(f"  • Std Dev Confidence (All Samples)  : {confidence_summary.get('overall_std', 0.0):.4f}")
            w(f"  • Min Confidence (All Samples)      : {confidence_summary.get('overall_min', 0.0):.4f}")
            w(f"  • Max Confidence (All Samples)      : {confidence_summary.get('overall_max', 0.0):.4f}")
            w(f"  • Mean Confidence (Correct Samples) : {confidence_summary.get('correct_mean', 0.0):.4f}")
            w(f"  • Mean Confidence (Misclassified)   : {confidence_summary.get('misclassified_mean', 0.0):.4f}")
            w(f"  • Low Confidence Samples (< 0.50)   : {confidence_summary.get('low_confidence_count', 0)}")
            w()

            # 5. Misclassifications and Failure Analysis
            w("5. ERROR ANALYSIS & MISCLASSIFICATIONS")
            w("-" * 80)
            if not misclassifications:
                w("  No misclassifications observed on the test set! Perfect 100% accuracy.")
            else:
                w(f"  Total Errors: {len(misclassifications)} sample(s)")
                for idx, err in enumerate(misclassifications, 1):
                    w(f"  [{idx}] Sample Index: {err.get('sample_index')}")
                    w(f"      Ground Truth : '{err.get('true_class')}' (ID: {err.get('true_label_id')})")
                    w(f"      Predicted    : '{err.get('predicted_class')}' (ID: {err.get('predicted_label_id')})")
                    w(f"      Confidence   : {err.get('confidence', 0.0):.4f}")
                    w(f"      True Class Probability: {err.get('true_class_probability', 0.0):.4f}")
                    top_probs = err.get("top_probabilities", {})
                    top_prob_str = ", ".join([f"{k}: {v:.4f}" for k, v in top_probs.items()])
                    w(f"      Class Probability Distribution: {top_prob_str}")
                    w()

            # 6. Safety & Wearable Domain Assessment
            w("6. SYSTEM CRITICALITY & DEPLOYMENT READINESS")
            w("-" * 80)
            safety_critical_classes = ["ambulance", "fire_alarm", "car_horn"]
            w("  Safety-Critical Emergency Sound Verification:")
            for sc in safety_critical_classes:
                if sc in per_class_metrics:
                    sc_rec = per_class_metrics[sc].get("recall", 0.0)
                    sc_prec = per_class_metrics[sc].get("precision", 0.0)
                    status = "OPTIMAL" if sc_rec >= 1.0 else ("ACCEPTABLE" if sc_rec >= 0.8 else "CRITICAL RISK")
                    w(f"  • {sc:<12} -> Recall: {sc_rec*100:.1f}%, Precision: {sc_prec*100:.1f}% [{status}]")
                else:
                    w(f"  • {sc:<12} -> Not present in test metrics")

            w()
            w("  Operational Considerations:")
            w("  - High-priority alarms (ambulance, fire_alarm, car_horn) must never be missed.")
            w("  - Dog bark misclassification as ambulance introduces a false alarm risk.")
            w("  - Deployment in Phase 7 real-time detection should incorporate temporal smoothing,")
            w("    a confirmation window, and an adjustable confidence threshold (e.g. >= 0.70).")
            w("=" * 80)
            w("                         END OF EVALUATION REPORT")
            w("=" * 80)

            report_text = "\n".join(lines)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(report_text)

            logger.info("Saved human-readable evaluation report to: %s", target_path)
            return target_path

        except Exception as exc:
            raise ReportGenerationError(f"Failed to generate human-readable report: {exc}") from exc
