"""Evaluation Metrics and Classification Utility Functions."""

from typing import Dict, List, Sequence, Union


def calculate_accuracy(y_true: Sequence[Union[int, str]], y_pred: Sequence[Union[int, str]]) -> float:
    """Calculates top-1 classification accuracy score.

    Args:
        y_true: Ground truth target labels.
        y_pred: Model predicted target labels.

    Returns:
        Accuracy ratio between 0.0 and 1.0.
    """
    if not y_true or len(y_true) != len(y_pred):
        return 0.0

    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)


def calculate_confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], class_labels: List[str]
) -> Dict[str, Dict[str, int]]:
    """Generates a structured dictionary representation of confusion matrix.

    Args:
        y_true: True class strings.
        y_pred: Predicted class strings.
        class_labels: List of unique label strings.

    Returns:
        Nested dictionary mapping [true_class][pred_class] -> count.
    """
    matrix: Dict[str, Dict[str, int]] = {
        label: {pred_label: 0 for pred_label in class_labels} for label in class_labels
    }

    for true, pred in zip(y_true, y_pred):
        if true in matrix and pred in matrix[true]:
            matrix[true][pred] += 1

    return matrix
