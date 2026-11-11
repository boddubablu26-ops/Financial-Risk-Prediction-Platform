"""Classification metrics at a chosen decision threshold."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


def positive_probabilities(estimator: Any, features: Any) -> np.ndarray:
    """Return P(default) for a classifier or pipeline.

    Args:
        estimator: Fitted estimator exposing ``predict_proba`` and ``classes_``.
        features: Model inputs.

    Returns:
        Probability of the positive class as a 1-d array.
    """

    probabilities = np.asarray(estimator.predict_proba(features))
    classes = [int(label) for label in list(getattr(estimator, "classes_", [0, 1]))]
    if 1 in classes:
        column = classes.index(1)
    else:
        column = probabilities.shape[1] - 1
    return probabilities[:, column].astype(float)


def compute_metrics(y_true: Any, y_prob: Any, threshold: float) -> dict[str, Any]:
    """Compute thresholded classification metrics plus ranking metrics.

    Args:
        y_true: Binary ground truth.
        y_prob: Predicted default probabilities.
        threshold: Cutoff at or above which a record is High Risk.

    Returns:
        Metric dictionary including the confusion-matrix counts.
    """

    truth = np.asarray(y_true).astype(int)
    scores = np.asarray(y_prob).astype(float)
    predicted = (scores >= threshold).astype(int)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (int(value) for value in matrix.ravel())
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1_score": float(f1_score(truth, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, scores)) if len(np.unique(truth)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(truth, scores)) if len(np.unique(truth)) > 1 else 0.0,
        "confusion_matrix": {
            "tn": true_negative,
            "fp": false_positive,
            "fn": false_negative,
            "tp": true_positive,
        },
    }


def curve_payloads(y_true: Any, y_prob: Any, max_points: int = 80) -> dict[str, list[float]]:
    """Downsample ROC and precision-recall curves for JSON storage.

    Args:
        y_true: Binary ground truth.
        y_prob: Predicted default probabilities.
        max_points: Maximum points retained per curve.

    Returns:
        Lists of false-positive rate, true-positive rate, precision, and recall.
    """

    truth = np.asarray(y_true).astype(int)
    scores = np.asarray(y_prob).astype(float)
    fpr, tpr, _ = roc_curve(truth, scores)
    precision, recall, _ = precision_recall_curve(truth, scores)
    fpr_s, tpr_s = _downsample(fpr, tpr, max_points)
    precision_s, recall_s = _downsample(precision, recall, max_points)
    return {
        "fpr": fpr_s,
        "tpr": tpr_s,
        "precision_curve": precision_s,
        "recall_curve": recall_s,
    }


def _downsample(x_values: np.ndarray, y_values: np.ndarray, max_points: int) -> tuple[list[float], list[float]]:
    if len(x_values) <= max_points:
        index = np.arange(len(x_values))
    else:
        index = np.linspace(0, len(x_values) - 1, max_points).astype(int)
    return (
        [round(float(x_values[i]), 6) for i in index],
        [round(float(y_values[i]), 6) for i in index],
    )
