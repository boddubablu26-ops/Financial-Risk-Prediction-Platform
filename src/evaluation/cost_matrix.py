"""Cost-sensitive threshold search for credit decisions.

False negatives (approving a borrower who defaults) cost five times a false
positive (declining a borrower who would have repaid), matching the modeling spec.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import FN_COST, FP_COST, THRESHOLD_MAX, THRESHOLD_MIN, THRESHOLD_STEP


def threshold_grid(
    minimum: float = THRESHOLD_MIN,
    maximum: float = THRESHOLD_MAX,
    step: float = THRESHOLD_STEP,
) -> np.ndarray:
    """Return the specification threshold grid from 0.10 to 0.60.

    Args:
        minimum: Inclusive start.
        maximum: Inclusive end.
        step: Grid spacing.

    Returns:
        One-dimensional array of thresholds.
    """

    count = int(round((maximum - minimum) / step)) + 1
    grid = minimum + step * np.arange(count)
    return np.round(grid, 4)


def expected_cost(
    y_true: Any,
    y_prob: Any,
    threshold: float,
    fp_cost: float = FP_COST,
    fn_cost: float = FN_COST,
) -> dict[str, float]:
    """Compute the portfolio cost of a single threshold.

    Args:
        y_true: Binary ground truth.
        y_prob: Predicted default probabilities.
        threshold: Decision cutoff.
        fp_cost: Cost of a false positive.
        fn_cost: Cost of a false negative.

    Returns:
        Confusion counts and total cost.
    """

    truth = np.asarray(y_true).astype(int)
    scores = np.asarray(y_prob).astype(float)
    predicted = scores >= threshold
    false_positive = int(np.sum((predicted == 1) & (truth == 0)))
    false_negative = int(np.sum((predicted == 0) & (truth == 1)))
    true_positive = int(np.sum((predicted == 1) & (truth == 1)))
    true_negative = int(np.sum((predicted == 0) & (truth == 0)))
    total = fp_cost * false_positive + fn_cost * false_negative
    return {
        "threshold": float(threshold),
        "fp": false_positive,
        "fn": false_negative,
        "tp": true_positive,
        "tn": true_negative,
        "expected_cost": float(total),
    }


def optimize_threshold(
    y_true: Any,
    y_prob: Any,
    fp_cost: float = FP_COST,
    fn_cost: float = FN_COST,
) -> dict[str, Any]:
    """Select the threshold on the specification grid that minimizes expected cost.

    Args:
        y_true: Binary ground truth, typically the holdout set.
        y_prob: Predicted default probabilities.
        fp_cost: Cost of a false positive.
        fn_cost: Cost of a false negative.

    Returns:
        The winning threshold, its cost, and the full cost curve.
    """

    curve = [
        expected_cost(y_true, y_prob, float(threshold), fp_cost=fp_cost, fn_cost=fn_cost)
        for threshold in threshold_grid()
    ]
    best = min(curve, key=lambda item: (item["expected_cost"], item["threshold"]))
    return {
        "optimal_threshold": best["threshold"],
        "minimum_cost": best["expected_cost"],
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
        "curve": curve,
    }
