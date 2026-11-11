"""Statistical profiling for the credit-risk dataset."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import TARGET_COLUMN


def _json_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def build_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute completeness, distribution, and target-balance statistics.

    Args:
        frame: Raw or cleaned applicant table.

    Returns:
        JSON-ready profiling payload.
    """

    numeric = frame.select_dtypes(include="number")
    describe: dict[str, dict[str, float | None]] = {}
    for column in numeric.columns:
        stats = numeric[column].describe()
        describe[str(column)] = {str(stat): _json_number(stats[stat]) for stat in stats.index}
    skewness = {str(column): _json_number(numeric[column].skew()) for column in numeric.columns}
    missing = {str(column): float(frame[column].isna().mean()) for column in frame.columns}
    target_proportion: dict[str, float] | None = None
    if TARGET_COLUMN in frame.columns:
        shares = frame[TARGET_COLUMN].value_counts(normalize=True, dropna=False)
        target_proportion = {str(key): float(value) for key, value in shares.items()}
    memory_bytes = int(frame.memory_usage(deep=True).sum())
    completeness = 1.0 - float(frame.isna().to_numpy().mean()) if len(frame) else 0.0
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "column_names": [str(column) for column in frame.columns],
        "memory_bytes": memory_bytes,
        "completeness": completeness,
        "missing_rate": missing,
        "describe": describe,
        "skewness": skewness,
        "target_proportion": target_proportion,
    }


def high_collinearity_pairs(
    frame: pd.DataFrame,
    threshold: float = 0.80,
    method: str = "pearson",
) -> list[dict[str, Any]]:
    """List numeric pairs whose absolute correlation meets ``threshold``.

    Args:
        frame: Table containing numeric predictors.
        threshold: Absolute correlation cutoff.
        method: ``pearson`` or ``spearman``.

    Returns:
        Pairs sorted by descending absolute correlation.
    """

    numeric = frame.select_dtypes(include="number").drop(columns=[TARGET_COLUMN], errors="ignore")
    if numeric.shape[1] < 2:
        return []
    correlation = numeric.corr(method=method).abs()
    pairs: list[dict[str, Any]] = []
    columns = list(correlation.columns)
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            value = float(correlation.loc[left, right])
            if np.isfinite(value) and value >= threshold:
                pairs.append(
                    {
                        "feature_a": str(left),
                        "feature_b": str(right),
                        "method": method,
                        "abs_correlation": round(value, 4),
                    }
                )
    pairs.sort(key=lambda item: item["abs_correlation"], reverse=True)
    return pairs
