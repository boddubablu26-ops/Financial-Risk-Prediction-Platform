"""Mutual-information ranking and collinearity audit for engineered features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src.config import RANDOM_STATE, ROBUST_FEATURES, STANDARD_FEATURES


def rank_features(
    engineered: pd.DataFrame,
    target: pd.Series,
    k: int = 15,
    collinearity_threshold: float = 0.85,
) -> dict[str, Any]:
    """Rank numeric engineered features and list highly collinear pairs.

    Args:
        engineered: Training frame after imputation and ratio construction.
        target: Binary default labels aligned to ``engineered``.
        k: Number of features to flag as the selected subset.
        collinearity_threshold: Absolute Pearson correlation that flags a pair.

    Returns:
        JSON-ready ranking, the top-k names, and collinear pairs.
    """

    columns = [column for column in ROBUST_FEATURES + STANDARD_FEATURES if column in engineered.columns]
    matrix = engineered[columns].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0.0)
    labels = np.asarray(target).astype(int)
    scores = mutual_info_classif(matrix.to_numpy(), labels, random_state=RANDOM_STATE)
    ranking = sorted(
        (
            {"feature": column, "mutual_information": round(float(score), 6)}
            for column, score in zip(columns, scores)
        ),
        key=lambda item: item["mutual_information"],
        reverse=True,
    )
    correlation = matrix.corr().abs()
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            value = float(correlation.loc[left, right])
            if np.isfinite(value) and value >= collinearity_threshold:
                pairs.append({"feature_a": left, "feature_b": right, "abs_correlation": round(value, 4)})
    pairs.sort(key=lambda item: item["abs_correlation"], reverse=True)
    return {
        "top_k": k,
        "selected_features": [item["feature"] for item in ranking[:k]],
        "ranking": ranking,
        "collinear_pairs": pairs,
        "note": (
            "All engineered features are retained in the model matrix so tree and "
            "linear models see the same inputs. Pairs with |r| >= 0.85 are reported "
            "for audit. loan_percent_income is excluded because it duplicates "
            "loan_to_income_ratio."
        ),
    }
