"""SHAP value extraction, aggregation, and global summary construction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import NOMINAL_FEATURES


def original_feature_name(transformed_name: str) -> str:
    """Map a ColumnTransformer output name back to a source feature.

    Args:
        transformed_name: Name such as ``nominal__loan_intent_MEDICAL``.

    Returns:
        Source feature name. One-hot columns collapse to their parent.
    """

    name = transformed_name.split("__", 1)[-1]
    for categorical in NOMINAL_FEATURES:
        prefix = f"{categorical}_"
        if name.startswith(prefix):
            return categorical
    return name


def extract_shap_matrix(shap_output: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Normalize SHAP outputs from TreeExplainer and the Explanation API.

    Args:
        shap_output: Array, list of class arrays, or a SHAP Explanation.

    Returns:
        Matrix shaped ``(n_rows, n_features)`` for the positive class, and the
        base value when the explainer provided one.
    """

    base: np.ndarray | None = None
    if hasattr(shap_output, "values"):
        values = np.asarray(shap_output.values)
        if getattr(shap_output, "base_values", None) is not None:
            base = np.asarray(shap_output.base_values)
    else:
        values = shap_output
        if isinstance(values, list):
            values = np.asarray(values[1] if len(values) > 1 else values[0])
        else:
            values = np.asarray(values)
    if values.ndim == 3:
        class_index = 1 if values.shape[-1] > 1 else 0
        values = values[:, :, class_index]
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values.astype(float), base


def aggregate_contributions(
    feature_names: list[str],
    shap_row: np.ndarray,
) -> list[dict[str, float | str]]:
    """Sum SHAP values that belong to the same original feature.

    Args:
        feature_names: Transformed feature names.
        shap_row: One applicant's SHAP vector.

    Returns:
        Contributions sorted by descending signed impact.
    """

    totals: dict[str, float] = {}
    for name, value in zip(feature_names, shap_row):
        key = original_feature_name(str(name))
        totals[key] = totals.get(key, 0.0) + float(value)
    contributions = [{"feature": key, "shap_value": value} for key, value in totals.items()]
    contributions.sort(key=lambda item: abs(float(item["shap_value"])), reverse=True)
    return contributions


def mean_absolute_importance(
    feature_names: list[str],
    shap_matrix: np.ndarray,
) -> list[dict[str, float | str]]:
    """Global mean absolute SHAP, aggregated to original features.

    Args:
        feature_names: Transformed feature names.
        shap_matrix: SHAP matrix for a cohort.

    Returns:
        Ranking from highest mean absolute contribution to lowest.
    """

    grouped: dict[str, list[float]] = {}
    for row in shap_matrix:
        for item in aggregate_contributions(feature_names, row):
            grouped.setdefault(str(item["feature"]), []).append(abs(float(item["shap_value"])))
    ranking = [
        {"feature": feature, "mean_abs_shap": float(np.mean(values))}
        for feature, values in grouped.items()
    ]
    ranking.sort(key=lambda item: float(item["mean_abs_shap"]), reverse=True)
    return ranking


def build_model_explainer(model: Any, background: np.ndarray | None = None) -> Any | None:
    """Create a SHAP explainer for a tree model or a logistic regression.

    Args:
        model: Fitted classifier, not the preprocessing pipeline.
        background: Transformed rows used by linear explainers.

    Returns:
        A SHAP explainer, or None when no suitable explainer can be built.
    """

    tree_explainer = build_tree_explainer(model)
    if tree_explainer is not None:
        return tree_explainer
    if background is None or type(model).__name__ != "LogisticRegression":
        return None
    import shap

    sample = np.asarray(background, dtype=float)
    if len(sample) > 200:
        sample = sample[:200]
    try:
        return shap.LinearExplainer(model, sample)
    except Exception:
        try:
            return shap.Explainer(model.predict_proba, sample)
        except Exception:
            return None


def build_tree_explainer(model: Any) -> Any | None:
    """Create a TreeExplainer when the estimator is tree-based.

    Args:
        model: Fitted classifier, not the full preprocessing pipeline.

    Returns:
        A SHAP explainer, or None when the model is not a tree ensemble.
    """

    class_name = type(model).__name__
    if class_name not in {"XGBClassifier", "RandomForestClassifier", "DecisionTreeClassifier"} and not hasattr(
        model, "get_booster"
    ):
        return None
    import shap

    try:
        return shap.TreeExplainer(model, model_output="probability")
    except Exception:
        return shap.TreeExplainer(model)


def transformed_feature_names(pipeline: Any) -> list[str]:
    """Read output names from the fitted column transformer.

    Args:
        pipeline: Fitted model pipeline containing a ``columns`` step.

    Returns:
        Transformed feature names.
    """

    columns = pipeline.named_steps["columns"]
    return [str(name) for name in columns.get_feature_names_out()]


def explain_matrix(explainer: Any, transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute SHAP values for an already transformed matrix.

    Args:
        explainer: Fitted SHAP explainer.
        transformed: Numeric model matrix.

    Returns:
        SHAP matrix and base values.
    """

    matrix = np.asarray(transformed, dtype=float)
    try:
        output = explainer(matrix)
    except Exception:
        output = explainer.shap_values(matrix)
    return extract_shap_matrix(output)


def sample_frame(frame: pd.DataFrame, limit: int, random_state: int) -> pd.DataFrame:
    """Draw a deterministic sample for global SHAP.

    Args:
        frame: Source table.
        limit: Maximum rows.
        random_state: Seed.

    Returns:
        Sampled frame.
    """

    if len(frame) <= limit:
        return frame.reset_index(drop=True)
    return frame.sample(n=limit, random_state=random_state).reset_index(drop=True)
