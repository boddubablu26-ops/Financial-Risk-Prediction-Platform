"""Compatibility wrapper for the explainability module path used in the tracker."""

from src.explainability.shap_explainer import (
    aggregate_contributions,
    build_model_explainer,
    build_tree_explainer,
    explain_matrix,
    mean_absolute_importance,
    original_feature_name,
)

__all__ = [
    "aggregate_contributions",
    "build_model_explainer",
    "build_tree_explainer",
    "explain_matrix",
    "mean_absolute_importance",
    "original_feature_name",
]
