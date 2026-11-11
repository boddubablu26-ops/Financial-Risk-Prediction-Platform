"""Compact hyperparameter search spaces from the modeling specification."""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from src.config import RANDOM_STATE
from src.features.feature_pipeline import build_model_pipeline
from src.models.model_registry import build_estimators


PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "logistic_regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs"],
        "model__class_weight": ["balanced"],
    },
    "decision_tree": {
        "model__max_depth": [4, 6, 8, 10],
        "model__min_samples_split": [10, 25, 50],
        "model__min_samples_leaf": [5, 10, 20],
        "model__criterion": ["gini", "entropy"],
    },
    "random_forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [8, 12, 16],
        "model__min_samples_split": [5, 10],
        "model__class_weight": ["balanced", "balanced_subsample"],
    },
    "xgboost": {
        "model__n_estimators": [150, 250],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [4, 6],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    },
    "svm": {
        "model__C": [0.1, 1.0, 5.0],
        "model__gamma": ["scale", "auto"],
    },
}


def tune_logistic_regression(features: Any, target: Any) -> dict[str, Any]:
    """Grid-search logistic regression ``C`` inside the full preprocessing pipeline.

    Args:
        features: Training features.
        target: Training labels.

    Returns:
        Best ``C``, the cross-validated ROC-AUC, and the parameter grid.
    """

    estimators = {key: estimator for key, _name, estimator in build_estimators(scale_pos_weight=1.0)}
    pipeline = build_model_pipeline(estimators["logistic_regression"])
    search = GridSearchCV(
        pipeline,
        param_grid={"model__C": [0.1, 1.0, 10.0]},
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
        refit=False,
    )
    search.fit(features, target)
    return {
        "best_C": float(search.best_params_["model__C"]),
        "best_roc_auc": float(search.best_score_),
        "param_grid": {"model__C": [0.1, 1.0, 10.0]},
    }


def randomized_search(
    features: Any,
    target: Any,
    model_key: str,
    scale_pos_weight: float,
    n_iter: int = 4,
    cv_folds: int = 3,
) -> dict[str, Any]:
    """Run a short randomized search for one of the five model families.

    Args:
        features: Training features.
        target: Training labels.
        model_key: Registry key such as ``xgboost``.
        scale_pos_weight: Passed through when the XGBoost model is built.
        n_iter: Number of random parameter draws.
        cv_folds: Stratified folds used inside the search.

    Returns:
        Best parameters and best ROC-AUC.
    """

    if model_key not in PARAM_GRIDS:
        raise KeyError(f"Unknown model key '{model_key}'.")
    estimators = {
        key: estimator
        for key, _name, estimator in build_estimators(scale_pos_weight=scale_pos_weight)
    }
    pipeline = build_model_pipeline(estimators[model_key])
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=PARAM_GRIDS[model_key],
        n_iter=min(n_iter, _grid_size(PARAM_GRIDS[model_key])),
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=False,
    )
    search.fit(features, target)
    return {
        "model_key": model_key,
        "best_params": {key: _jsonify(value) for key, value in search.best_params_.items()},
        "best_roc_auc": float(search.best_score_),
    }


def _grid_size(grid: dict[str, list[Any]]) -> int:
    size = 1
    for values in grid.values():
        size *= max(len(values), 1)
    return size


def _jsonify(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
