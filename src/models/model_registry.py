"""Model constructors, the subsampled SVM, and artifact input/output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_STATE


class CalibratedSubsampledSVC(BaseEstimator, ClassifierMixin):
    """RBF support-vector classifier with Platt scaling on a stratified subsample.

    A full RBF SVM with probability calibration on every cross-validation fold of
    26,000 rows is too slow for interactive training. The estimator keeps the
    specified RBF kernel and calibrates probabilities, while fitting on a
    deterministic stratified subset.
    """

    def __init__(
        self,
        C: float = 1.0,
        gamma: str = "scale",
        max_samples: int = 2000,
        random_state: int = RANDOM_STATE,
    ) -> None:
        """Store SVM hyperparameters.

        Args:
            C: Soft-margin penalty.
            gamma: RBF kernel width. ``scale`` or ``auto``.
            max_samples: Maximum stratified rows used in ``fit``.
            random_state: Deterministic seed.
        """

        self.C = C
        self.gamma = gamma
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "CalibratedSubsampledSVC":
        """Fit a probability-calibrated RBF SVM.

        Args:
            X: Transformed feature matrix.
            y: Binary labels.

        Returns:
            The fitted estimator.
        """

        matrix = np.asarray(X)
        labels = np.asarray(y).astype(int)
        sample_x, sample_y = stratified_subsample(matrix, labels, int(self.max_samples), int(self.random_state))
        base = SVC(
            C=self.C,
            kernel="rbf",
            gamma=self.gamma,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self.model_ = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
        self.model_.fit(sample_x, sample_y)
        self.classes_ = np.asarray(self.model_.classes_)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return calibrated class probabilities."""

        return np.asarray(self.model_.predict_proba(np.asarray(X)))

    def predict(self, X: Any) -> np.ndarray:
        """Return class labels at the estimator's default 0.5 cutoff."""

        return np.asarray(self.model_.predict(np.asarray(X)))


def stratified_subsample(
    matrix: np.ndarray,
    labels: np.ndarray,
    max_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a deterministic stratified subset.

    Args:
        matrix: Feature matrix.
        labels: Integer labels.
        max_samples: Cap on the number of rows.
        random_state: Seed.

    Returns:
        Subsampled matrix and labels. The full input is returned when it is
        already small enough.
    """

    if len(labels) <= max_samples:
        return matrix, labels
    generator = np.random.RandomState(random_state)
    chosen: list[np.ndarray] = []
    for label in np.unique(labels):
        label_index = np.flatnonzero(labels == label)
        share = int(round(max_samples * (len(label_index) / len(labels))))
        share = max(1, min(share, len(label_index)))
        chosen.append(generator.choice(label_index, size=share, replace=False))
    index = np.concatenate(chosen)
    generator.shuffle(index)
    return matrix[index], labels[index]


def build_estimators(
    scale_pos_weight: float,
    fast: bool = False,
    logistic_c: float = 1.0,
) -> list[tuple[str, str, Any]]:
    """Create the five specification classifiers.

    Args:
        scale_pos_weight: Negative/positive ratio for XGBoost.
        fast: Use smaller ensembles for smoke tests.
        logistic_c: Inverse regularization strength for logistic regression.

    Returns:
        Tuples of ``(file_key, display_name, estimator)``.
    """

    logistic = LogisticRegression(
        C=logistic_c,
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    tree = DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=25,
        min_samples_leaf=10,
        criterion="gini",
        ccp_alpha=0.0001,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    forest = RandomForestClassifier(
        n_estimators=40 if fast else 200,
        max_depth=8 if fast else 12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    boosting = XGBClassifier(
        n_estimators=60 if fast else 250,
        learning_rate=0.05,
        max_depth=4 if fast else 6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=float(scale_pos_weight),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    svm = CalibratedSubsampledSVC(
        C=1.0,
        gamma="scale",
        max_samples=400 if fast else 2000,
        random_state=RANDOM_STATE,
    )
    return [
        ("logistic_regression", "Logistic Regression", logistic),
        ("decision_tree", "Decision Tree", tree),
        ("random_forest", "Random Forest", forest),
        ("xgboost", "XGBoost Classifier", boosting),
        ("svm", "Support Vector Machine", svm),
    ]


def estimator_params(estimator: Any) -> dict[str, Any]:
    """Return JSON-safe constructor parameters."""

    raw = estimator.get_params(deep=False)
    safe: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def dump_json(path: Path, payload: Any) -> None:
    """Write pretty JSON, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    """Load a JSON object, or return an empty dict when the file is missing."""

    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_joblib(path: Path, artifact: Any) -> None:
    """Serialize an artifact with joblib."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_joblib(path: Path) -> Any:
    """Load a joblib artifact."""

    return joblib.load(path)
