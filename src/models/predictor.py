"""Load the champion pipeline and score or explain applicant records."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import DISPLAY_NAMES
from src.evaluation.evaluator import positive_probabilities
from src.explainability.driver_generator import NaturalLanguageRiskExplainer, build_adverse_action_notice
from src.explainability.shap_explainer import (
    aggregate_contributions,
    build_model_explainer,
    explain_matrix,
    transformed_feature_names,
)
from src.models.model_registry import read_json


class RiskRuntime:
    """In-memory champion model used by the API and the dashboard."""

    def __init__(self) -> None:
        self.pipeline: Any = None
        self.best: dict[str, Any] = {}
        self.comparison: dict[str, Any] = {}
        self.shap_summary: dict[str, Any] = {}
        self.explainer: Any = None
        self.feature_names: list[str] = []
        self.linear_coef_: np.ndarray | None = None
        self.background_mean_: np.ndarray | None = None
        self.linear_intercept_: float = 0.0
        self.error: str | None = None

    @classmethod
    def load(cls) -> "RiskRuntime":
        """Load champion artifacts from the paths in ``src.config``.

        Returns:
            A runtime that is ready when the champion file exists and unpickles.
        """

        from src import config

        runtime = cls()
        path = config.CHAMPION_PATH if config.CHAMPION_PATH.exists() else config.CHAMPION_ALIAS_PATH
        if not path.exists():
            runtime.error = (
                "Champion model is not on disk. Run `python -m src.models.train` first."
            )
            return runtime
        try:
            from src.models.model_registry import load_joblib

            runtime.pipeline = load_joblib(path)
            runtime.best = read_json(config.BEST_MODEL_PATH) or read_json(config.BEST_MODEL_ALIAS_PATH)
            runtime.comparison = read_json(config.COMPARISON_PATH)
            runtime.shap_summary = read_json(config.SHAP_SUMMARY_PATH)
            runtime.feature_names = transformed_feature_names(runtime.pipeline)
            runtime.linear_coef_ = None
            runtime.background_mean_ = None
            model = runtime.pipeline.named_steps["model"]
            background = _load_background()
            if type(model).__name__ == "LogisticRegression" and background is not None:
                coefficient = np.asarray(model.coef_, dtype=float).reshape(-1)
                if coefficient.shape[0] == background.shape[1]:
                    runtime.linear_coef_ = coefficient
                    runtime.background_mean_ = background.mean(axis=0)
                    runtime.linear_intercept_ = float(np.asarray(model.intercept_).reshape(-1)[0])
            runtime.explainer = None
            if runtime.linear_coef_ is None:
                runtime.explainer = build_model_explainer(model, background)
        except Exception as exc:  # Artifact can be absent or incompatible; keep the API up.
            runtime.error = str(exc)
            runtime.pipeline = None
        return runtime

    @property
    def ready(self) -> bool:
        """Whether a champion pipeline is loaded."""

        return self.pipeline is not None

    @property
    def threshold(self) -> float:
        """Cost-optimized decision threshold, defaulting to 0.35."""

        return float(self.best.get("optimal_threshold", 0.35))

    @property
    def model_name(self) -> str:
        """Display name of the champion."""

        return str(self.best.get("model_name", "Champion model"))

    def predict_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Score every row in ``frame``.

        Args:
            frame: Raw applicant columns.

        Returns:
            Copy of the frame with probability, tier, and recommendation columns.
        """

        self._require()
        probabilities = positive_probabilities(self.pipeline, frame)
        scored = frame.copy().reset_index(drop=True)
        scored["risk_probability"] = probabilities
        scored["risk_score"] = np.clip(np.rint(probabilities * 1000), 0, 1000).astype(int)
        scored["decision_threshold"] = self.threshold
        scored["is_default_predicted"] = probabilities >= self.threshold
        scored["risk_category"] = np.where(scored["is_default_predicted"], "High Risk", "Low Risk")
        scored["risk_tier"] = [tier_for(float(value), self.threshold) for value in probabilities]
        scored["recommendation"] = [recommendation_for(float(value), self.threshold) for value in probabilities]
        return scored

    def explain_dataframe(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        """Explain each row with SHAP when a tree explainer is available.

        Args:
            frame: Raw applicant columns.

        Returns:
            One explanation payload per row.
        """

        self._require()
        imputed = self.pipeline.named_steps["imputer"].transform(frame)
        engineered = self.pipeline.named_steps["engineer"].transform(imputed)
        matrix = np.asarray(self.pipeline.named_steps["columns"].transform(engineered), dtype=float)
        probabilities = self._positive_from_matrix(matrix)
        contribution_rows, base_value = self._contributions_from_matrix(matrix, frame)
        narrator = NaturalLanguageRiskExplainer(feature_names=[])
        explanations: list[dict[str, Any]] = []
        for index in range(len(probabilities)):
            record = _display_record(frame.iloc[index].to_dict(), engineered.iloc[index])
            probability = float(probabilities[index])
            category = "High Risk" if probability >= self.threshold else "Low Risk"
            contributions = contribution_rows[index]
            names = [str(item["feature"]) for item in contributions]
            values = [float(item["shap_value"]) for item in contributions]
            narrative = narrator.explain_instance(
                raw_record=record,
                shap_values=values,
                predicted_prob=probability,
                risk_category=category,
                feature_names=names,
            )
            notice = build_adverse_action_notice(category, probability, narrative["key_risk_factors"])
            explanations.append(
                {
                    "base_value": None if base_value is None else _scalar(base_value, index),
                    "output_value": probability,
                    "risk_category": category,
                    "risk_probability": probability,
                    "risk_tier": tier_for(probability, self.threshold),
                    "recommendation": recommendation_for(probability, self.threshold),
                    "summary_statement": narrative["summary_statement"],
                    "adverse_action_notice": notice,
                    "key_risk_factors": narrative["key_risk_factors"],
                    "positive_credit_factors": narrative["positive_credit_factors"],
                    "key_risk_drivers": _drivers(narrative),
                    "shap_contributions": [
                        {
                            "feature": str(item["feature"]),
                            "display_name": DISPLAY_NAMES.get(str(item["feature"]), str(item["feature"])),
                            "shap_value": round(float(item["shap_value"]), 6),
                            "value": _json_safe(item.get("raw_value")),
                        }
                        for item in narrative["raw_contributions"]
                    ],
                }
            )
        return explanations

    def _positive_from_matrix(self, matrix: np.ndarray) -> np.ndarray:
        model = self.pipeline.named_steps["model"]
        probabilities = np.asarray(model.predict_proba(matrix))
        classes = [int(label) for label in list(getattr(model, "classes_", [0, 1]))]
        column = classes.index(1) if 1 in classes else probabilities.shape[1] - 1
        return probabilities[:, column].astype(float)

    def _contributions_from_matrix(
        self,
        matrix: np.ndarray,
        frame: pd.DataFrame,
    ) -> tuple[list[list[dict[str, Any]]], Any]:
        if self.linear_coef_ is not None and matrix.shape[1] == self.linear_coef_.shape[0] and self.feature_names:
            values = (matrix - self.background_mean_) * self.linear_coef_
            rows = [
                aggregate_contributions(self.feature_names, values[index])
                for index in range(values.shape[0])
            ]
            return rows, self.linear_intercept_
        if self.explainer is not None and self.feature_names:
            try:
                shap_matrix, base = explain_matrix(self.explainer, matrix)
                rows = [
                    aggregate_contributions(self.feature_names, shap_matrix[index])
                    for index in range(shap_matrix.shape[0])
                ]
                return rows, base
            except Exception:
                pass
        rows = [_heuristic_contributions(frame.iloc[index].to_dict()) for index in range(len(frame))]
        return rows, None

    def _require(self) -> None:
        if not self.ready:
            raise RuntimeError(self.error or "Model pipeline is not loaded.")


def _load_background() -> np.ndarray | None:
    from src import config

    if not config.SHAP_BACKGROUND_PATH.exists():
        return None
    with np.load(config.SHAP_BACKGROUND_PATH) as payload:
        return np.asarray(payload["matrix"], dtype=float)


def tier_for(probability: float, threshold: float) -> str:
    """Map a default probability onto a pricing tier.

    Args:
        probability: Predicted P(default).
        threshold: Cost-optimized high-risk cutoff.

    Returns:
        One of Prime, Near-Prime, Sub-Prime, or Deep Sub-Prime.
    """

    if probability < min(0.15, threshold):
        return "Prime"
    if probability < threshold:
        return "Near-Prime"
    if probability < 0.60:
        return "Sub-Prime"
    return "Deep Sub-Prime"


def recommendation_for(probability: float, threshold: float) -> str:
    """Map a probability to APPROVE, MANUAL_REVIEW, or REJECT.

    Args:
        probability: Predicted P(default).
        threshold: Cost-optimized high-risk cutoff.

    Returns:
        Decision code.
    """

    if probability < threshold:
        return "APPROVE"
    if probability < max(threshold, 0.55):
        return "MANUAL_REVIEW"
    return "REJECT"


def recommendation_detail(code: str) -> str:
    """Expand a decision code into an underwriter sentence."""

    details = {
        "APPROVE": "Approve — predicted default risk is below the cost-optimized threshold.",
        "MANUAL_REVIEW": "Manual underwriter review required — elevated default risk.",
        "REJECT": "Reject — default risk exceeds underwriting tolerance.",
    }
    return details.get(code, code)


def _heuristic_contributions(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Domain fallback when a tree explainer is unavailable."""

    income = _float(record.get("person_income"), 50_000.0)
    amount = _float(record.get("loan_amnt"), 10_000.0)
    ratio = amount / income if income else 0.0
    rate = _float(record.get("loan_int_rate"), 11.0)
    grade = str(record.get("loan_grade") or "C").upper()
    grade_number = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}.get(grade, 3)
    default_flag = str(record.get("cb_person_default_on_file") or "N").upper() == "Y"
    employment = _float(record.get("person_emp_length"), 4.0)
    credit_length = _float(record.get("cb_person_cred_hist_length"), 6.0)
    home = str(record.get("person_home_ownership") or "RENT").upper()
    items = [
        ("loan_to_income_ratio", (ratio - 0.18) * 1.4),
        ("loan_grade", (grade_number - 3) * 0.12),
        ("loan_int_rate", (rate - 11.0) * 0.02),
        ("default_on_file_binary", 0.35 if default_flag else -0.12),
        ("person_income", (55_000.0 - income) / 250_000.0),
        ("person_emp_length", (4.0 - employment) * 0.03),
        ("cb_person_cred_hist_length", (8.0 - credit_length) * 0.02),
        ("person_home_ownership", 0.08 if home == "RENT" else -0.06),
        ("financial_stress_score", (ratio - 0.2) * 0.5),
    ]
    return [{"feature": name, "shap_value": float(value)} for name, value in items]


def _display_record(raw: dict[str, Any], engineered: pd.Series) -> dict[str, Any]:
    record = dict(raw)
    for column in engineered.index:
        value = engineered[column]
        if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool):
            record[str(column)] = float(value)
        else:
            record[str(column)] = value if not _missing(value) else record.get(str(column))
    return record


def _drivers(narrative: dict[str, Any]) -> list[dict[str, Any]]:
    drivers: list[dict[str, Any]] = []
    indexed = {item["feature"]: item for item in narrative["raw_contributions"]}
    for phrase in narrative["key_risk_factors"]:
        match = _match_phrase(phrase, indexed, positive=True)
        if match:
            drivers.append(
                {
                    "feature": match["feature"],
                    "impact": "Increases Risk",
                    "shap_value": round(float(match["shap_value"]), 6),
                    "description": phrase,
                }
            )
    for phrase in narrative["positive_credit_factors"]:
        match = _match_phrase(phrase, indexed, positive=False)
        if match:
            drivers.append(
                {
                    "feature": match["feature"],
                    "impact": "Mitigates Risk",
                    "shap_value": round(float(match["shap_value"]), 6),
                    "description": phrase,
                }
            )
    return drivers


def _match_phrase(
    phrase: str,
    indexed: dict[str, dict[str, Any]],
    positive: bool,
) -> dict[str, Any] | None:
    ranked = sorted(
        indexed.values(),
        key=lambda item: float(item["shap_value"]),
        reverse=positive,
    )
    pool = [item for item in ranked if (float(item["shap_value"]) > 0) == positive]
    for item in pool:
        feature = str(item["feature"]).replace("_", " ")
        if feature.split(" ")[0].lower() in phrase.lower() or str(item["feature"]) in phrase:
            return item
    return pool[0] if pool else None


def _scalar(base: Any, index: int) -> float:
    array = np.asarray(base).reshape(-1)
    if array.size == 1:
        return float(array[0])
    return float(array[min(index, array.size - 1)])


def _json_safe(value: Any) -> Any:
    """Convert numpy and pandas scalars into JSON-friendly Python values."""

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if _missing(value):
        return None
    return value


def _float(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(number):
        return fallback
    return number


def _missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
