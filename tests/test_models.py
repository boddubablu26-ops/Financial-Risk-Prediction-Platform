"""Tests for metrics, serialization, and deterministic scoring."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.config import BEST_MODEL_PATH, CHAMPION_PATH
from src.evaluation.cost_matrix import expected_cost, optimize_threshold
from src.evaluation.evaluator import positive_probabilities
from src.explainability.driver_generator import NaturalLanguageRiskExplainer
from src.features.feature_pipeline import build_model_pipeline
from src.models.model_registry import load_joblib
from tests.helpers import modeling_frame, sample_payload


def test_expected_cost_counts_false_negatives_five_times() -> None:
    truth = [0, 0, 1, 1]
    scores = [0.1, 0.8, 0.2, 0.9]
    result = expected_cost(truth, scores, threshold=0.5)
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["expected_cost"] == 6


def test_threshold_search_returns_a_grid_point() -> None:
    generator = np.random.default_rng(42)
    truth = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.05, 0.2, 0.55, 0.4, 0.7, 0.9])
    best = optimize_threshold(truth, scores)
    assert 0.10 <= best["optimal_threshold"] <= 0.60
    assert best["minimum_cost"] <= expected_cost(truth, scores, 0.5)["expected_cost"]
    assert generator is not None


def test_tiny_pipeline_probabilities_are_deterministic() -> None:
    frame = modeling_frame(30)
    target = frame.pop("loan_status")
    pipeline = build_model_pipeline(
        LogisticRegression(max_iter=400, class_weight="balanced", random_state=42)
    )
    pipeline.fit(frame, target)
    first = positive_probabilities(pipeline, frame.head(8))
    second = positive_probabilities(pipeline, frame.head(8))
    assert np.allclose(first, second)
    assert np.all((first >= 0.0) & (first <= 1.0))


def test_narrative_mentions_the_dominant_driver() -> None:
    explainer = NaturalLanguageRiskExplainer()
    narrative = explainer.explain_instance(
        raw_record=sample_payload(),
        shap_values=[0.42, -0.2, 0.1],
        predicted_prob=0.84,
        risk_category="High Risk",
        feature_names=["loan_to_income_ratio", "person_income", "loan_int_rate"],
    )
    assert narrative["key_risk_factors"]
    assert "income" in narrative["key_risk_factors"][0].lower()
    assert narrative["positive_credit_factors"]


def test_champion_artifact_scores_when_trained() -> None:
    if not CHAMPION_PATH.exists():
        return
    pipeline = load_joblib(CHAMPION_PATH)
    frame = modeling_frame(5).drop(columns=["loan_status"]).head(3)
    probabilities = positive_probabilities(pipeline, frame)
    assert probabilities.shape == (3,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    if BEST_MODEL_PATH.exists():
        import json

        metadata = json.loads(BEST_MODEL_PATH.read_text(encoding="utf-8"))
        assert "roc_auc" in metadata["metrics"]
        assert metadata["metrics"]["roc_auc"] >= 0.85
        assert metadata["metrics"]["recall"] >= 0.70
