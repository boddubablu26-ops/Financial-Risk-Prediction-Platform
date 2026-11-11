"""API tests using a tiny in-memory champion so they do not require the full train."""

from __future__ import annotations

import json

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.features.feature_pipeline import build_model_pipeline
from tests.helpers import modeling_frame, sample_payload


def _patch_artifacts(tmp_path, monkeypatch) -> None:
    frame = modeling_frame(20)
    target = frame.pop("loan_status")
    pipeline = build_model_pipeline(
        LogisticRegression(max_iter=400, class_weight="balanced", random_state=42)
    )
    pipeline.fit(frame, target)
    model_path = tmp_path / "champion_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    comparison_path = tmp_path / "model_comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "champion_model": "Logistic Regression",
                "evaluation_dataset": "unit-test",
                "models": [
                    {
                        "model_name": "Logistic Regression",
                        "accuracy": 0.8,
                        "precision": 0.7,
                        "recall": 0.75,
                        "f1_score": 0.72,
                        "roc_auc": 0.86,
                        "pr_auc": 0.7,
                        "is_champion": True,
                        "optimal_threshold": 0.4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    best_path = tmp_path / "best_model_info.json"
    best_path.write_text(
        json.dumps(
            {
                "model_name": "Logistic Regression",
                "version": "1.0.0",
                "optimal_threshold": 0.4,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.config.CHAMPION_PATH", model_path)
    monkeypatch.setattr("src.config.CHAMPION_ALIAS_PATH", model_path)
    monkeypatch.setattr("src.config.COMPARISON_PATH", comparison_path)
    monkeypatch.setattr("src.config.BEST_MODEL_PATH", best_path)
    monkeypatch.setattr("src.config.BEST_MODEL_ALIAS_PATH", best_path)
    monkeypatch.setattr("src.config.SHAP_SUMMARY_PATH", tmp_path / "shap.json")


def test_health_predict_explain_and_models(tmp_path, monkeypatch) -> None:
    _patch_artifacts(tmp_path, monkeypatch)
    from src.api.main import app

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["pipeline_loaded"] is True

        scored = client.post("/api/v1/predict", json=sample_payload())
        assert scored.status_code == 200
        body = scored.json()
        assert 0.0 <= body["risk_probability"] <= 1.0
        assert body["risk_category"] in {"Low Risk", "High Risk"}
        assert body["recommendation"] in {"APPROVE", "MANUAL_REVIEW", "REJECT"}
        assert body["key_risk_factors"]
        assert body["latency_ms"] < 2000

        rejected = dict(sample_payload())
        rejected["person_age"] = 10
        invalid = client.post("/predict", json=rejected)
        assert invalid.status_code == 422

        explained = client.post("/explain", json=sample_payload())
        assert explained.status_code == 200
        assert explained.json()["shap_contributions"]

        leaderboard = client.get("/models")
        assert leaderboard.status_code == 200
        assert leaderboard.json()["models"][0]["is_champion"] is True

        batch = client.post("/api/v1/predict/batch", json=[sample_payload(), sample_payload()])
        assert batch.status_code == 200
        assert batch.json()["total_records_processed"] == 2

        cached = client.get("/api/v1/explain/APP-TEST-1")
        assert cached.status_code == 200


def test_batch_csv_upload(tmp_path, monkeypatch) -> None:
    _patch_artifacts(tmp_path, monkeypatch)
    from src.api.main import app

    frame = pd.DataFrame([sample_payload(), sample_payload()])
    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            files={"file": ("applicants.csv", frame.to_csv(index=False), "text/csv")},
        )
        assert response.status_code == 200
        assert response.json()["total_records_processed"] == 2
