"""Model leaderboard routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import LeaderboardResponse, ModelMetricItem

router = APIRouter()


@router.get("/models", response_model=LeaderboardResponse)
@router.get("/api/v1/models", response_model=LeaderboardResponse)
@router.get("/api/v1/model/metrics", response_model=LeaderboardResponse)
def model_leaderboard(request: Request) -> LeaderboardResponse:
    """Return holdout metrics for every trained model."""

    runtime = getattr(request.app.state, "runtime", None)
    comparison = getattr(runtime, "comparison", None) or {}
    models = comparison.get("models") or []
    if not models:
        raise HTTPException(status_code=503, detail="Model leaderboard has not been generated yet.")
    return LeaderboardResponse(
        champion_model=str(comparison.get("champion_model", runtime.model_name if runtime else "unknown")),
        evaluation_dataset=str(comparison.get("evaluation_dataset", "holdout")),
        models=[
            ModelMetricItem(
                model_name=item["model_name"],
                accuracy=item["accuracy"],
                precision=item["precision"],
                recall=item["recall"],
                f1_score=item["f1_score"],
                roc_auc=item["roc_auc"],
                pr_auc=item["pr_auc"],
                is_champion=bool(item.get("is_champion", False)),
                optimal_threshold=item.get("optimal_threshold"),
            )
            for item in models
        ],
    )
