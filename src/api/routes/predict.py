"""Prediction and batch-scoring routes."""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from src.api.dependencies import get_runtime
from src.api.schemas import ApplicantInput, BatchItem, BatchResponse, PredictionResponse
from src.config import MODEL_VERSION, PROJECT_ROOT
from src.data.cleaner import clean_records
from src.data.validator import validate_dataframe
from src.models.predictor import RiskRuntime, recommendation_detail

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
@router.post("/api/v1/predict", response_model=PredictionResponse)
def predict_applicant(
    applicant: ApplicantInput,
    request: Request,
    runtime: RiskRuntime = Depends(get_runtime),
) -> PredictionResponse:
    """Score one applicant and return the decision, tier, and top drivers."""

    started = time.perf_counter()
    frame = _applicant_frame(applicant)
    try:
        explanation = runtime.explain_dataframe(frame)[0]
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    latency_ms = (time.perf_counter() - started) * 1000
    applicant_id = applicant.applicant_id or "APP-TEMP"
    probability = float(explanation["risk_probability"])
    payload = PredictionResponse(
        applicant_id=applicant_id,
        model_name=runtime.model_name,
        model_version=str(runtime.best.get("version", MODEL_VERSION)),
        risk_category=str(explanation["risk_category"]),
        is_default_predicted=probability >= runtime.threshold,
        risk_probability=round(probability, 6),
        risk_score=int(min(1000, max(0, round(probability * 1000)))),
        decision_threshold=runtime.threshold,
        risk_tier=str(explanation["risk_tier"]),
        key_risk_drivers=explanation["key_risk_drivers"],
        key_risk_factors=explanation["key_risk_factors"],
        positive_credit_factors=explanation["positive_credit_factors"],
        recommendation=str(explanation["recommendation"]),
        recommendation_detail=recommendation_detail(str(explanation["recommendation"])),
        summary_statement=explanation["summary_statement"],
        latency_ms=round(latency_ms, 3),
    )
    request.app.state.scored[applicant_id] = payload.model_dump()
    return payload


@router.post("/predict/batch", response_model=BatchResponse)
@router.post("/api/v1/predict/batch/upload", response_model=BatchResponse)
async def predict_batch_file(
    file: UploadFile = File(...),
    runtime: RiskRuntime = Depends(get_runtime),
) -> BatchResponse:
    """Score a CSV upload of applicant rows."""

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc
    return _score_frame(frame, runtime)


@router.post("/api/v1/predict/batch", response_model=BatchResponse)
def predict_batch_json(
    applicants: list[ApplicantInput],
    runtime: RiskRuntime = Depends(get_runtime),
) -> BatchResponse:
    """Score a JSON list of applicants (maximum 5,000 rows)."""

    if len(applicants) > 5000:
        raise HTTPException(status_code=400, detail="Batch limit is 5,000 applicants.")
    frame = pd.DataFrame([item.model_dump() for item in applicants])
    return _score_frame(frame, runtime)


@router.post("/api/v1/data/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, object]:
    """Store an uploaded CSV and return a validation summary. This does not train."""

    content = await file.read()
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Interactive uploads are limited to 100 MB.")
    destination_dir = PROJECT_ROOT / "data" / "raw" / "uploads"
    destination_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(file.filename or "upload.csv").name
    destination = destination_dir / filename
    destination.write_bytes(content)
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc
    validation = validate_dataframe(frame, require_target=TARGET_PRESENT(frame))
    report = None
    if not validation["schema_errors"]:
        _cleaned, clean_report = clean_records(validation["frame"], require_target=TARGET_PRESENT(frame))
        report = clean_report.to_dict()
    return {
        "status": "accepted",
        "stored_at": str(destination),
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "schema_errors": validation["schema_errors"],
        "invalid_row_count": validation["invalid_row_count"],
        "clean_report": report,
    }


def _score_frame(frame: pd.DataFrame, runtime: RiskRuntime) -> BatchResponse:
    if len(frame) == 0:
        raise HTTPException(status_code=400, detail="Batch contains no rows.")
    if len(frame) > 20000:
        raise HTTPException(status_code=400, detail="Batch limit is 20,000 rows.")
    started = time.perf_counter()
    try:
        scored = runtime.predict_dataframe(frame)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    results = [
        BatchItem(
            record_id=int(index),
            applicant_id=None if "applicant_id" not in scored.columns else _text(scored.loc[index, "applicant_id"]),
            risk_category=str(scored.loc[index, "risk_category"]),
            risk_probability=round(float(scored.loc[index, "risk_probability"]), 6),
            recommendation=str(scored.loc[index, "recommendation"]),
        )
        for index in range(len(scored))
    ]
    high = int((scored["risk_category"] == "High Risk").sum())
    elapsed = time.perf_counter() - started
    return BatchResponse(
        total_records_processed=len(results),
        high_risk_count=high,
        low_risk_count=len(results) - high,
        average_risk_probability=round(float(scored["risk_probability"].mean()), 6),
        portfolio_risk_rate=round(high / len(results), 6),
        results=results,
        batch_latency_seconds=round(elapsed, 4),
    )


def _applicant_frame(applicant: ApplicantInput) -> pd.DataFrame:
    payload = applicant.model_dump()
    payload.pop("applicant_id", None)
    return pd.DataFrame([payload])


def _text(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


def TARGET_PRESENT(frame: pd.DataFrame) -> bool:
    """Whether an upload includes the training target."""

    return "loan_status" in {str(column).strip().lower() for column in frame.columns}
