"""Local explanation routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.dependencies import get_runtime
from src.api.schemas import ApplicantInput, ExplainResponse
from src.models.predictor import RiskRuntime

router = APIRouter()


@router.post("/explain", response_model=ExplainResponse)
@router.post("/api/v1/explain", response_model=ExplainResponse)
def explain_applicant(
    applicant: ApplicantInput,
    runtime: RiskRuntime = Depends(get_runtime),
) -> ExplainResponse:
    """Return aggregated SHAP contributions and the adverse-action narrative."""

    frame = applicant.model_dump()
    frame.pop("applicant_id", None)
    import pandas as pd

    explanation = runtime.explain_dataframe(pd.DataFrame([frame]))[0]
    return ExplainResponse(
        applicant_id=applicant.applicant_id,
        model_name=runtime.model_name,
        base_value=explanation["base_value"],
        output_value=explanation["output_value"],
        risk_category=explanation["risk_category"],
        risk_probability=explanation["risk_probability"],
        summary_statement=explanation["summary_statement"],
        adverse_action_notice=explanation["adverse_action_notice"],
        key_risk_factors=explanation["key_risk_factors"],
        positive_credit_factors=explanation["positive_credit_factors"],
        shap_contributions=explanation["shap_contributions"],
    )


@router.get("/api/v1/explain/{applicant_id}", response_model=ExplainResponse)
def explain_cached(applicant_id: str, request: Request, runtime: RiskRuntime = Depends(get_runtime)) -> ExplainResponse:
    """Return the explanation stored when this process last scored ``applicant_id``."""

    cached = request.app.state.scored.get(applicant_id)
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail="No scored applicant with that id is cached. POST /api/v1/explain with the applicant payload.",
        )
    return ExplainResponse(
        applicant_id=applicant_id,
        model_name=runtime.model_name,
        base_value=None,
        output_value=cached["risk_probability"],
        risk_category=cached["risk_category"],
        risk_probability=cached["risk_probability"],
        summary_statement=cached["summary_statement"],
        adverse_action_notice=cached["summary_statement"],
        key_risk_factors=cached["key_risk_factors"],
        positive_credit_factors=cached["positive_credit_factors"],
        shap_contributions=cached["key_risk_drivers"],
    )
