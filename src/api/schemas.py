"""Request and response contracts for the risk API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import (
    DEFAULT_ON_FILE_VALUES,
    HOME_OWNERSHIP_VALUES,
    LOAN_GRADE_VALUES,
    LOAN_INTENT_VALUES,
)


class ApplicantInput(BaseModel):
    """One credit applicant. Enriched bureau fields are optional."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    applicant_id: str | None = "APP-TEMP"
    person_age: int = Field(..., ge=18, le=100)
    person_income: float = Field(..., ge=1000)
    person_home_ownership: str
    person_emp_length: float | None = Field(default=None, ge=0.0, le=60.0)
    loan_intent: str
    loan_grade: str
    loan_amnt: float = Field(..., ge=500, le=100_000)
    loan_int_rate: float | None = Field(default=None, ge=0.0, le=40.0)
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int = Field(..., ge=0, le=60)
    loan_percent_income: float | None = Field(default=None, ge=0.0)
    revolving_utilization: float | None = Field(default=None, ge=0.0, le=2.5)
    delinq_2yrs: int | None = Field(default=None, ge=0, le=30)
    total_outstanding_debt: float | None = Field(default=None, ge=0.0)

    @field_validator(
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
        mode="before",
    )
    @classmethod
    def _uppercase(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("person_home_ownership")
    @classmethod
    def _home(cls, value: str) -> str:
        if value not in HOME_OWNERSHIP_VALUES:
            raise ValueError(f"person_home_ownership must be one of {HOME_OWNERSHIP_VALUES}")
        return value

    @field_validator("loan_intent")
    @classmethod
    def _intent(cls, value: str) -> str:
        if value not in LOAN_INTENT_VALUES:
            raise ValueError(f"loan_intent must be one of {LOAN_INTENT_VALUES}")
        return value

    @field_validator("loan_grade")
    @classmethod
    def _grade(cls, value: str) -> str:
        if value not in LOAN_GRADE_VALUES:
            raise ValueError(f"loan_grade must be one of {LOAN_GRADE_VALUES}")
        return value

    @field_validator("cb_person_default_on_file")
    @classmethod
    def _default_flag(cls, value: str) -> str:
        if value not in DEFAULT_ON_FILE_VALUES:
            raise ValueError("cb_person_default_on_file must be Y or N")
        return value


class RiskFactor(BaseModel):
    """One signed driver attached to a prediction."""

    feature: str
    impact: str
    shap_value: float
    description: str


class PredictionResponse(BaseModel):
    """Single-applicant score consumed by the dashboard and external clients."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = "success"
    applicant_id: str
    model_name: str
    model_version: str
    risk_category: str
    is_default_predicted: bool
    risk_probability: float
    risk_score: int
    decision_threshold: float
    risk_tier: str
    key_risk_drivers: list[RiskFactor]
    key_risk_factors: list[str]
    positive_credit_factors: list[str]
    recommendation: Literal["APPROVE", "MANUAL_REVIEW", "REJECT"]
    recommendation_detail: str
    summary_statement: str
    latency_ms: float


class BatchItem(BaseModel):
    """One row of a batch scoring response."""

    record_id: int
    applicant_id: str | None = None
    risk_category: str
    risk_probability: float
    recommendation: str


class BatchResponse(BaseModel):
    """Summary of a scored batch."""

    status: str = "success"
    total_records_processed: int
    high_risk_count: int
    low_risk_count: int
    average_risk_probability: float
    portfolio_risk_rate: float
    results: list[BatchItem]
    batch_latency_seconds: float


class ModelMetricItem(BaseModel):
    """One row of the model leaderboard."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    is_champion: bool
    optimal_threshold: float | None = None


class LeaderboardResponse(BaseModel):
    """Champion pointer plus every trained model."""

    champion_model: str
    evaluation_dataset: str
    models: list[ModelMetricItem]


class ExplainResponse(BaseModel):
    """Local explanation payload."""

    applicant_id: str | None = None
    model_name: str
    base_value: float | None = None
    output_value: float
    risk_category: str
    risk_probability: float
    summary_statement: str
    adverse_action_notice: str
    key_risk_factors: list[str]
    positive_credit_factors: list[str]
    shap_contributions: list[dict[str, Any]]
