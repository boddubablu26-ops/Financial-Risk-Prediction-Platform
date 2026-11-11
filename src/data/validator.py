"""Schema and domain validation for credit-risk applicant records.

Rules follow ``docs/DATASET_SPECIFICATION.md``. Vectorized checks are used for
batch frames. A Pydantic model validates single records the same way the API does.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import (
    DEFAULT_ON_FILE_VALUES,
    HOME_OWNERSHIP_VALUES,
    LOAN_GRADE_VALUES,
    LOAN_INTENT_VALUES,
    REQUIRED_TRAINING_COLUMNS,
    TARGET_COLUMN,
)


class CreditRecord(BaseModel):
    """Single-record bounds check aligned with the dataset specification."""

    model_config = ConfigDict(extra="ignore")

    person_age: int = Field(..., ge=18, le=100)
    person_income: float = Field(..., ge=1000, le=10_000_000)
    person_home_ownership: str
    person_emp_length: float | None = Field(default=None, ge=0.0, le=60.0)
    loan_intent: str
    loan_grade: str
    loan_amnt: float = Field(..., ge=500, le=100_000)
    loan_int_rate: float | None = Field(default=None, ge=3.0, le=40.0)
    loan_status: int | None = Field(default=None)
    loan_percent_income: float | None = Field(default=None, ge=0.0)
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int = Field(..., ge=0, le=60)

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
            raise ValueError("cb_person_default_on_file must be 'Y' or 'N'")
        return value

    @field_validator("loan_status")
    @classmethod
    def _status(cls, value: int | None) -> int | None:
        if value is not None and value not in (0, 1):
            raise ValueError("loan_status must be 0 or 1")
        return value


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, lowercase headers, and turn spaces into underscores."""

    renamed = frame.copy()
    renamed.columns = [str(column).strip().lower().replace(" ", "_") for column in renamed.columns]
    return renamed


def missing_required_columns(frame: pd.DataFrame, require_target: bool = True) -> list[str]:
    """Return canonical columns that are absent from ``frame``."""

    expected = REQUIRED_TRAINING_COLUMNS if require_target else [
        column for column in REQUIRED_TRAINING_COLUMNS if column != TARGET_COLUMN
    ]
    # Employment length, interest rate, and loan percent income may be absent on
    # single-record uploads; they are imputed or recomputed downstream.
    optional = {"person_emp_length", "loan_int_rate", "loan_percent_income"}
    return [column for column in expected if column not in frame.columns and column not in optional]


def row_issue_mask(frame: pd.DataFrame, require_target: bool = True) -> pd.Series:
    """Boolean mask that is True when a row violates a hard domain rule.

    Missing employment length and interest rate are allowed. Impossible ages,
    non-positive income, unknown categories, and an invalid target are not.
    """

    issues = pd.Series(False, index=frame.index)
    age = pd.to_numeric(frame.get("person_age"), errors="coerce")
    income = pd.to_numeric(frame.get("person_income"), errors="coerce")
    amount = pd.to_numeric(frame.get("loan_amnt"), errors="coerce")
    issues = issues | age.isna() | (age < 18) | (age > 100)
    issues = issues | income.isna() | (income.abs() < 1000)
    issues = issues | amount.isna() | (amount <= 0)

    if "person_home_ownership" in frame.columns:
        home = frame["person_home_ownership"].astype("string").str.strip().str.upper()
        issues = issues | ~home.isin(HOME_OWNERSHIP_VALUES)
    else:
        issues = issues | True

    if "loan_intent" in frame.columns:
        intent = frame["loan_intent"].astype("string").str.strip().str.upper()
        issues = issues | ~intent.isin(LOAN_INTENT_VALUES)
    else:
        issues = issues | True

    if "loan_grade" in frame.columns:
        grade = frame["loan_grade"].astype("string").str.strip().str.upper()
        issues = issues | ~grade.isin(LOAN_GRADE_VALUES)
    else:
        issues = issues | True

    if "cb_person_default_on_file" in frame.columns:
        flag = frame["cb_person_default_on_file"].astype("string").str.strip().str.upper()
        issues = issues | ~flag.isin(DEFAULT_ON_FILE_VALUES)
    else:
        issues = issues | True

    if require_target and TARGET_COLUMN in frame.columns:
        status = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
        issues = issues | ~status.isin([0, 1])

    cred = pd.to_numeric(frame.get("cb_person_cred_hist_length"), errors="coerce")
    issues = issues | cred.isna() | (cred < 0) | (cred > 60)
    return issues.fillna(True)


def validate_dataframe(frame: pd.DataFrame, require_target: bool = True) -> dict[str, Any]:
    """Validate a batch and return a structured report.

    Args:
        frame: Raw applicant table.
        require_target: When True, ``loan_status`` must be present and binary.

    Returns:
        Report with schema errors, invalid-row count, and a boolean valid mask.
    """

    normalized = normalize_columns(frame)
    schema_errors = missing_required_columns(normalized, require_target=require_target)
    if schema_errors:
        return {
            "is_valid": False,
            "schema_errors": schema_errors,
            "invalid_row_count": int(len(normalized)),
            "valid_mask": pd.Series(False, index=normalized.index),
            "frame": normalized,
        }

    invalid = row_issue_mask(normalized, require_target=require_target)
    return {
        "is_valid": bool((~invalid).any()) and not schema_errors,
        "schema_errors": [],
        "invalid_row_count": int(invalid.sum()),
        "valid_mask": ~invalid,
        "frame": normalized,
    }
