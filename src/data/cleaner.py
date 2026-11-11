"""Row-level cleaning and leak-free imputation for credit-risk records.

Row filters (impossible ages, illegal categories, duplicates) do not learn from
data and run before the train/test split. Median imputation is intentionally
not done here. ``LeakFreeImputer`` in the feature package learns medians on the
training partition only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_ON_FILE_VALUES,
    HOME_OWNERSHIP_VALUES,
    LOAN_GRADE_VALUES,
    LOAN_INTENT_VALUES,
    TARGET_COLUMN,
)
from src.data.validator import normalize_columns


@dataclass
class CleanReport:
    """Counts produced by a cleaning pass."""

    rows_in: int
    rows_out: int
    dropped_duplicates: int
    dropped_age: int
    dropped_income: int
    dropped_amount: int
    dropped_category: int
    dropped_target: int
    clamped_employment: int
    clamped_credit_history: int
    winsorized_interest: int
    winsorized_income: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _as_category(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def clean_records(frame: pd.DataFrame, require_target: bool = True) -> tuple[pd.DataFrame, CleanReport]:
    """Apply domain guardrails that do not require fitted statistics.

    Args:
        frame: Raw applicant table, with or without the target column.
        require_target: Drop rows whose ``loan_status`` is not 0 or 1.

    Returns:
        Cleaned frame and a report of drops and clamps. Employment length and
        interest rate may still contain missing values.
    """

    original_rows = int(len(frame))
    data = normalize_columns(frame)

    feature_cols = [column for column in data.columns if column != "applicant_id"]
    before_dupes = len(data)
    data = data.drop_duplicates(subset=feature_cols, keep="first").copy()
    dropped_duplicates = before_dupes - len(data)

    data["person_age"] = pd.to_numeric(data["person_age"], errors="coerce")
    data["person_income"] = pd.to_numeric(data["person_income"], errors="coerce").abs()
    data["loan_amnt"] = pd.to_numeric(data["loan_amnt"], errors="coerce")
    if "person_emp_length" not in data.columns:
        data["person_emp_length"] = np.nan
    data["person_emp_length"] = pd.to_numeric(data["person_emp_length"], errors="coerce")
    if "loan_int_rate" not in data.columns:
        data["loan_int_rate"] = np.nan
    data["loan_int_rate"] = pd.to_numeric(data["loan_int_rate"], errors="coerce")
    data["cb_person_cred_hist_length"] = pd.to_numeric(
        data["cb_person_cred_hist_length"], errors="coerce"
    )

    for column, allowed in (
        ("person_home_ownership", HOME_OWNERSHIP_VALUES),
        ("loan_intent", LOAN_INTENT_VALUES),
        ("loan_grade", LOAN_GRADE_VALUES),
        ("cb_person_default_on_file", DEFAULT_ON_FILE_VALUES),
    ):
        data[column] = _as_category(data[column])
        invalid_category = ~data[column].isin(allowed)
        # Counted below after the combined category mask is applied once.
        data.loc[invalid_category, column] = pd.NA

    category_bad = (
        data["person_home_ownership"].isna()
        | data["loan_intent"].isna()
        | data["loan_grade"].isna()
        | data["cb_person_default_on_file"].isna()
    )
    dropped_category = int(category_bad.sum())
    data = data.loc[~category_bad].copy()

    dropped_target = 0
    if require_target:
        if TARGET_COLUMN not in data.columns:
            raise ValueError("Training data is missing required column 'loan_status'.")
        data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
        bad_target = ~data[TARGET_COLUMN].isin([0, 1])
        dropped_target = int(bad_target.sum())
        data = data.loc[~bad_target].copy()
        data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)

    income_bad = data["person_income"].isna() | (data["person_income"] < 1000)
    dropped_income = int(income_bad.sum())
    data = data.loc[~income_bad].copy()

    amount_bad = data["loan_amnt"].isna() | (data["loan_amnt"] <= 0)
    dropped_amount = int(amount_bad.sum())
    data = data.loc[~amount_bad].copy()

    age_bad = data["person_age"].isna() | (data["person_age"] < 18) | (data["person_age"] > 100)
    dropped_age = int(age_bad.sum())
    data = data.loc[~age_bad].copy()

    cred_bad = data["cb_person_cred_hist_length"].isna() | (data["cb_person_cred_hist_length"] < 0)
    data = data.loc[~cred_bad].copy()

    # Floor modest incomes at the underwriting minimum and cap extreme incomes.
    winsor_income_mask = (data["person_income"] < 4000) | (data["person_income"] > 2_500_000)
    winsorized_income = int(winsor_income_mask.sum())
    data["person_income"] = data["person_income"].clip(lower=4000, upper=2_500_000)
    data["loan_amnt"] = data["loan_amnt"].clip(lower=500, upper=50_000)

    max_emp = (data["person_age"] - 18).clip(lower=0)
    emp_anomaly = data["person_emp_length"].notna() & (
        data["person_emp_length"] > (data["person_age"] - 14)
    )
    clamped_employment = int(emp_anomaly.sum())
    data.loc[emp_anomaly, "person_emp_length"] = max_emp.loc[emp_anomaly]
    data.loc[data["person_emp_length"] < 0, "person_emp_length"] = 0.0

    max_cred = (data["person_age"] - 18).clip(lower=0)
    cred_anomaly = data["cb_person_cred_hist_length"] > (data["person_age"] - 14)
    clamped_credit_history = int(cred_anomaly.sum())
    data.loc[cred_anomaly, "cb_person_cred_hist_length"] = max_cred.loc[cred_anomaly]
    data["cb_person_cred_hist_length"] = data["cb_person_cred_hist_length"].clip(lower=0)

    rate = data["loan_int_rate"]
    rate_anomaly = rate.notna() & ((rate < 2.0) | (rate > 40.0))
    winsorized_interest = int(rate_anomaly.sum())
    data.loc[rate.notna() & (rate < 2.0), "loan_int_rate"] = 3.0
    data.loc[rate.notna() & (rate > 40.0), "loan_int_rate"] = 36.0

    data["loan_percent_income"] = (data["loan_amnt"] / data["person_income"]).round(4)
    data["person_age"] = data["person_age"].round().astype(int)
    data["cb_person_cred_hist_length"] = data["cb_person_cred_hist_length"].round().astype(int)

    for optional in ("revolving_utilization", "delinq_2yrs", "total_outstanding_debt"):
        if optional in data.columns:
            data[optional] = pd.to_numeric(data[optional], errors="coerce")

    data = data.reset_index(drop=True)
    report = CleanReport(
        rows_in=original_rows,
        rows_out=int(len(data)),
        dropped_duplicates=int(dropped_duplicates),
        dropped_age=dropped_age,
        dropped_income=dropped_income,
        dropped_amount=dropped_amount,
        dropped_category=dropped_category,
        dropped_target=dropped_target,
        clamped_employment=clamped_employment,
        clamped_credit_history=clamped_credit_history,
        winsorized_interest=winsorized_interest,
        winsorized_income=winsorized_income,
    )
    return data, report


def cleaning_summary(report: CleanReport) -> dict[str, Any]:
    """JSON-ready view of a clean report plus the retention rate."""

    payload = report.to_dict()
    payload["retention_rate"] = (
        0.0 if report.rows_in == 0 else round(report.rows_out / report.rows_in, 6)
    )
    return payload
