"""Shared applicant tables for the unit tests."""

from __future__ import annotations

import pandas as pd


def sample_payload() -> dict[str, object]:
    """One valid API applicant."""

    return {
        "applicant_id": "APP-TEST-1",
        "person_age": 30,
        "person_income": 65000.0,
        "person_home_ownership": "RENT",
        "person_emp_length": 5.0,
        "loan_intent": "DEBTCONSOLIDATION",
        "loan_grade": "C",
        "loan_amnt": 12500.0,
        "loan_int_rate": 11.89,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 6,
    }


def modeling_frame(rows_per_class: int = 30) -> pd.DataFrame:
    """Small labeled frame with both classes and varied numeric ranges."""

    records = []
    for index in range(rows_per_class * 2):
        default = index % 2
        grade = "A" if default == 0 else "D"
        records.append(
            {
                "person_age": 25 + (index % 20),
                "person_income": 40000 + index * 1500,
                "person_home_ownership": "MORTGAGE" if default == 0 else "RENT",
                "person_emp_length": float(index % 12),
                "loan_intent": "EDUCATION" if index % 3 == 0 else "MEDICAL",
                "loan_grade": grade,
                "loan_amnt": 4000 + (index % 10) * 800,
                "loan_int_rate": 7.5 if default == 0 else 16.5,
                "loan_status": default,
                "loan_percent_income": 0.1,
                "cb_person_default_on_file": "N" if default == 0 else "Y",
                "cb_person_cred_hist_length": 4 + (index % 8),
            }
        )
    return pd.DataFrame(records)
