"""Tests for financial ratios and the preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.feature_pipeline import build_preprocessor
from src.features.transformers import FinancialFeatureEngineer, LeakFreeImputer
from tests.helpers import modeling_frame


def test_ratio_formulas_match_the_specification() -> None:
    frame = pd.DataFrame(
        [
            {
                "person_age": 30,
                "person_income": 60000.0,
                "person_home_ownership": "RENT",
                "person_emp_length": 4.0,
                "loan_intent": "PERSONAL",
                "loan_grade": "B",
                "loan_amnt": 12000.0,
                "loan_int_rate": 12.0,
                "loan_percent_income": 0.2,
                "cb_person_default_on_file": "Y",
                "cb_person_cred_hist_length": 6,
            }
        ]
    )
    imputed = LeakFreeImputer().fit_transform(frame)
    engineered = FinancialFeatureEngineer().fit_transform(imputed)
    assert engineered.loc[0, "loan_to_income_ratio"] == pytest_approx(12000 / 60001)
    assert engineered.loc[0, "interest_burden_index"] == pytest_approx(12000 * 0.12 / 60001)
    assert engineered.loc[0, "cred_hist_age_ratio"] == pytest_approx(6 / 30)
    monthly_rate = 0.12 / 12
    periods = 36
    growth = (1 + monthly_rate) ** periods
    payment = 12000 * monthly_rate * growth / (growth - 1)
    assert engineered.loc[0, "payment_to_income"] == pytest_approx(payment / (60000 / 12 + 1))
    assert engineered.loc[0, "default_on_file_binary"] == 1
    assert engineered.loc[0, "historical_risk_weight"] == 2
    assert 0 <= float(engineered.loc[0, "financial_stress_score"]) <= 1


def test_preprocessor_has_no_nulls_and_stable_shape() -> None:
    frame = modeling_frame(25)
    target = frame.pop("loan_status")
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(frame, target)
    matrix = np.asarray(transformed, dtype=float)
    holdout = preprocessor.transform(frame.iloc[:5])
    assert matrix.shape[0] == len(frame)
    assert matrix.shape[1] > 10
    assert not np.isnan(matrix).any()
    assert np.asarray(holdout).shape[1] == matrix.shape[1]


def pytest_approx(value: float) -> object:
    """Local wrapper so this module does not depend on pytest at import time for helpers."""

    import pytest

    return pytest.approx(value, rel=1e-5, abs=1e-6)
