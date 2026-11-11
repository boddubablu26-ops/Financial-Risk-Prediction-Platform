"""Tests for validation, cleaning, and the stratified split."""

from __future__ import annotations

import pandas as pd

from src.data.cleaner import clean_records
from src.data.data_loader import prepare_modeling_tables
from src.data.splitter import stratified_split
from src.data.validator import CreditRecord, validate_dataframe
from src.features.transformers import LeakFreeImputer
from tests.helpers import modeling_frame


def test_credit_record_rejects_impossible_age() -> None:
    try:
        CreditRecord(
            person_age=130,
            person_income=50000,
            person_home_ownership="RENT",
            person_emp_length=2,
            loan_intent="EDUCATION",
            loan_grade="B",
            loan_amnt=5000,
            loan_int_rate=10,
            cb_person_default_on_file="N",
            cb_person_cred_hist_length=4,
        )
    except Exception as exc:
        assert "person_age" in str(exc).lower() or "less than or equal" in str(exc).lower()
        return
    raise AssertionError("Age 130 should fail validation.")


def test_cleaner_drops_impossible_age_and_clamps_employment() -> None:
    frame = modeling_frame(4)
    frame.loc[0, "person_age"] = 144
    frame.loc[1, "person_age"] = 30
    frame.loc[1, "person_emp_length"] = 40
    duplicated = frame.iloc[[2]].copy()
    frame = pd.concat([frame, duplicated], ignore_index=True)
    frame.loc[3, "person_home_ownership"] = "CASTLE"
    cleaned, report = clean_records(frame, require_target=True)
    assert report.dropped_age >= 1
    assert report.dropped_duplicates >= 1
    assert report.dropped_category >= 1
    assert cleaned["person_age"].max() <= 100
    clamped = cleaned.loc[cleaned["person_age"] == 30, "person_emp_length"]
    if len(clamped):
        assert float(clamped.max()) <= 12


def test_grade_median_imputation_uses_training_groups() -> None:
    frame = pd.DataFrame(
        [
            _row(22, 1.0, "A", 7.0, 0),
            _row(23, 3.0, "A", 9.0, 0),
            _row(22, None, "A", None, 1),
            _row(40, 10.0, "B", 15.0, 0),
            _row(42, None, "B", None, 1),
        ]
    )
    imputer = LeakFreeImputer().fit(frame.drop(columns=["loan_status"]))
    transformed = imputer.transform(frame.drop(columns=["loan_status"]))
    young_missing = transformed.loc[transformed["person_age"] == 22].iloc[-1]
    older_missing = transformed.loc[transformed["person_age"] == 42].iloc[0]
    assert young_missing["person_emp_length"] == 2
    assert older_missing["person_emp_length"] == 10
    assert young_missing["loan_int_rate"] == 8
    assert older_missing["loan_int_rate"] == 15
    assert int(young_missing["emp_length_missing_flag"]) == 1
    assert transformed.isna().sum().sum() == 0


def test_stratified_split_preserves_default_rate() -> None:
    frame = modeling_frame(40)
    _x_train, _x_test, y_train, y_test = stratified_split(frame)
    assert abs(float(y_train.mean()) - float(y_test.mean())) < 0.02


def test_benchmark_clean_and_split_when_dataset_exists() -> None:
    tables = prepare_modeling_tables(persist=False)
    assert tables["clean_report"].rows_out > 30_000
    assert tables["cleaned"]["person_age"].max() <= 100
    assert tables["balance"]["absolute_rate_gap"] < 0.005
    validation = validate_dataframe(tables["raw"], require_target=True)
    assert validation["schema_errors"] == []


def _row(age: int, employment: float | None, grade: str, rate: float | None, status: int) -> dict[str, object]:
    return {
        "person_age": age,
        "person_income": 50000.0,
        "person_home_ownership": "RENT",
        "person_emp_length": employment,
        "loan_intent": "EDUCATION",
        "loan_grade": grade,
        "loan_amnt": 10000.0,
        "loan_int_rate": rate,
        "loan_status": status,
        "loan_percent_income": 0.2,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 5,
    }
