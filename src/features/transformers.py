"""Domain feature construction and leak-free group median imputation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    AGE_BINS,
    AGE_LABELS,
    DEFAULT_ON_FILE_VALUES,
    GRADE_ORDER,
    HOME_OWNERSHIP_VALUES,
    LOAN_INTENT_VALUES,
)
from src.data.validator import normalize_columns


class LeakFreeImputer(BaseEstimator, TransformerMixin):
    """Impute employment length and interest rate using training-set statistics.

    Employment length uses the median within an age bracket, falling back to the
    global training median. Interest rate uses the median within ``loan_grade``.
    A missingness indicator is added for employment length. Statistics are learned
    only in ``fit``.
    """

    def __init__(self) -> None:
        """Initialize an unfitted imputer. No hyperparameters are learned here."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "LeakFreeImputer":
        """Learn group medians from the training frame.

        Args:
            X: Applicant features. The target column is ignored if present.
            y: Unused. Accepted for scikit-learn pipeline compatibility.

        Returns:
            The fitted imputer.
        """

        frame = self._prepare(X)
        brackets = self._age_bracket(frame["person_age"])
        grouped = frame.groupby(brackets, observed=False)["person_emp_length"].median()
        self.emp_medians_ = {str(key): float(value) for key, value in grouped.items() if pd.notna(value)}
        self.global_emp_ = self._median_or(frame["person_emp_length"], 0.0)
        grade_rates = frame.groupby("loan_grade")["loan_int_rate"].median()
        self.grade_rates_ = {str(key): float(value) for key, value in grade_rates.items() if pd.notna(value)}
        self.global_rate_ = self._median_or(frame["loan_int_rate"], 10.99)
        self.enriched_medians_ = {}
        for column in ("revolving_utilization", "delinq_2yrs", "total_outstanding_debt"):
            if column in frame.columns and frame[column].notna().any():
                self.enriched_medians_[column] = float(frame[column].median())
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted medians. Rows are never dropped.

        Args:
            X: Applicant features in the raw schema.

        Returns:
            A dataframe with employment length and interest rate filled in.
        """

        self._check_fitted()
        frame = self._prepare(X)
        frame["emp_length_missing_flag"] = frame["person_emp_length"].isna().astype(int)
        brackets = self._age_bracket(frame["person_age"]).astype(str)
        bracket_fill = brackets.map(self.emp_medians_)
        frame["person_emp_length"] = (
            frame["person_emp_length"].fillna(bracket_fill).fillna(self.global_emp_).astype(float)
        )
        for grade, rate in self.grade_rates_.items():
            mask = (frame["loan_grade"] == grade) & frame["loan_int_rate"].isna()
            frame.loc[mask, "loan_int_rate"] = rate
        frame["loan_int_rate"] = frame["loan_int_rate"].fillna(self.global_rate_).astype(float)
        recomputed = (frame["loan_amnt"] / frame["person_income"]).round(4)
        if "loan_percent_income" not in frame.columns:
            frame["loan_percent_income"] = recomputed
        else:
            frame["loan_percent_income"] = pd.to_numeric(
                frame["loan_percent_income"], errors="coerce"
            ).fillna(recomputed)
        for column, fill in self.enriched_medians_.items():
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(fill)
        return frame.reset_index(drop=True)

    def to_metadata(self) -> dict[str, Any]:
        """Export fitted medians for the preprocessing metadata artifact."""

        self._check_fitted()
        return {
            "employment_median_by_age_bracket": self.emp_medians_,
            "employment_global_median": self.global_emp_,
            "interest_rate_median_by_grade": self.grade_rates_,
            "interest_rate_global_median": self.global_rate_,
            "enriched_medians": self.enriched_medians_,
        }

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("LeakFreeImputer expects a pandas DataFrame with named columns.")
        frame = normalize_columns(X)
        if "loan_status" in frame.columns:
            frame = frame.drop(columns=["loan_status"])
        for column in ("person_emp_length", "loan_int_rate", "loan_percent_income"):
            if column not in frame.columns:
                frame[column] = np.nan
        numeric = [
            "person_age",
            "person_income",
            "person_emp_length",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_cred_hist_length",
            "revolving_utilization",
            "delinq_2yrs",
            "total_outstanding_debt",
        ]
        for column in numeric:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["person_home_ownership"] = self._clean_category(
            frame["person_home_ownership"], HOME_OWNERSHIP_VALUES, "RENT"
        )
        frame["loan_intent"] = self._clean_category(frame["loan_intent"], LOAN_INTENT_VALUES, "PERSONAL")
        frame["loan_grade"] = self._clean_category(frame["loan_grade"], GRADE_ORDER, "C")
        frame["cb_person_default_on_file"] = self._clean_category(
            frame["cb_person_default_on_file"], DEFAULT_ON_FILE_VALUES, "N"
        )
        frame["person_income"] = frame["person_income"].fillna(0).abs().clip(lower=1.0)
        frame["loan_amnt"] = frame["loan_amnt"].fillna(0).clip(lower=1.0)
        frame["person_age"] = frame["person_age"].fillna(30).clip(lower=18, upper=100)
        frame["cb_person_cred_hist_length"] = (
            frame["cb_person_cred_hist_length"].fillna(0).clip(lower=0, upper=60)
        )
        return frame

    @staticmethod
    def _clean_category(series: pd.Series, allowed: tuple[str, ...] | list[str], fallback: str) -> pd.Series:
        cleaned = series.astype("string").str.strip().str.upper()
        return cleaned.where(cleaned.isin(list(allowed)), fallback).astype(str)

    @staticmethod
    def _age_bracket(age: pd.Series) -> pd.Series:
        return pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True)

    @staticmethod
    def _median_or(series: pd.Series, fallback: float) -> float:
        if series.notna().any():
            value = float(series.median())
            if np.isfinite(value):
                return value
        return fallback

    def _check_fitted(self) -> None:
        if not hasattr(self, "grade_rates_"):
            raise RuntimeError("LeakFreeImputer must be fitted before transform().")


class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    """Build the financial ratios required by the modeling specification.

    The benchmark dataset has no revolving balance, delinquency count, or
    outstanding-debt balance. Those inputs are used when they were present
    during ``fit``; otherwise debt burden is represented by the loan-to-income
    ratio and prior default stands in for delinquency history.
    """

    def __init__(self, epsilon: float = 1.0, loan_term_months: int = 36) -> None:
        """Store ratio parameters.

        Args:
            epsilon: Added to income denominators to avoid division by zero.
            loan_term_months: Amortizing term used for the monthly payment estimate.
        """

        self.epsilon = epsilon
        self.loan_term_months = loan_term_months

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FinancialFeatureEngineer":
        """Remember which enriched columns the training schema actually contains.

        Args:
            X: Imputed applicant frame.
            y: Unused.

        Returns:
            The fitted transformer.
        """

        frame = self._as_frame(X)
        self.use_debt_ = self._usable(frame, "total_outstanding_debt")
        self.use_revolving_ = self._usable(frame, "revolving_utilization")
        self.use_delinq_ = self._usable(frame, "delinq_2yrs")
        self.debt_median_ = self._median(frame, "total_outstanding_debt", 0.0)
        self.util_median_ = self._median(frame, "revolving_utilization", 0.35)
        self.delinq_median_ = self._median(frame, "delinq_2yrs", 0.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append derived financial features.

        Args:
            X: Imputed applicant frame.

        Returns:
            Copy of the frame with ratio, stability, and stress columns added.
        """

        if not hasattr(self, "use_debt_"):
            raise RuntimeError("FinancialFeatureEngineer must be fitted before transform().")
        frame = self._as_frame(X).copy()
        epsilon = float(self.epsilon)
        income = frame["person_income"].astype(float)
        amount = frame["loan_amnt"].astype(float)
        rate = frame["loan_int_rate"].astype(float)
        age = frame["person_age"].astype(float).clip(lower=1.0)
        credit_length = frame["cb_person_cred_hist_length"].astype(float).clip(lower=0.0)
        employment = frame["person_emp_length"].astype(float).clip(lower=0.0)

        frame["loan_to_income_ratio"] = (amount / (income + epsilon)).clip(0.0, 2.0)
        frame["interest_burden_index"] = (amount * (rate / 100.0) / (income + epsilon)).clip(0.0, 2.0)
        frame["cred_hist_age_ratio"] = (credit_length / age).clip(0.0, 1.5)

        grade_map = {grade: index + 1 for index, grade in enumerate(GRADE_ORDER)}
        frame["loan_grade_numeric"] = frame["loan_grade"].map(grade_map).fillna(4).astype(float)
        frame["default_on_file_binary"] = (
            frame["cb_person_default_on_file"].astype(str).str.upper() == "Y"
        ).astype(float)
        frame["historical_risk_weight"] = frame["default_on_file_binary"] * frame["loan_grade_numeric"]
        frame["stability_index"] = ((employment + 1.0) / (credit_length + 1.0)).clip(0.0, 10.0)

        monthly_rate = rate / 100.0 / 12.0
        periods = float(self.loan_term_months)
        growth = (1.0 + monthly_rate) ** periods
        payment_factor = np.where(
            monthly_rate <= 1e-8,
            1.0 / periods,
            monthly_rate * growth / (growth - 1.0),
        )
        monthly_payment = amount * payment_factor
        frame["payment_to_income"] = (monthly_payment / (income / 12.0 + epsilon)).clip(0.0, 5.0)

        if self.use_debt_:
            debt = pd.to_numeric(frame.get("total_outstanding_debt"), errors="coerce").fillna(self.debt_median_)
            frame["debt_to_income"] = (debt / (income + epsilon)).clip(0.0, 5.0)
            frame["disposable_income"] = ((income - 0.35 * debt) / 12.0).clip(lower=-5000.0)
        else:
            frame["debt_to_income"] = frame["loan_to_income_ratio"]
            frame["disposable_income"] = ((income / 12.0) - monthly_payment).clip(lower=-5000.0)

        if self.use_delinq_:
            delinquencies = pd.to_numeric(frame.get("delinq_2yrs"), errors="coerce").fillna(self.delinq_median_)
        else:
            delinquencies = frame["default_on_file_binary"]
        frame["delinq_severity_index"] = (delinquencies * 10.0 / (credit_length + 1.0)).clip(0.0, 50.0)

        rate_norm = (rate / 30.0).clip(0.0, 1.0)
        dti_norm = frame["debt_to_income"].clip(0.0, 1.0)
        if self.use_revolving_:
            utilization = pd.to_numeric(frame.get("revolving_utilization"), errors="coerce").fillna(
                self.util_median_
            )
            utilization_norm = utilization.clip(0.0, 1.0)
            frame["financial_stress_score"] = (
                0.30 * utilization_norm
                + 0.30 * dti_norm
                + 0.25 * rate_norm
                + 0.15 * frame["default_on_file_binary"]
            )
        else:
            burden = frame["loan_to_income_ratio"].clip(0.0, 1.0)
            frame["financial_stress_score"] = (
                0.45 * burden + 0.35 * rate_norm + 0.20 * frame["default_on_file_binary"]
            )
        frame["financial_stress_score"] = frame["financial_stress_score"].clip(0.0, 1.0)
        if "emp_length_missing_flag" not in frame.columns:
            frame["emp_length_missing_flag"] = 0
        frame["emp_length_missing_flag"] = frame["emp_length_missing_flag"].astype(float)
        return frame.reset_index(drop=True)

    @staticmethod
    def _as_frame(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FinancialFeatureEngineer expects a pandas DataFrame.")
        return normalize_columns(X)

    @staticmethod
    def _usable(frame: pd.DataFrame, column: str) -> bool:
        return column in frame.columns and bool(frame[column].notna().any())

    @staticmethod
    def _median(frame: pd.DataFrame, column: str, fallback: float) -> float:
        if column not in frame.columns or not frame[column].notna().any():
            return fallback
        value = float(frame[column].median())
        return value if np.isfinite(value) else fallback
