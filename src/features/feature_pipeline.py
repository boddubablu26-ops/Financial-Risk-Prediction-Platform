"""Scikit-learn preprocessing pipeline: ratios, encoding, and scaling."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from src.config import (
    GRADE_ORDER,
    NOMINAL_FEATURES,
    ORDINAL_FEATURES,
    ROBUST_FEATURES,
    STANDARD_FEATURES,
)
from src.features.transformers import FinancialFeatureEngineer, LeakFreeImputer


def build_column_transformer() -> ColumnTransformer:
    """Assemble encode-and-scale steps for the engineered credit features.

    RobustScaler is used for skewed money amounts and ratios. StandardScaler is
    used for the remaining numeric features. Loan grade is ordinal. Home
    ownership and loan intent are one-hot encoded with unknown levels ignored.

    Returns:
        An unfitted ``ColumnTransformer``.
    """

    robust = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    standard = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    nominal = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    ordinal = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[GRADE_ORDER],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("robust", robust, ROBUST_FEATURES),
            ("standard", standard, STANDARD_FEATURES),
            ("ordinal", ordinal, ORDINAL_FEATURES),
            ("nominal", nominal, NOMINAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_preprocessor() -> Pipeline:
    """Build the leak-free preprocessor used by every classifier.

    Returns:
        Pipeline of group imputation, financial ratios, and the column transformer.
    """

    return Pipeline(
        steps=[
            ("imputer", LeakFreeImputer()),
            ("engineer", FinancialFeatureEngineer()),
            ("columns", build_column_transformer()),
        ]
    )


def build_model_pipeline(estimator: object) -> Pipeline:
    """Wrap a classifier in a fresh copy of the preprocessing pipeline.

    Args:
        estimator: Any scikit-learn compatible classifier with ``predict_proba``.

    Returns:
        Unfitted end-to-end pipeline.
    """

    return Pipeline(
        steps=[
            ("imputer", LeakFreeImputer()),
            ("engineer", FinancialFeatureEngineer()),
            ("columns", build_column_transformer()),
            ("model", estimator),
        ]
    )
