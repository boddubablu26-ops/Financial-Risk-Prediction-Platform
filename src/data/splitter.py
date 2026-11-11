"""Stratified train/test partitioning for the credit default label."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def stratified_split(
    frame: pd.DataFrame,
    target: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a cleaned frame 80/20, stratified on the default label.

    Args:
        frame: Cleaned modeling table containing ``target``.
        target: Binary target column.
        test_size: Holdout fraction.
        random_state: Deterministic seed. The project standard is 42.

    Returns:
        ``X_train``, ``X_test``, ``y_train``, ``y_test``.
    """

    if target not in frame.columns:
        raise ValueError(f"Cannot split without target column '{target}'.")
    features = frame.drop(columns=[target])
    labels = frame[target].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
        shuffle=True,
    )
    return (
        x_train.reset_index(drop=True),
        x_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def split_balance_report(y_train: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    """Compare the positive-class rate of the two partitions."""

    train_rate = float(y_train.mean())
    test_rate = float(y_test.mean())
    return {
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "train_default_rate": train_rate,
        "test_default_rate": test_rate,
        "absolute_rate_gap": abs(train_rate - test_rate),
    }
