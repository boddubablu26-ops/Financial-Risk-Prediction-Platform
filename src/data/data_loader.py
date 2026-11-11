"""Dataset ingestion, cleaning orchestration, and processed-split persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    PREPROCESSING_METADATA_PATH,
    PROCESSED_DIR,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)
from src.data.cleaner import clean_records, cleaning_summary
from src.data.splitter import split_balance_report, stratified_split
from src.data.validator import validate_dataframe


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the benchmark CSV from disk.

    Args:
        path: Optional override. Defaults to ``data/raw/credit_risk_dataset.csv``.

    Returns:
        Raw frame with the file's original columns.
    """

    source = Path(path) if path is not None else RAW_DATA_PATH
    if not source.exists():
        raise FileNotFoundError(
            f"Dataset not found at {source}. Run scripts/download_or_generate_dataset.py first."
        )
    return pd.read_csv(source)


def prepare_modeling_tables(
    frame: pd.DataFrame | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Validate, clean, and stratified-split a dataset.

    Imputation is intentionally left to the training pipeline so medians are
    fit on the training partition only. Persisted parquet files keep those
    residual missing values, plus a JSON balance report.

    Args:
        frame: Optional in-memory table. When omitted, the raw benchmark is loaded.
        persist: Write ``data/processed/train.parquet`` and ``test.parquet``.

    Returns:
        Dictionary with cleaned splits, the clean report, and the balance report.
    """

    raw = load_raw_dataset() if frame is None else frame
    validation = validate_dataframe(raw, require_target=True)
    if validation["schema_errors"]:
        missing = ", ".join(validation["schema_errors"])
        raise ValueError(f"Dataset failed schema validation. Missing columns: {missing}")

    cleaned, report = clean_records(validation["frame"], require_target=True)
    x_train, x_test, y_train, y_test = stratified_split(cleaned)
    balance = split_balance_report(y_train, y_test)

    if persist:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        train_out = x_train.copy()
        train_out[TARGET_COLUMN] = y_train.to_numpy()
        test_out = x_test.copy()
        test_out[TARGET_COLUMN] = y_test.to_numpy()
        _write_table(train_out, PROCESSED_DIR / "train.parquet", PROCESSED_DIR / "train.csv")
        _write_table(test_out, PROCESSED_DIR / "test.parquet", PROCESSED_DIR / "test.csv")
        y_train.to_frame(TARGET_COLUMN).to_csv(PROCESSED_DIR / "train_target.csv", index=False)
        y_test.to_frame(TARGET_COLUMN).to_csv(PROCESSED_DIR / "test_target.csv", index=False)
        metadata = {
            "clean_report": cleaning_summary(report),
            "split": balance,
            "imputation": "Deferred to LeakFreeImputer inside the model pipeline.",
        }
        PREPROCESSING_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "cleaned": cleaned,
        "clean_report": report,
        "balance": balance,
        "raw": validation["frame"],
    }


def load_processed_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load persisted train and test tables, preferring parquet."""

    train_path = _existing(PROCESSED_DIR / "train.parquet", PROCESSED_DIR / "train.csv")
    test_path = _existing(PROCESSED_DIR / "test.parquet", PROCESSED_DIR / "test.csv")
    return _read_table(train_path), _read_table(test_path)


def _write_table(frame: pd.DataFrame, parquet_path: Path, csv_path: Path) -> None:
    frame.to_csv(csv_path, index=False)
    try:
        frame.to_parquet(parquet_path, index=False)
    except (ImportError, ValueError, OSError):
        # CSV is always written. Parquet needs pyarrow, which is optional at runtime.
        return


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _existing(primary: Path, fallback: Path) -> Path:
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Neither {primary} nor {fallback} exists. Run the training pipeline.")
