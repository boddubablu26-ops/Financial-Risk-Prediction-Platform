"""Single-command training entry point for the Financial Risk Prediction Platform."""

from __future__ import annotations

import argparse
import os

from src.config import PROJECT_ROOT
from src.data.download_dataset import main as acquire_dataset
from src.models.train import train_project


def main() -> None:
    """Acquire the benchmark dataset, then train and evaluate all five models."""

    parser = argparse.ArgumentParser(description="Run the FRP-ML training pipeline.")
    parser.add_argument("--tune", action="store_true", help="Add a short XGBoost randomized search.")
    parser.add_argument("--fast", action="store_true", help="Smaller ensembles for a smoke test.")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    acquire_dataset()
    train_project(tune=args.tune, fast=args.fast)


if __name__ == "__main__":
    main()
