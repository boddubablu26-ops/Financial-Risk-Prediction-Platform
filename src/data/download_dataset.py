"""Acquire the benchmark dataset via the project generator script.

Running ``python -m src.data.download_dataset`` writes
``data/raw/credit_risk_dataset.csv`` when it is not already present.
"""

from __future__ import annotations

import os
import runpy

from src.config import PROJECT_ROOT


def main() -> None:
    """Execute the dataset acquisition script from the project root."""

    os.chdir(PROJECT_ROOT)
    script = PROJECT_ROOT / "scripts" / "download_or_generate_dataset.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
