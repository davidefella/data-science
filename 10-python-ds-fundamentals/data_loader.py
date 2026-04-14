"""
Shared utility for loading datasets from 00-datasets/.
Used by all demo scripts in 10-python-ds-fundamentals/.
"""

import csv
from pathlib import Path

# .parents[1] goes up 1 dir: 10-python-ds-fundamentals → data-science
DATASETS_DIR = Path(__file__).resolve().parents[1] / "00-datasets"


def load_csv(filename):
    """Load a CSV file from 00-datasets/ and return a list of dicts.

    Each dict is one row, with column names as keys (from the CSV header).
    All values are strings — numeric conversion is the caller's responsibility.

    Example:
        data = load_csv("wallet.csv")
        # data → [{"date": "2026-03-31", "amount": "-7.5", ...}, {...}, ...]
    """
    path = DATASETS_DIR / filename
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))
