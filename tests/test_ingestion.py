import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion import load_data, validate_columns, missingness_report, balance_summary, ingest

SAMPLE_CSV = Path(__file__).parent.parent / "data/raw/sample_data.csv"

CONFIG = {
    "data": {"path": str(SAMPLE_CSV), "id_col": "subject_id", "time_col": "year"},
    "outcome": "earnings",
    "treatment": "program_participation",
    "covariates": ["age", "education", "prior_earnings"],
    "instrument": "distance_to_office",
}


def test_load_data():
    df = load_data(CONFIG)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_validate_columns_passes():
    df = load_data(CONFIG)
    validate_columns(df, CONFIG)  # should not raise


def test_validate_columns_fails_on_missing():
    df = load_data(CONFIG)
    bad_config = {**CONFIG, "outcome": "nonexistent_col"}
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_columns(df, bad_config)


def test_missingness_report_empty_for_clean_data():
    df = load_data(CONFIG)
    report = missingness_report(df)
    assert report.empty


def test_balance_summary_shape():
    df = load_data(CONFIG)
    summary = balance_summary(df, CONFIG["treatment"], CONFIG["covariates"])
    assert len(summary) == len(CONFIG["covariates"])
    assert "smd" in summary.columns


def test_ingest_returns_tuple():
    df, miss, balance = ingest(CONFIG)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(miss, pd.DataFrame)
    assert isinstance(balance, pd.DataFrame)
