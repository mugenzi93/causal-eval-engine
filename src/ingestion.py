import pandas as pd
import numpy as np
from pathlib import Path


def load_data(config: dict) -> pd.DataFrame:
    path = Path(config["data"]["path"])
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df


def validate_columns(df: pd.DataFrame, config: dict) -> None:
    required = (
        [config["outcome"], config["treatment"]]
        + config["covariates"]
    )
    optional = ["instrument", "running_variable", "time_col", "id_col"]
    for key in optional:
        val = config.get(key) or (config.get("data") or {}).get(key)
        if val and val in df.columns:
            required.append(val)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.isnull().sum()
    pct = counts / len(df) * 100
    report = pd.DataFrame({"missing_n": counts, "missing_pct": pct.round(2)})
    return report[report["missing_n"] > 0]


def balance_summary(df: pd.DataFrame, treatment: str, covariates: list) -> pd.DataFrame:
    rows = []
    for col in covariates:
        if col not in df.columns:
            continue
        treated = df.loc[df[treatment] == 1, col].dropna()
        control = df.loc[df[treatment] == 0, col].dropna()
        mean_t = treated.mean()
        mean_c = control.mean()
        pooled_sd = np.sqrt((treated.std() ** 2 + control.std() ** 2) / 2)
        smd = (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else np.nan
        rows.append({
            "covariate": col,
            "mean_treated": round(mean_t, 3),
            "mean_control": round(mean_c, 3),
            "smd": round(smd, 3),
        })
    return pd.DataFrame(rows)


def ingest(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_data(config)
    validate_columns(df, config)
    miss = missingness_report(df)
    balance = balance_summary(df, config["treatment"], config["covariates"])
    return df, miss, balance
