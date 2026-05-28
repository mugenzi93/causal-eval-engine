import numpy as np
import pandas as pd


def _2sls(y: np.ndarray, d: np.ndarray, z: np.ndarray, X: np.ndarray) -> tuple:
    """Manual two-stage least squares."""
    # First stage: regress D on Z and X
    Xz = np.column_stack([X, z])
    beta_fs = np.linalg.lstsq(Xz, d, rcond=None)[0]
    d_hat = Xz @ beta_fs
    f_stat = _f_stat_first_stage(d, d_hat, Xz)

    # Second stage: regress Y on D_hat and X
    Xd = np.column_stack([X, d_hat])
    beta_ss = np.linalg.lstsq(Xd, y, rcond=None)[0]
    late = beta_ss[-1]

    residuals = y - np.column_stack([X, d]) @ beta_ss
    sigma2 = (residuals ** 2).sum() / (len(y) - Xd.shape[1])
    cov = sigma2 * np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(cov[-1, -1])

    return float(late), float(se), float(f_stat)


def _f_stat_first_stage(d: np.ndarray, d_hat: np.ndarray, Xz: np.ndarray) -> float:
    rss_restricted = ((d - d.mean()) ** 2).sum()
    rss_full = ((d - d_hat) ** 2).sum()
    n, k = Xz.shape
    f = ((rss_restricted - rss_full) / 1) / (rss_full / (n - k))
    return f


def run_iv(df: pd.DataFrame, config: dict) -> dict:
    treatment = config["treatment"]
    outcome = config["outcome"]
    instrument = config.get("instrument")
    covariates = config.get("covariates", [])

    if not instrument or instrument not in df.columns:
        return {
            "method": "iv",
            "error": "instrument not specified or not found in data.",
        }

    sub = df[[outcome, treatment, instrument] + covariates].dropna()
    y = sub[outcome].values
    d = sub[treatment].values
    z = sub[instrument].values
    X = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in covariates])

    late, se, f_stat = _2sls(y, d, z, X)
    ci_lower = late - 1.96 * se
    ci_upper = late + 1.96 * se

    weak_instrument = f_stat < 10

    return {
        "method": "iv",
        "late": round(late, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "first_stage_f": round(f_stat, 2),
        "weak_instrument_warning": weak_instrument,
        "n_obs": len(sub),
    }
