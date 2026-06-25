import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def _estimate_propensity(df: pd.DataFrame, treatment: str, covariates: list) -> pd.Series:
    """Fit PS model on complete cases only; returns a Series indexed by those rows."""
    cols = [treatment] + covariates
    complete = df[cols].dropna()
    X = complete[covariates].values
    y = complete[treatment].astype(int).values
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="saga")),
    ])
    model.fit(X, y)
    return pd.Series(model.predict_proba(X)[:, 1], index=complete.index)


def _match_1to1(df: pd.DataFrame, treatment: str, ps: np.ndarray) -> tuple:
    """1:1 nearest-neighbour matching on PS. ps must be a plain numpy array aligned to df."""
    treated_idx = df.index[df[treatment] == 1].tolist()
    control_idx = df.index[df[treatment] == 0].tolist()

    ps_treated = ps[df[treatment].values == 1].reshape(-1, 1)
    ps_control = ps[df[treatment].values == 0].reshape(-1, 1)

    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    nn.fit(ps_control)
    _, indices = nn.kneighbors(ps_treated)

    matched_control_idx = [control_idx[i[0]] for i in indices]
    matched_df = df.loc[treated_idx + matched_control_idx].copy()
    return matched_df, treated_idx, matched_control_idx


def run_psm(df: pd.DataFrame, config: dict) -> dict:
    treatment = config["treatment"]
    outcome   = config["outcome"]
    covariates = config["covariates"]

    # Fit and match within complete cases only
    ps_series   = _estimate_propensity(df, treatment, covariates)
    complete_df = df.loc[ps_series.index]
    ps_arr      = ps_series.values

    matched_df, treated_idx, matched_control_idx = _match_1to1(complete_df, treatment, ps_arr)

    y_t = matched_df.loc[matched_df[treatment] == 1, outcome]
    y_c = matched_df.loc[matched_df[treatment] == 0, outcome]
    ate = y_t.mean() - y_c.mean()

    n  = len(y_t)
    se = np.sqrt(y_t.var() / n + y_c.var() / n)
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se
    t_stat   = ate / se if se > 0 else np.nan
    p_value  = float(2 * stats.t.sf(abs(t_stat), df=2 * n - 2)) if not np.isnan(t_stat) else np.nan

    # matched_weights aligned to full df index (0 for incomplete/unmatched rows)
    matched_weights = pd.Series(0.0, index=df.index)
    matched_weights.loc[treated_idx + matched_control_idx] = 1.0

    return {
        "method":    "psm",
        "ate":       round(float(ate), 4),
        "ci_lower":  round(float(ci_lower), 4),
        "ci_upper":  round(float(ci_upper), 4),
        "p_value":   round(p_value, 4) if not np.isnan(p_value) else None,
        "n_matched": int(n),
        "propensity_scores": ps_series,
        "matched_weights":   matched_weights,
    }
