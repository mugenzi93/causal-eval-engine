import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def _estimate_propensity(df: pd.DataFrame, treatment: str, covariates: list) -> np.ndarray:
    X = df[covariates].fillna(df[covariates].median())
    y = df[treatment]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model.predict_proba(X)[:, 1]


def _match_1to1(
    df: pd.DataFrame, treatment: str, ps: np.ndarray
) -> tuple[pd.DataFrame, list, list]:
    treated_idx = df.index[df[treatment] == 1].tolist()
    control_idx = df.index[df[treatment] == 0].tolist()

    ps_treated = ps[df[treatment] == 1].reshape(-1, 1)
    ps_control = ps[df[treatment] == 0].reshape(-1, 1)

    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    nn.fit(ps_control)
    _, indices = nn.kneighbors(ps_treated)

    matched_control = [control_idx[i[0]] for i in indices]
    matched_df = df.loc[treated_idx + matched_control].copy()
    return matched_df, treated_idx, matched_control


def run_psm(df: pd.DataFrame, config: dict) -> dict:
    treatment = config["treatment"]
    outcome = config["outcome"]
    covariates = config["covariates"]

    ps = _estimate_propensity(df, treatment, covariates)
    matched_df, treated_idx, matched_control_idx = _match_1to1(df, treatment, ps)

    y_t = matched_df.loc[matched_df[treatment] == 1, outcome]
    y_c = matched_df.loc[matched_df[treatment] == 0, outcome]
    ate = y_t.mean() - y_c.mean()

    n = len(y_t)
    se = np.sqrt(y_t.var() / n + y_c.var() / n)
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se

    # matched_weights: 1 for every unit that appears in the matched sample, 0 otherwise
    matched_weights = pd.Series(0.0, index=df.index)
    matched_weights.loc[treated_idx + matched_control_idx] = 1.0

    return {
        "method": "psm",
        "ate": round(float(ate), 4),
        "ci_lower": round(float(ci_lower), 4),
        "ci_upper": round(float(ci_upper), 4),
        "n_matched": int(n),
        "propensity_scores": pd.Series(ps, index=df.index),
        "matched_weights": matched_weights,
    }
