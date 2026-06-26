import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_drdid(df: pd.DataFrame, config: dict) -> dict:
    """Doubly-Robust DiD (Sant'anna & Zhao 2020).

    Works for both panel and repeated cross-section designs.  Consistent if
    either the propensity score model or the outcome regression model is
    correctly specified.  Identification relies on conditional parallel trends
    rather than unconditional parallel trends.

    Influence function (per observation):
        psi_i = (D_i/p - (1-D_i)*e_i/((1-e_i)*p))
                * (time_weight_i * Y_i - delta_mu0_i)

    where:
        p            = E[D]
        pi           = E[T]  (share of post-period observations)
        time_weight  = T/pi - (1-T)/(1-pi)
        delta_mu0(X) = mu_{0,post}(X) - mu_{0,pre}(X)  (outcome model for
                       control units — counterfactual DiD for each covariate
                       profile)

    Point estimate: mean(psi_i)
    SE:             std(psi_i) / sqrt(n)
    """
    treatment  = config["treatment"]
    outcome    = config["outcome"]
    time_col   = config.get("data", {}).get("time_col")
    covariates = config.get("covariates", [])

    if time_col is None or time_col not in df.columns:
        return {
            "method": "drdid",
            "error": "time_col not specified or not found; DR-DiD requires a pre/post time indicator.",
        }

    times = sorted(df[time_col].unique())
    if len(times) < 2:
        return {"method": "drdid", "error": "DR-DiD requires at least two time periods."}

    t_pre, t_post = times[0], times[-1]

    # Restrict to two periods and complete cases
    df2 = df[df[time_col].isin([t_pre, t_post])].copy()
    df2["_post"] = (df2[time_col] == t_post).astype(int)

    needed = [treatment, outcome, "_post"] + [c for c in covariates if c in df2.columns]
    df2 = df2[needed].dropna().reset_index(drop=True)

    covs_used = [c for c in covariates if c in df2.columns]
    D = df2[treatment].astype(int).values
    T = df2["_post"].values
    Y = df2[outcome].values.astype(float)
    X = df2[covs_used].values.astype(float)
    n = len(df2)

    p_bar  = D.mean()   # P(D = 1)
    pi_bar = T.mean()   # P(post period)

    if p_bar <= 0 or p_bar >= 1:
        return {"method": "drdid", "error": "Treatment is constant — cannot estimate propensity scores."}
    if pi_bar <= 0 or pi_bar >= 1:
        return {"method": "drdid", "error": "All observations are in the same time period."}

    # ── Step 1: propensity score (complete cases only) ───────────────────────
    ps_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=2000, solver="saga")),
    ])
    ps_model.fit(X, D)
    e = np.clip(ps_model.predict_proba(X)[:, 1], 0.01, 0.99)

    # ── Step 2: outcome models for control group ─────────────────────────────
    # mu_{0,pre}(X) and mu_{0,post}(X) fitted on control units in each period
    def _fit_ridge(X_fit, y_fit, X_pred):
        m = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])
        m.fit(X_fit, y_fit)
        return m.predict(X_pred)

    mask_c0 = (D == 0) & (T == 0)   # control, pre
    mask_c1 = (D == 0) & (T == 1)   # control, post

    if mask_c0.sum() < 3 or mask_c1.sum() < 3:
        return {
            "method": "drdid",
            "error": "Too few control observations in one or both periods to fit outcome models.",
        }

    mu_pre  = _fit_ridge(X[mask_c0], Y[mask_c0], X)   # mu_{0,pre}(X)
    mu_post = _fit_ridge(X[mask_c1], Y[mask_c1], X)   # mu_{0,post}(X)

    # Counterfactual DiD prediction for each unit
    delta_mu0 = mu_post - mu_pre   # E[Y(0,post) - Y(0,pre) | X]

    # ── Step 3: efficient influence function ─────────────────────────────────
    time_weight = T / pi_bar - (1 - T) / (1 - pi_bar)
    iw          = D / p_bar - (1 - D) * e / ((1 - e) * p_bar)

    psi = iw * (time_weight * Y - delta_mu0)

    tau      = float(psi.mean())
    se       = float(psi.std()) / np.sqrt(n)
    ci_lower = tau - 1.96 * se
    ci_upper = tau + 1.96 * se
    z_stat   = tau / se if se > 0 else np.nan
    p_value  = float(2 * stats.norm.sf(abs(z_stat))) if not np.isnan(z_stat) else np.nan

    return {
        "method":            "drdid",
        "ate":               round(tau, 4),
        "ci_lower":          round(ci_lower, 4),
        "ci_upper":          round(ci_upper, 4),
        "p_value":           round(p_value, 4) if not np.isnan(p_value) else None,
        "n_obs":             int(n),
        "covariate_adjusted": True,
        "engine":            "DR-DiD (logistic PS + Ridge outcome, Sant'anna & Zhao 2020)",
        "note":              "Consistent if either the PS model or the outcome regression is correctly specified.",
    }
