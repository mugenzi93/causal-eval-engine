import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def _fit_propensity(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="saga")),
    ])
    # Cross-fitted propensity scores to avoid overfitting bias
    ps = cross_val_predict(model, X, T, cv=5, method="predict_proba")[:, 1]
    # Trim extreme propensity scores (positivity)
    ps = np.clip(ps, 0.01, 0.99)
    return ps


def _fit_outcome_models(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted potential outcome predictions mu1(x) and mu0(x)."""
    from sklearn.model_selection import KFold

    mu1 = np.zeros(len(Y))
    mu0 = np.zeros(len(Y))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        T_train = T[train_idx]
        Y_train = Y[train_idx]

        for val, mu_arr in [(1, mu1), (0, mu0)]:
            mask = T_train == val
            if mask.sum() < 3:
                continue
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0)),
            ])
            model.fit(X_train[mask], Y_train[mask])
            mu_arr[test_idx] = model.predict(X_test)

    return mu1, mu0


def run_aipw(df: pd.DataFrame, config: dict) -> dict:
    outcome = config["outcome"]
    treatment = config["treatment"]
    covariates = config["covariates"]

    sub = df[[outcome, treatment] + covariates].dropna()
    Y = sub[outcome].values.astype(float)
    T = sub[treatment].values.astype(int)
    X = sub[covariates].values.astype(float)
    n = len(sub)

    ps = _fit_propensity(X, T)
    mu1, mu0 = _fit_outcome_models(X, T, Y)

    # AIPW / augmented IPW estimator
    # τ_AIPW = E[μ₁(X) - μ₀(X) + T/e(X)*(Y - μ₁(X)) - (1-T)/(1-e(X))*(Y - μ₀(X))]
    influence = (
        mu1 - mu0
        + T / ps * (Y - mu1)
        - (1 - T) / (1 - ps) * (Y - mu0)
    )
    ate = influence.mean()
    se = influence.std() / np.sqrt(n)
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se
    z_stat = ate / se if se > 0 else np.nan
    p_value = float(2 * stats.norm.sf(abs(z_stat))) if not np.isnan(z_stat) else np.nan

    return {
        "method": "aipw",
        "ate": round(float(ate), 4),
        "ci_lower": round(float(ci_lower), 4),
        "ci_upper": round(float(ci_upper), 4),
        "n_obs": int(n),
        "p_value": round(p_value, 4) if not np.isnan(p_value) else None,
        "note": "Doubly-robust AIPW with cross-fitted nuisance models (5-fold).",
    }
