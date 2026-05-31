import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def _twfe_numpy(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    entity_col: str,
    time_col: str,
    covariates: list,
) -> tuple:
    """Two-way FE via within-transformation with optional covariate adjustment."""
    cols = [outcome, treatment, entity_col, time_col] + covariates
    d    = df[cols].dropna().copy()

    def _within(series, entity, time):
        grand  = series.mean()
        e_mean = d.groupby(entity)[series.name].transform("mean")
        t_mean = d.groupby(time)[series.name].transform("mean")
        return series - e_mean - t_mean + grand

    y_w = _within(d[outcome],    entity_col, time_col)
    t_w = _within(d[treatment],  entity_col, time_col)

    cov_cols_w = []
    for c in covariates:
        cw = d[c].copy()
        cw.name = c
        cov_cols_w.append(_within(cw, entity_col, time_col).values)

    X = np.column_stack([np.ones(len(t_w)), t_w.values] + cov_cols_w)
    y = y_w.values

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    ate           = beta[1]

    # Cluster-robust SE by entity
    residuals    = y - X @ beta
    n, k         = len(y), X.shape[1]
    entities     = d[entity_col].values
    unique_ents  = np.unique(entities)
    G            = len(unique_ents)
    meat         = np.zeros((k, k))
    for ent in unique_ents:
        m      = entities == ent
        score  = (X[m].T @ residuals[m].reshape(-1, 1))
        meat  += score @ score.T
    bread    = np.linalg.pinv(X.T @ X)
    sandwich = bread @ meat @ bread * G / max(G - 1, 1) * n / max(n - k, 1)
    se       = float(np.sqrt(sandwich[1, 1]))

    return float(ate), se, int(n), int(G)


def _parallel_trends_plot(
    df: pd.DataFrame, outcome: str, treatment: str, time_col: str
) -> Path:
    trends = df.groupby([time_col, treatment])[outcome].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    for val, label, color in [(1, "Treated", "#e07b54"), (0, "Control", "#4c72b0")]:
        sub = trends[trends[treatment] == val]
        ax.plot(sub[time_col], sub[outcome], marker="o", label=label, color=color)
    ax.set_xlabel(time_col)
    ax.set_ylabel(f"Mean {outcome}")
    ax.set_title("Parallel Trends Check")
    ax.legend()
    plt.tight_layout()
    out = Path(__file__).resolve().parent.parent.parent / "output" / "figures" / "parallel_trends.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _run_linearmodels(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    entity_col: str,
    time_col: str,
    covariates: list,
) -> dict | None:
    try:
        from linearmodels.panel import PanelOLS
        pdata    = df.set_index([entity_col, time_col])
        cov_str  = (" + " + " + ".join(covariates)) if covariates else ""
        formula  = f"{outcome} ~ {treatment}{cov_str} + EntityEffects + TimeEffects"
        mod      = PanelOLS.from_formula(formula, data=pdata)
        res      = mod.fit(cov_type="clustered", cluster_entity=True)
        ate      = float(res.params[treatment])
        se       = float(res.std_errors[treatment])
        ci       = res.conf_int().loc[treatment]
        return {
            "ate":      round(ate, 4),
            "se":       round(se, 4),
            "ci_lower": round(float(ci.iloc[0]), 4),
            "ci_upper": round(float(ci.iloc[1]), 4),
            "engine":   "linearmodels.PanelOLS",
        }
    except Exception:
        return None


def run_fixed_effects(df: pd.DataFrame, config: dict) -> dict:
    outcome    = config["outcome"]
    treatment  = config["treatment"]
    covariates = config.get("covariates", [])
    time_col   = config.get("data", {}).get("time_col")
    entity_col = config.get("data", {}).get("id_col")

    if not time_col or time_col not in df.columns:
        return {"method": "fixed_effects", "error": "time_col not specified or not found; FE requires panel data."}
    if not entity_col or entity_col not in df.columns:
        return {"method": "fixed_effects", "error": "id_col not specified or not found; FE requires panel data."}

    covs_in_data = [c for c in covariates if c in df.columns]
    plot_path    = _parallel_trends_plot(df, outcome, treatment, time_col)

    result = _run_linearmodels(df, outcome, treatment, entity_col, time_col, covs_in_data)
    if result:
        return {
            "method":             "fixed_effects",
            "ate":                result["ate"],
            "ci_lower":           result["ci_lower"],
            "ci_upper":           result["ci_upper"],
            "n_obs":              len(df[[outcome, treatment, entity_col, time_col]].dropna()),
            "engine":             result["engine"],
            "covariate_adjusted": bool(covs_in_data),
            "parallel_trends_plot": str(plot_path),
        }

    ate, se, n_obs, n_entities = _twfe_numpy(df, outcome, treatment, entity_col, time_col, covs_in_data)
    return {
        "method":             "fixed_effects",
        "ate":                round(ate, 4),
        "ci_lower":           round(ate - 1.96 * se, 4),
        "ci_upper":           round(ate + 1.96 * se, 4),
        "n_obs":              n_obs,
        "n_entities":         n_entities,
        "engine":             "within-transform (Python fallback)",
        "covariate_adjusted": bool(covs_in_data),
        "parallel_trends_plot": str(plot_path),
    }
