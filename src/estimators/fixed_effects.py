import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def _twfe_numpy(df: pd.DataFrame, outcome: str, treatment: str, entity_col: str, time_col: str) -> tuple:
    """Two-way fixed effects via within-transformation (demeaning)."""
    d = df[[outcome, treatment, entity_col, time_col]].dropna().copy()

    # Within-transform: subtract entity mean, time mean, add grand mean
    grand_mean_y = d[outcome].mean()
    grand_mean_t = d[treatment].mean()

    entity_mean_y = d.groupby(entity_col)[outcome].transform("mean")
    entity_mean_t = d.groupby(entity_col)[treatment].transform("mean")
    time_mean_y = d.groupby(time_col)[outcome].transform("mean")
    time_mean_t = d.groupby(time_col)[treatment].transform("mean")

    y_within = d[outcome] - entity_mean_y - time_mean_y + grand_mean_y
    t_within = d[treatment] - entity_mean_t - time_mean_t + grand_mean_t

    # OLS on demeaned data
    X = np.column_stack([np.ones(len(t_within)), t_within.values])
    y = y_within.values
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    ate = beta[1]

    residuals = y - X @ beta
    n = len(y)
    k = 2
    # Cluster-robust SE by entity
    entities = d[entity_col].values
    unique_entities = np.unique(entities)
    meat = np.zeros((k, k))
    for ent in unique_entities:
        mask = entities == ent
        Xe = X[mask]
        re = residuals[mask].reshape(-1, 1)
        score = Xe.T @ re
        meat += score @ score.T
    bread = np.linalg.pinv(X.T @ X)
    G = len(unique_entities)
    sandwich = bread @ meat @ bread * G / (G - 1) * n / (n - k)
    se = np.sqrt(sandwich[1, 1])

    return float(ate), float(se), int(n), int(G)


def _parallel_trends_plot(df: pd.DataFrame, outcome: str, treatment: str, time_col: str) -> Path:
    trends = (
        df.groupby([time_col, treatment])[outcome]
        .mean()
        .reset_index()
    )
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


def _run_linearmodels(df: pd.DataFrame, outcome: str, treatment: str, entity_col: str, time_col: str) -> dict | None:
    try:
        from linearmodels.panel import PanelOLS
        pdata = df.set_index([entity_col, time_col])
        mod = PanelOLS.from_formula(
            f"{outcome} ~ {treatment} + EntityEffects + TimeEffects",
            data=pdata,
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        ate = float(res.params[treatment])
        se = float(res.std_errors[treatment])
        ci = res.conf_int().loc[treatment]
        return {
            "ate": round(ate, 4),
            "se": round(se, 4),
            "ci_lower": round(float(ci.iloc[0]), 4),
            "ci_upper": round(float(ci.iloc[1]), 4),
            "engine": "linearmodels.PanelOLS",
        }
    except Exception:
        return None


def run_fixed_effects(df: pd.DataFrame, config: dict) -> dict:
    outcome = config["outcome"]
    treatment = config["treatment"]
    time_col = config.get("data", {}).get("time_col")
    entity_col = config.get("data", {}).get("id_col")

    if not time_col or time_col not in df.columns:
        return {"method": "fixed_effects", "error": "time_col not specified or not found; FE requires panel data."}
    if not entity_col or entity_col not in df.columns:
        return {"method": "fixed_effects", "error": "id_col not specified or not found; FE requires panel data."}

    plot_path = _parallel_trends_plot(df, outcome, treatment, time_col)

    result = _run_linearmodels(df, outcome, treatment, entity_col, time_col)
    if result:
        return {
            "method": "fixed_effects",
            "ate": result["ate"],
            "ci_lower": result["ci_lower"],
            "ci_upper": result["ci_upper"],
            "n_obs": len(df[[outcome, treatment, entity_col, time_col]].dropna()),
            "engine": result["engine"],
            "parallel_trends_plot": str(plot_path),
        }

    # NumPy fallback
    ate, se, n_obs, n_entities = _twfe_numpy(df, outcome, treatment, entity_col, time_col)
    return {
        "method": "fixed_effects",
        "ate": round(ate, 4),
        "ci_lower": round(ate - 1.96 * se, 4),
        "ci_upper": round(ate + 1.96 * se, 4),
        "n_obs": n_obs,
        "n_entities": n_entities,
        "engine": "within-transform (Python fallback)",
        "parallel_trends_plot": str(plot_path),
    }
