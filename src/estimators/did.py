import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path


def run_did(df: pd.DataFrame, config: dict) -> dict:
    treatment = config["treatment"]
    outcome = config["outcome"]
    time_col = config.get("data", {}).get("time_col")

    if time_col is None or time_col not in df.columns:
        return {
            "method": "did",
            "error": "time_col not specified or not found in data; DiD requires panel data.",
        }

    times = sorted(df[time_col].unique())
    if len(times) < 2:
        return {"method": "did", "error": "DiD requires at least two time periods."}

    # Two-period DiD using first and last time point
    t_pre, t_post = times[0], times[-1]
    pre = df[df[time_col] == t_pre]
    post = df[df[time_col] == t_post]

    dd = (
        post.loc[post[treatment] == 1, outcome].mean()
        - post.loc[post[treatment] == 0, outcome].mean()
    ) - (
        pre.loc[pre[treatment] == 1, outcome].mean()
        - pre.loc[pre[treatment] == 0, outcome].mean()
    )

    # Pooled OLS regression version for SE
    df2 = df[df[time_col].isin([t_pre, t_post])].copy()
    df2["post"] = (df2[time_col] == t_post).astype(int)
    df2["interaction"] = df2[treatment] * df2["post"]

    X = df2[["post", treatment, "interaction"]].assign(const=1)[["const", "post", treatment, "interaction"]]
    y = df2[outcome]
    try:
        beta = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
        residuals = y.values - X.values @ beta
        sigma2 = (residuals ** 2).sum() / (len(y) - X.shape[1])
        cov = sigma2 * np.linalg.inv(X.values.T @ X.values)
        se_dd = np.sqrt(cov[3, 3])
    except np.linalg.LinAlgError:
        se_dd = np.nan

    ate = float(dd)
    ci_lower = ate - 1.96 * se_dd if not np.isnan(se_dd) else np.nan
    ci_upper = ate + 1.96 * se_dd if not np.isnan(se_dd) else np.nan
    dof = max(len(df2) - 4, 1)
    t_stat = ate / se_dd if (not np.isnan(se_dd) and se_dd > 0) else np.nan
    p_value = float(2 * stats.t.sf(abs(t_stat), df=dof)) if not np.isnan(t_stat) else np.nan

    # Event study plot (if more than 2 periods)
    event_path = None
    if len(times) > 2:
        event_path = _event_study_plot(df, treatment, outcome, time_col, t_pre)

    return {
        "method": "did",
        "ate": round(ate, 4),
        "ci_lower": round(float(ci_lower), 4) if not np.isnan(ci_lower) else None,
        "ci_upper": round(float(ci_upper), 4) if not np.isnan(ci_upper) else None,
        "p_value": round(p_value, 4) if not np.isnan(p_value) else None,
        "n_obs": len(df2),
        "event_study_plot": str(event_path) if event_path else None,
    }


def _event_study_plot(
    df: pd.DataFrame, treatment: str, outcome: str, time_col: str, base_period
) -> Path:
    times = sorted(df[time_col].unique())
    effects = []
    for t in times:
        yt = df.loc[(df[time_col] == t) & (df[treatment] == 1), outcome].mean()
        yc = df.loc[(df[time_col] == t) & (df[treatment] == 0), outcome].mean()
        effects.append(yt - yc)

    base_idx = times.index(base_period)
    base_effect = effects[base_idx]
    centered = [e - base_effect for e in effects]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, centered, marker="o", color="#4c72b0")
    ax.axvline(base_period, color="gray", linestyle="--", label="Base period")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel(time_col)
    ax.set_ylabel("Treatment - Control (centered)")
    ax.set_title("Event Study Plot")
    ax.legend()
    plt.tight_layout()

    out = Path(__file__).resolve().parent.parent.parent / "output" / "figures" / "event_study.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
