import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FIGURES_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Color + marker scheme per adjustment method
_SCHEME_STYLE = {
    "Unadjusted":       {"color": "#d62728", "marker": "o"},
    "IPW":              {"color": "#1f77b4", "marker": "D"},
    "Overlap weights":  {"color": "#2ca02c", "marker": "s"},
    "Matching weights": {"color": "#9467bd", "marker": "^"},
}


# ── Propensity score fitting ────────────────────────────────────────────────

def _fit_propensity(df: pd.DataFrame, treatment: str, covariates: list) -> np.ndarray:
    X = df[covariates].fillna(df[covariates].median()).values
    y = df[treatment].astype(int).values
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="saga")),
    ])
    model.fit(X, y)
    return model.predict_proba(X)[:, 1]


# ── Weighted SMD computation ────────────────────────────────────────────────

def _compute_smd(
    df: pd.DataFrame,
    treatment: str,
    covariates: list,
    weights: pd.Series | None = None,
) -> pd.Series:
    """Compute absolute SMD per covariate, optionally weighted.

    Denominator always uses the unweighted pooled SD so all adjustment
    schemes are directly comparable on the same scale.
    """
    smds = {}
    for col in covariates:
        if col not in df.columns:
            continue
        mask_t = df[treatment] == 1
        mask_c = df[treatment] == 0
        t_vals = df.loc[mask_t, col].dropna()
        c_vals = df.loc[mask_c, col].dropna()

        # Unweighted pooled SD (used as denominator for all schemes)
        pooled_sd = np.sqrt((t_vals.std() ** 2 + c_vals.std() ** 2) / 2)
        if pooled_sd == 0 or np.isnan(pooled_sd):
            smds[col] = np.nan
            continue

        if weights is not None:
            wt = weights[mask_t].reindex(t_vals.index).fillna(0)
            wc = weights[mask_c].reindex(c_vals.index).fillna(0)
            if wt.sum() == 0 or wc.sum() == 0:
                smds[col] = np.nan
                continue
            mean_t = np.average(t_vals, weights=wt)
            mean_c = np.average(c_vals, weights=wc)
        else:
            mean_t = t_vals.mean()
            mean_c = c_vals.mean()

        smds[col] = (mean_t - mean_c) / pooled_sd

    return pd.Series(smds)


# ── Weight derivation ───────────────────────────────────────────────────────

def _ipw_weights(df: pd.DataFrame, treatment: str, ps: np.ndarray) -> pd.Series:
    ps_s = pd.Series(ps, index=df.index).clip(0.01, 0.99)
    w = pd.Series(0.0, index=df.index)
    w[df[treatment] == 1] = 1.0 / ps_s[df[treatment] == 1]
    w[df[treatment] == 0] = 1.0 / (1.0 - ps_s[df[treatment] == 0])
    return w


def _overlap_weights(df: pd.DataFrame, treatment: str, ps: np.ndarray) -> pd.Series:
    ps_s = pd.Series(ps, index=df.index).clip(0.01, 0.99)
    w = pd.Series(0.0, index=df.index)
    w[df[treatment] == 1] = 1.0 - ps_s[df[treatment] == 1]
    w[df[treatment] == 0] = ps_s[df[treatment] == 0]
    return w


# ── Table 1 ────────────────────────────────────────────────────────────────

def table_one(
    df: pd.DataFrame,
    treatment: str,
    covariates: list,
    weights: pd.Series | None = None,
) -> pd.DataFrame:
    rows = []
    for col in covariates:
        if col not in df.columns:
            continue
        t = df.loc[df[treatment] == 1, col].dropna()
        c = df.loc[df[treatment] == 0, col].dropna()
        wt = weights[df[treatment] == 1].dropna() if weights is not None else None
        wc = weights[df[treatment] == 0].dropna() if weights is not None else None

        if wt is not None:
            mean_t = np.average(t, weights=wt)
            mean_c = np.average(c, weights=wc)
            std_t = np.sqrt(np.average((t - mean_t) ** 2, weights=wt))
            std_c = np.sqrt(np.average((c - mean_c) ** 2, weights=wc))
        else:
            mean_t, std_t = t.mean(), t.std()
            mean_c, std_c = c.mean(), c.std()

        pooled_sd = np.sqrt((std_t ** 2 + std_c ** 2) / 2)
        smd = (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else np.nan

        rows.append({
            "Covariate": col,
            "Mean (Treated)": round(mean_t, 3),
            "SD (Treated)": round(std_t, 3),
            "Mean (Control)": round(mean_c, 3),
            "SD (Control)": round(std_c, 3),
            "SMD": round(smd, 3),
        })
    return pd.DataFrame(rows)


# ── Love Plot ───────────────────────────────────────────────────────────────

def love_plot(
    smd_dict: dict,
    threshold: float = 0.1,
    save_path: Path | None = None,
) -> Path:
    """Multi-scheme love plot.

    smd_dict: ordered dict of {scheme_label: pd.Series of SMDs}
    Schemes are plotted in insertion order; Unadjusted should come first.
    """
    # Align all series to the same covariate order (from Unadjusted)
    first_key = next(iter(smd_dict))
    covs = smd_dict[first_key].dropna().index.tolist()

    n_covs = len(covs)
    fig, ax = plt.subplots(figsize=(9, max(5, n_covs * 0.45)))

    y_pos = list(range(n_covs))

    for label, smd_series in smd_dict.items():
        style = _SCHEME_STYLE.get(label, {"color": "#888888", "marker": "o"})
        vals = smd_series.reindex(covs).abs().values
        ax.scatter(
            vals, y_pos,
            label=label,
            color=style["color"],
            marker=style["marker"],
            s=55,
            zorder=3,
            alpha=0.9,
        )

    # Connect Unadjusted → each adjusted scheme with faint lines per covariate
    if len(smd_dict) > 1:
        unadj = smd_dict[first_key].reindex(covs).abs().values
        for scheme_label, smd_series in list(smd_dict.items())[1:]:
            adj = smd_series.reindex(covs).abs().values
            style = _SCHEME_STYLE.get(scheme_label, {"color": "#888888"})
            for i in range(n_covs):
                if not (np.isnan(unadj[i]) or np.isnan(adj[i])):
                    ax.plot(
                        [unadj[i], adj[i]], [i, i],
                        color=style["color"], alpha=0.25, linewidth=0.9,
                    )

    ax.axvline(threshold, color="gray", linestyle="--", linewidth=1,
               label=f"|SMD| = {threshold} threshold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(covs, fontsize=9)
    ax.set_xlabel("Absolute Standardized Mean Difference")
    ax.set_title("Love Plot: Covariate Balance Before and After Adjustment")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()

    out = save_path or FIGURES_DIR / "love_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Density / Overlap Plots ─────────────────────────────────────────────────

def density_plots(
    df: pd.DataFrame,
    treatment: str,
    covariates: list,
    propensity_scores: pd.Series | None = None,
    save_dir: Path | None = None,
) -> list:
    out_dir = save_dir or FIGURES_DIR
    paths = []

    cols_to_plot = (["propensity_score"] if propensity_scores is not None else []) + covariates
    plot_df = df.copy()
    if propensity_scores is not None:
        plot_df["propensity_score"] = propensity_scores.values

    for col in cols_to_plot:
        if col not in plot_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, val, color in [("Treated", 1, "#e07b54"), ("Control", 0, "#4c72b0")]:
            subset = plot_df.loc[plot_df[treatment] == val, col].dropna()
            sns.kdeplot(subset, ax=ax, label=label, color=color, fill=True, alpha=0.35)
        ax.set_xlabel(col)
        ax.set_title(f"Distribution of {col}")
        ax.legend()
        plt.tight_layout()
        p = out_dir / f"density_{col}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    return paths


# ── ECDF Plots ───────────────────────────────────────────────────────────────

def ecdf_plots(
    df: pd.DataFrame,
    treatment: str,
    covariates: list,
    save_dir: Path | None = None,
) -> list:
    out_dir = save_dir or FIGURES_DIR
    paths = []

    for col in covariates:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, val, color in [("Treated", 1, "#e07b54"), ("Control", 0, "#4c72b0")]:
            x = np.sort(df.loc[df[treatment] == val, col].dropna().values)
            y = np.arange(1, len(x) + 1) / len(x)
            ax.step(x, y, label=label, color=color, where="post")
        ax.set_xlabel(col)
        ax.set_ylabel("Cumulative Probability")
        ax.set_title(f"ECDF: {col}")
        ax.legend()
        plt.tight_layout()
        p = out_dir / f"ecdf_{col}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    return paths


# ── Main diagnostics wrapper ─────────────────────────────────────────────────

def run_diagnostics(
    df: pd.DataFrame,
    treatment: str,
    covariates: list,
    propensity_scores: pd.Series | None = None,
    matching_weights: pd.Series | None = None,
    label: str = "pre",
) -> dict:
    # Table 1 uses raw (unadjusted) data
    t1 = table_one(df, treatment, covariates)

    # Fit propensity scores if not provided
    try:
        ps = propensity_scores.values if propensity_scores is not None else _fit_propensity(df, treatment, covariates)
        ps_series = pd.Series(ps, index=df.index)
    except Exception:
        ps_series = None

    # Build SMD dict in display order
    smd_dict = {}
    smd_dict["Unadjusted"] = _compute_smd(df, treatment, covariates)

    if ps_series is not None:
        ipw_w  = _ipw_weights(df, treatment, ps_series.values)
        olap_w = _overlap_weights(df, treatment, ps_series.values)
        smd_dict["IPW"]             = _compute_smd(df, treatment, covariates, weights=ipw_w)
        smd_dict["Overlap weights"] = _compute_smd(df, treatment, covariates, weights=olap_w)

    if matching_weights is not None:
        smd_dict["Matching weights"] = _compute_smd(df, treatment, covariates, weights=matching_weights)

    love_path = love_plot(smd_dict, save_path=FIGURES_DIR / f"love_plot_{label}.png")
    density_paths = density_plots(df, treatment, covariates, ps_series, save_dir=FIGURES_DIR)
    ecdf_paths    = ecdf_plots(df, treatment, covariates, save_dir=FIGURES_DIR)

    return {
        "table_one":     t1,
        "love_plot":     str(love_path),
        "density_plots": [str(p) for p in density_paths],
        "ecdf_plots":    [str(p) for p in ecdf_paths],
        "propensity_scores": ps_series,
    }
