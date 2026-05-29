import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FIGURES_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Colour + marker palette — one entry per weighting scheme
_SCHEME_STYLE = {
    "Unadjusted":      {"color": "#999999", "marker": "o", "zorder": 3},
    "IPW (ATE)":       {"color": "#4c72b0", "marker": "D", "zorder": 4},
    "IPW (ATT)":       {"color": "#e07b54", "marker": "s", "zorder": 5},
    "Overlap weights": {"color": "#2ca02c", "marker": "^", "zorder": 6},
    "Matching weights":{"color": "#9467bd", "marker": "P", "zorder": 7},
}


# ── Internal helpers ─────────────────────────────────────────────────────────

def _fit_propensity(df: pd.DataFrame, treatment: str, covariates: list) -> pd.Series:
    X = df[covariates].fillna(df[covariates].median())
    y = df[treatment]
    model = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=1.0))])
    model.fit(X, y)
    ps = pd.Series(model.predict_proba(X)[:, 1], index=df.index)
    return ps.clip(0.01, 0.99)


def _compute_smd_series(
    df: pd.DataFrame,
    treatment: str,
    covariates: list,
    weights: pd.Series | None = None,
) -> pd.Series:
    """Return |SMD| per covariate under an optional weight set."""
    smds = {}
    for col in covariates:
        if col not in df.columns:
            continue
        mask_t = df[treatment] == 1
        mask_c = df[treatment] == 0
        yt = df.loc[mask_t, col].dropna().values.astype(float)
        yc = df.loc[mask_c, col].dropna().values.astype(float)

        if weights is not None:
            wt = weights[mask_t].reindex(df[mask_t].index).fillna(0).values.astype(float)
            wc = weights[mask_c].reindex(df[mask_c].index).fillna(0).values.astype(float)
            # guard against zero-sum weights
            if wt.sum() == 0 or wc.sum() == 0:
                smds[col] = np.nan
                continue
            mean_t = np.average(yt, weights=wt)
            mean_c = np.average(yc, weights=wc)
            var_t = np.average((yt - mean_t) ** 2, weights=wt)
            var_c = np.average((yc - mean_c) ** 2, weights=wc)
        else:
            mean_t, var_t = yt.mean(), yt.var(ddof=1) if len(yt) > 1 else 0.0
            mean_c, var_c = yc.mean(), yc.var(ddof=1) if len(yc) > 1 else 0.0

        pooled_sd = np.sqrt((var_t + var_c) / 2)
        smds[col] = (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else np.nan

    return pd.Series(smds)


def _weight_sets(df: pd.DataFrame, treatment: str, ps: pd.Series) -> dict:
    """Derive the four standard weighting schemes from propensity scores."""
    t = df[treatment].values
    e = ps.values

    w_ipw_ate = pd.Series(np.where(t == 1, 1.0 / e, 1.0 / (1 - e)), index=df.index)
    w_ipw_att = pd.Series(np.where(t == 1, 1.0,     e / (1 - e)),    index=df.index)
    w_overlap  = pd.Series(np.where(t == 1, 1 - e,  e),               index=df.index)

    return {
        "IPW (ATE)":       w_ipw_ate,
        "IPW (ATT)":       w_ipw_att,
        "Overlap weights": w_overlap,
    }


# ── Table 1 ──────────────────────────────────────────────────────────────────

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
        mask_t = df[treatment] == 1
        mask_c = df[treatment] == 0
        t_vals = df.loc[mask_t, col].dropna()
        c_vals = df.loc[mask_c, col].dropna()
        wt = weights[mask_t].dropna() if weights is not None else None
        wc = weights[mask_c].dropna() if weights is not None else None

        if wt is not None:
            mean_t = np.average(t_vals, weights=wt)
            mean_c = np.average(c_vals, weights=wc)
            std_t = np.sqrt(np.average((t_vals - mean_t) ** 2, weights=wt))
            std_c = np.sqrt(np.average((c_vals - mean_c) ** 2, weights=wc))
        else:
            mean_t, std_t = t_vals.mean(), t_vals.std()
            mean_c, std_c = c_vals.mean(), c_vals.std()

        pooled_sd = np.sqrt((std_t ** 2 + std_c ** 2) / 2)
        smd = (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else np.nan
        rows.append({
            "Covariate":       col,
            "Mean (Treated)":  round(mean_t, 3),
            "SD (Treated)":    round(std_t, 3),
            "Mean (Control)":  round(mean_c, 3),
            "SD (Control)":    round(std_c, 3),
            "SMD":             round(smd, 3),
        })
    return pd.DataFrame(rows)


# ── Love Plot ─────────────────────────────────────────────────────────────────

def love_plot(
    smd_dict: dict,           # {scheme_label: pd.Series(covariate -> SMD)}
    threshold: float = 0.1,
    save_path: Path | None = None,
) -> Path:
    """
    Multi-series love plot.
    smd_dict keys must match _SCHEME_STYLE or will fall back to auto styling.
    The first key is assumed to be the unadjusted baseline.
    """
    # Align covariates across all series (use order from first entry)
    covs = list(next(iter(smd_dict.values())).index)
    y_pos = list(range(len(covs)))
    n_schemes = len(smd_dict)

    fig, ax = plt.subplots(figsize=(9, max(5, len(covs) * 0.55 + 1.5)))

    fallback_colors  = ["#999", "#4c72b0", "#e07b54", "#2ca02c", "#9467bd", "#8c564b"]
    fallback_markers = ["o", "D", "s", "^", "P", "*"]

    for i, (label, smd_series) in enumerate(smd_dict.items()):
        style = _SCHEME_STYLE.get(label, {
            "color":  fallback_colors[i % len(fallback_colors)],
            "marker": fallback_markers[i % len(fallback_markers)],
            "zorder": 3 + i,
        })
        vals = [abs(float(smd_series.get(c, np.nan))) for c in covs]
        ax.scatter(
            vals, y_pos,
            label=label,
            color=style["color"],
            marker=style["marker"],
            s=65,
            zorder=style["zorder"],
            edgecolors="white",
            linewidths=0.4,
        )

    # Connect dots across schemes for each covariate (light guide lines)
    for j, cov in enumerate(covs):
        x_vals = [abs(float(smd_dict[lbl].get(cov, np.nan))) for lbl in smd_dict]
        ax.plot(x_vals, [j] * len(x_vals), color="#cccccc", linewidth=0.8, zorder=2)

    ax.axvline(threshold, color="#555", linestyle="--", linewidth=0.9,
               label=f"Balance threshold (|SMD| = {threshold})")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(covs, fontsize=9)
    ax.set_xlabel("Absolute Standardized Mean Difference", fontsize=10)
    ax.set_title("Love Plot: Covariate Balance Across Weighting Schemes", fontsize=11, pad=10)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(left=0)
    plt.tight_layout()

    out = save_path or FIGURES_DIR / "love_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Density / Overlap Plots ───────────────────────────────────────────────────

def density_plots(
    df: pd.DataFrame,
    treatment: str,
    covariates: list,
    propensity_scores: pd.Series | None = None,
    save_dir: Path | None = None,
) -> list[Path]:
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
        for lbl, val, color in [("Treated", 1, "#e07b54"), ("Control", 0, "#4c72b0")]:
            subset = plot_df.loc[plot_df[treatment] == val, col].dropna()
            sns.kdeplot(subset, ax=ax, label=lbl, color=color, fill=True, alpha=0.35)
        ax.set_xlabel(col)
        ax.set_title(f"Distribution of {col}")
        ax.legend()
        plt.tight_layout()
        p = out_dir / f"density_{col}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


# ── ECDF Plots ────────────────────────────────────────────────────────────────

def ecdf_plots(
    df: pd.DataFrame,
    treatment: str,
    covariates: list,
    save_dir: Path | None = None,
) -> list[Path]:
    out_dir = save_dir or FIGURES_DIR
    paths = []
    for col in covariates:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for lbl, val, color in [("Treated", 1, "#e07b54"), ("Control", 0, "#4c72b0")]:
            x = np.sort(df.loc[df[treatment] == val, col].dropna().values)
            y = np.arange(1, len(x) + 1) / len(x)
            ax.step(x, y, label=lbl, color=color, where="post")
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


# ── Main diagnostic wrapper ───────────────────────────────────────────────────

def run_diagnostics(
    df: pd.DataFrame,
    treatment: str,
    covariates: list,
    propensity_scores: pd.Series | None = None,
    matched_weights: pd.Series | None = None,
    label: str = "pre",
) -> dict:
    """
    Run all diagnostics and produce a multi-scheme love plot.

    propensity_scores: pre-computed PS (e.g. from PSM). If None, fitted internally.
    matched_weights:   weight Series from PSM (1 = matched, 0 = unmatched).
                       When provided, adds a 'Matching weights' series to the love plot.
    """
    ps = propensity_scores if propensity_scores is not None else _fit_propensity(df, treatment, covariates)

    # Build ordered dict of SMD series — order controls legend / dot order
    smd_dict: dict[str, pd.Series] = {
        "Unadjusted": _compute_smd_series(df, treatment, covariates),
    }
    for scheme_label, w in _weight_sets(df, treatment, ps).items():
        smd_dict[scheme_label] = _compute_smd_series(df, treatment, covariates, w)
    if matched_weights is not None:
        smd_dict["Matching weights"] = _compute_smd_series(df, treatment, covariates, matched_weights)

    love_path     = love_plot(smd_dict, save_path=FIGURES_DIR / f"love_plot_{label}.png")
    t1            = table_one(df, treatment, covariates)
    density_paths = density_plots(df, treatment, covariates, ps, save_dir=FIGURES_DIR)
    ecdf_paths    = ecdf_plots(df, treatment, covariates, save_dir=FIGURES_DIR)

    return {
        "table_one":      t1,
        "love_plot":      str(love_path),
        "density_plots":  [str(p) for p in density_paths],
        "ecdf_plots":     [str(p) for p in ecdf_paths],
        "propensity_scores": ps,
    }


def update_love_plot(
    diagnostics: dict,
    df: pd.DataFrame,
    treatment: str,
    covariates: list,
    matched_weights: pd.Series,
    label: str = "final",
) -> None:
    """
    Regenerate the love plot in-place after PSM adds matching weights.
    Updates diagnostics['love_plot'] with the new path.
    """
    ps = diagnostics.get("propensity_scores")
    if ps is None:
        ps = _fit_propensity(df, treatment, covariates)

    smd_dict: dict[str, pd.Series] = {
        "Unadjusted": _compute_smd_series(df, treatment, covariates),
    }
    for scheme_label, w in _weight_sets(df, treatment, ps).items():
        smd_dict[scheme_label] = _compute_smd_series(df, treatment, covariates, w)
    smd_dict["Matching weights"] = _compute_smd_series(df, treatment, covariates, matched_weights)

    love_path = love_plot(smd_dict, save_path=FIGURES_DIR / f"love_plot_{label}.png")
    diagnostics["love_plot"] = str(love_path)
