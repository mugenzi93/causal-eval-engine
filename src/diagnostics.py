import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


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
    smd_before: pd.Series,
    smd_after: pd.Series | None = None,
    threshold: float = 0.1,
    save_path: Path | None = None,
) -> Path:
    covs = smd_before.index.tolist()
    y = range(len(covs))

    fig, ax = plt.subplots(figsize=(8, max(4, len(covs) * 0.4)))
    ax.scatter(smd_before.abs().values, y, label="Before", marker="o", color="#e07b54", zorder=3)
    if smd_after is not None:
        ax.scatter(smd_after.abs().values, y, label="After", marker="D", color="#4c72b0", zorder=3)

    ax.axvline(threshold, color="gray", linestyle="--", linewidth=0.8, label=f"Threshold ({threshold})")
    ax.set_yticks(list(y))
    ax.set_yticklabels(covs, fontsize=9)
    ax.set_xlabel("Absolute Standardized Mean Difference")
    ax.set_title("Love Plot: Covariate Balance")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
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
) -> list[Path]:
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


# ── Convenience wrapper ──────────────────────────────────────────────────────

def run_diagnostics(
    df: pd.DataFrame,
    treatment: str,
    covariates: list,
    propensity_scores: pd.Series | None = None,
    weights_after: pd.Series | None = None,
    label: str = "pre",
) -> dict:
    t1_before = table_one(df, treatment, covariates)
    smd_before = t1_before.set_index("Covariate")["SMD"]

    smd_after = None
    if weights_after is not None:
        t1_after = table_one(df, treatment, covariates, weights=weights_after)
        smd_after = t1_after.set_index("Covariate")["SMD"]

    love_path = love_plot(smd_before, smd_after, save_path=FIGURES_DIR / f"love_plot_{label}.png")
    density_paths = density_plots(df, treatment, covariates, propensity_scores, save_dir=FIGURES_DIR)
    ecdf_paths = ecdf_plots(df, treatment, covariates, save_dir=FIGURES_DIR)

    return {
        "table_one": t1_before,
        "love_plot": str(love_path),
        "density_plots": [str(p) for p in density_paths],
        "ecdf_plots": [str(p) for p in ecdf_paths],
    }
