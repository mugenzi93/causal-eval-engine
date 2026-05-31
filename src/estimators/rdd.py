import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ── Bandwidth selection (Imbens-Kalyanaraman rule-of-thumb) ────────────────

def _ik_bandwidth(y: np.ndarray, x: np.ndarray, cutoff: float) -> float:
    """Simplified IK bandwidth selector using the rule-of-thumb."""
    n = len(x)
    h_pilot = 1.84 * np.std(x) * n ** (-1 / 5)
    left = (x >= cutoff - h_pilot) & (x < cutoff)
    right = (x >= cutoff) & (x <= cutoff + h_pilot)

    m2_left = np.polyfit(x[left] - cutoff, y[left], 2)[0] if left.sum() > 2 else 0.0
    m2_right = np.polyfit(x[right] - cutoff, y[right], 2)[0] if right.sum() > 2 else 0.0

    sigma2 = np.var(y - np.polyval(np.polyfit(x - cutoff, y, 1), x - cutoff))
    f_x = len(x[(x >= cutoff - h_pilot) & (x <= cutoff + h_pilot)]) / (2 * h_pilot * n)

    numerator = 3.4375 * sigma2 / f_x
    denominator = (m2_left - m2_right) ** 2 + 1e-8
    h = (numerator / denominator) ** (1 / 5) * n ** (-1 / 5)
    return max(h, h_pilot * 0.5)


# ── Local linear regression ────────────────────────────────────────────────

def _local_linear(
    y: np.ndarray, x: np.ndarray, cutoff: float, bandwidth: float
) -> tuple[float, float, float, float]:
    """Fit local linear on each side; return (tau, se, intercept_left, intercept_right)."""
    xc = x - cutoff
    mask_l = (xc >= -bandwidth) & (xc < 0)
    mask_r = (xc >= 0) & (xc <= bandwidth)

    def _fit(xv, yv):
        X = np.column_stack([np.ones(len(xv)), xv])
        beta, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
        yhat = X @ beta
        resid = yv - yhat
        sigma2 = (resid ** 2).sum() / max(len(yv) - 2, 1)
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        return beta[0], np.sqrt(cov[0, 0])

    if mask_l.sum() < 3 or mask_r.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan

    intercept_l, se_l = _fit(xc[mask_l], y[mask_l])
    intercept_r, se_r = _fit(xc[mask_r], y[mask_r])

    tau = intercept_r - intercept_l
    se = np.sqrt(se_l ** 2 + se_r ** 2)
    return tau, se, intercept_l, intercept_r


# ── RDD plot ───────────────────────────────────────────────────────────────

def _rdd_plot(
    y: np.ndarray,
    x: np.ndarray,
    cutoff: float,
    bandwidth: float,
    tau: float,
) -> Path:
    xc = x - cutoff
    mask = np.abs(xc) <= bandwidth * 1.5

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x[mask], y[mask], alpha=0.4, s=20, color="#888", label="Observations")

    for side, color, left_side in [("left", "#4c72b0", True), ("right", "#e07b54", False)]:
        m = ((xc < 0) if left_side else (xc >= 0)) & (np.abs(xc) <= bandwidth)
        if m.sum() < 3 or len(np.unique(x[m])) < 2:
            continue
        try:
            xs = np.linspace(x[m].min(), x[m].max(), 100)
            coef = np.polyfit(x[m] - cutoff, y[m], 1)
            ax.plot(xs, np.polyval(coef, xs - cutoff), color=color, linewidth=2)
        except np.linalg.LinAlgError:
            pass

    ax.axvline(cutoff, color="gray", linestyle="--", linewidth=1, label=f"Cutoff = {cutoff}")
    ax.set_xlabel("Running variable")
    ax.set_ylabel("Outcome")
    ax.set_title(f"RDD: Local Linear Fit  (τ = {tau:.3f}, bw = {bandwidth:.2f})")
    ax.legend()
    plt.tight_layout()

    out = Path(__file__).resolve().parent.parent.parent / "output" / "figures" / "rdd_plot.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── rpy2 wrapper (optional) ────────────────────────────────────────────────

def _run_rdrobust(y: np.ndarray, x: np.ndarray, cutoff: float) -> dict | None:
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
        numpy2ri.activate()
        rdrobust = importr("rdrobust")
        result = rdrobust.rdrobust(y, x, c=cutoff)
        coefs = np.array(result.rx2("coef"))
        ses = np.array(result.rx2("se"))
        ci = np.array(result.rx2("ci"))
        bw = float(np.array(result.rx2("bws"))[0])
        tau = float(coefs[0])
        se = float(ses[2])  # robust SE
        return {
            "tau": round(tau, 4),
            "se": round(se, 4),
            "ci_lower": round(float(ci[2, 0]), 4),
            "ci_upper": round(float(ci[2, 1]), 4),
            "bandwidth": round(bw, 4),
            "engine": "rdrobust (R)",
        }
    except Exception:
        return None


# ── Public entry point ─────────────────────────────────────────────────────

def run_rdd(df: pd.DataFrame, config: dict) -> dict:
    running_var = config.get("running_variable")
    cutoff = config.get("cutoff")
    outcome = config["outcome"]

    if not running_var or running_var not in df.columns:
        return {"method": "rdd", "error": "running_variable not specified or not found in data."}
    if cutoff is None:
        return {"method": "rdd", "error": "cutoff not specified in config."}

    # RDD is cross-sectional — collapse panel data to one row per unit
    # by averaging the outcome across time periods for each running-variable value
    id_col = config.get('data', {}).get('id_col')
    if id_col and id_col in df.columns:
        sub = df[[id_col, outcome, running_var]].dropna().groupby(id_col).mean().reset_index()
    else:
        sub = df[[outcome, running_var]].dropna().drop_duplicates(subset=[running_var])
    y = sub[outcome].values.astype(float)
    x = sub[running_var].values.astype(float)
    cutoff = float(cutoff)

    # Try R's rdrobust first
    r_result = _run_rdrobust(y, x, cutoff)
    if r_result:
        tau = r_result["tau"]
        bw = r_result["bandwidth"]
        plot_path = _rdd_plot(y, x, cutoff, bw, tau)
        return {
            "method": "rdd",
            "tau": r_result["tau"],
            "ci_lower": r_result["ci_lower"],
            "ci_upper": r_result["ci_upper"],
            "bandwidth": r_result["bandwidth"],
            "n_obs": len(sub),
            "engine": r_result["engine"],
            "rdd_plot": str(plot_path),
        }

    # Python fallback: local linear with IK bandwidth
    bw = _ik_bandwidth(y, x, cutoff)
    tau, se, _, _ = _local_linear(y, x, cutoff, bw)

    # If IK bandwidth is too tight for the sample, widen progressively
    if np.isnan(tau):
        for factor in [2.0, 4.0, (x.max() - x.min()) / 2]:
            bw = _ik_bandwidth(y, x, cutoff) * factor
            tau, se, _, _ = _local_linear(y, x, cutoff, bw)
            if not np.isnan(tau):
                break

    if np.isnan(tau):
        return {"method": "rdd", "error": "Insufficient observations near cutoff for local linear fit."}

    ci_lower = tau - 1.96 * se
    ci_upper = tau + 1.96 * se
    plot_path = _rdd_plot(y, x, cutoff, bw, tau)

    return {
        "method": "rdd",
        "tau": round(float(tau), 4),
        "ci_lower": round(float(ci_lower), 4),
        "ci_upper": round(float(ci_upper), 4),
        "bandwidth": round(float(bw), 4),
        "n_obs": len(sub),
        "engine": "local linear (Python fallback)",
        "rdd_plot": str(plot_path),
    }
