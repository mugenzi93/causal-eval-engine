import base64
from pathlib import Path
from datetime import datetime
import jinja2
import pandas as pd


TEMPLATE_DIR = Path(__file__).parent / "templates"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def _img_to_b64(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode()


def _df_to_html(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "<p><em>No data.</em></p>"
    return df.to_html(index=False, classes="table", border=0)

# ── Narrative generation ──────────────────────────────────────────────────────

_METHOD_CONTEXT = {
    "psm": (
        "ATE",
        "This PSM estimate compares matched treated and control units with similar "
        "propensity scores, isolating the effect of treatment from measured confounders."
    ),
    "did": (
        "ATE",
        "This DiD estimate captures the treatment effect by differencing out pre-existing "
        "trends, comparing how outcomes changed over time for treated versus control units."
    ),
    "iv": (
        "LATE",
        "This IV estimate is a Local Average Treatment Effect (LATE) — it applies only to "
        "compliers (units whose treatment status was shifted by the instrument), not the "
        "full population."
    ),
    "rdd": (
        "LATE",
        "This RDD estimate is a Local Average Treatment Effect at the cutoff — it applies "
        "only to units near the threshold and may not generalise to the broader population."
    ),
    "fixed_effects": (
        "ATE",
        "This two-way FE estimate removes all time-invariant unit-level confounders and "
        "common time shocks, identifying the effect from within-unit variation over time."
    ),
    "aipw": (
        "ATE",
        "This doubly-robust AIPW estimate is consistent if either the propensity model or "
        "the outcome model is correctly specified, and is semiparametrically efficient."
    ),
}


def _magnitude_label(abs_pct: float) -> str:
    if abs_pct >= 15:
        return "large"
    if abs_pct >= 7:
        return "moderate"
    if abs_pct >= 2:
        return "small"
    return "very small"


def generate_narrative(result: dict, outcome: str, treatment: str,
                       outcome_mean: float | None = None) -> str | None:
    """
    Return a 2-sentence plain-English interpretation of a causal estimate.
    Returns None if the result contains an error or no estimate.
    """
    if "error" in result or "skipped" in result:
        return None

    method   = result.get("method", "")
    est_key  = "ate" if "ate" in result else ("late" if "late" in result else "tau")
    est      = result.get(est_key)
    if est is None:
        return None

    ci_lower = result.get("ci_lower")
    ci_upper = result.get("ci_upper")

    # ── Direction sentence ────────────────────────────────────────────────────
    direction   = "increases" if est > 0 else "decreases"
    direction_n = "increase"  if est > 0 else "decrease"
    abs_est     = abs(est)
    sign_word   = "positive"  if est > 0 else "negative"

    if outcome_mean and outcome_mean != 0:
        pct = abs_est / abs(outcome_mean) * 100
        mag = _magnitude_label(pct)
        magnitude_str = f"{abs_est:,.1f} units ({pct:.1f}% of the mean {outcome})"
    else:
        mag = ""
        magnitude_str = f"{abs_est:,.2f} units"

    if ci_lower is not None and ci_upper is not None:
        sig      = (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0)
        sig_str  = "statistically significant (95% CI excludes zero)" if sig else                    "not statistically significant at the 5% level (95% CI includes zero)"
        ci_str   = f"[{ci_lower:,.1f}, {ci_upper:,.1f}]"
        sentence1 = (
            f"The estimate is {sign_word}: {treatment} {direction} {outcome} by "
            f"{magnitude_str}, and the effect is {sig_str} {ci_str}."
        )
    else:
        sentence1 = (
            f"The estimate is {sign_word}: {treatment} {direction} {outcome} by "
            f"{magnitude_str}."
        )

    # ── Method-context sentence ───────────────────────────────────────────────
    _, context = _METHOD_CONTEXT.get(method, ("ATE", ""))
    sentence2 = context

    # ── Magnitude qualifier (only when we have a reference mean) ─────────────
    if outcome_mean and mag:
        sentence2 += (
            f" The {direction_n} of {magnitude_str} represents a {mag} effect "
            f"relative to the baseline mean {outcome} of {outcome_mean:,.1f}."
        )

    return f"{sentence1} {sentence2}".strip()



def build_report(
    config: dict,
    diagnostics: dict,
    dag_info: dict,
    estimator_results: list[dict],
    sensitivity_results: list[dict],
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=True,
    )
    env.filters["b64img"] = lambda path: _img_to_b64(path) or ""
    template = env.get_template("report.html.j2")

    # Encode images as base64 for self-contained HTML
    love_b64 = _img_to_b64(diagnostics.get("love_plot"))
    dag_b64 = _img_to_b64(dag_info.get("dag_figure"))

    density_b64 = [
        _img_to_b64(p) for p in diagnostics.get("density_plots", []) if _img_to_b64(p)
    ]
    ecdf_b64 = [
        _img_to_b64(p) for p in diagnostics.get("ecdf_plots", []) if _img_to_b64(p)
    ]

    event_study_b64 = None
    for r in estimator_results:
        if r.get("method") == "did" and r.get("event_study_plot"):
            event_study_b64 = _img_to_b64(r["event_study_plot"])

    # Attach narrative to each estimator result
    outcome      = config.get("outcome", "outcome")
    treatment    = config.get("treatment", "treatment")
    outcome_mean = diagnostics.get("table_one", pd.DataFrame()).pipe(
        lambda df: None if df.empty else None   # outcome mean passed via run_eval
    )
    outcome_mean = diagnostics.get("outcome_mean")
    for r in estimator_results:
        if "narrative" not in r:
            r["narrative"] = generate_narrative(r, outcome, treatment, outcome_mean)

    rendered = template.render(
        title="Causal Evaluation Report",
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        config=config,
        table_one_html=_df_to_html(diagnostics.get("table_one")),
        love_plot_b64=love_b64,
        dag_b64=dag_b64,
        density_plots_b64=density_b64,
        ecdf_plots_b64=ecdf_b64,
        event_study_b64=event_study_b64,
        estimator_results=estimator_results,
        sensitivity_results=sensitivity_results,
        adjustment_set=dag_info.get("adjustment_set", []),
    )

    out = OUTPUT_DIR / "report.html"
    out.write_text(rendered, encoding="utf-8")
    return out
