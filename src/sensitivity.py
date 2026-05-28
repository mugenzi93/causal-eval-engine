import numpy as np
import pandas as pd
from scipy import stats


def evalue(ate: float, se: float, null_effect: float = 0.0) -> dict:
    """
    Compute the E-value for an observed ATE under an additive outcome model.

    The E-value is the minimum strength of association (on the risk ratio scale)
    that an unmeasured confounder would need to have with both treatment and outcome
    to fully explain away the observed effect.

    Uses the conversion from VanderWeele & Ding (2017) via the approximate
    formula for continuous outcomes: E = (|ATE - null| / SE)^2 / 2 + sqrt(...).
    For continuous outcomes we report the E-value based on the t-statistic.
    """
    t_stat = abs(ate - null_effect) / se if se > 0 else np.inf
    # Approximate E-value for continuous outcomes (Haneuse et al. 2019 approximation)
    eval = t_stat + np.sqrt(t_stat ** 2 - 1) if t_stat >= 1 else 1.0
    ci_eval = max(1.0, eval - 1.96)  # lower bound sensitivity

    return {
        "ate": round(ate, 4),
        "se": round(se, 4),
        "t_statistic": round(t_stat, 3),
        "evalue": round(float(eval), 3),
        "evalue_ci_lower": round(float(ci_eval), 3),
        "interpretation": (
            f"An unmeasured confounder associated with both treatment and outcome "
            f"by a factor of {eval:.2f} on a risk ratio scale would be needed "
            f"to fully explain away the observed effect."
        ),
    }


def compute_se_from_result(result: dict) -> float | None:
    """Extract or approximate SE from an estimator result dict."""
    ci_l = result.get("ci_lower")
    ci_u = result.get("ci_upper")
    if ci_l is not None and ci_u is not None:
        return (ci_u - ci_l) / (2 * 1.96)
    return None


def run_sensitivity(results: list[dict]) -> list[dict]:
    sensitivity_outputs = []
    for res in results:
        method = res.get("method", "unknown")
        if "error" in res:
            sensitivity_outputs.append({"method": method, "skipped": True, "reason": res["error"]})
            continue

        ate_key = "ate" if "ate" in res else "late"
        ate = res.get(ate_key)
        se = compute_se_from_result(res)

        if ate is None or se is None or se == 0:
            sensitivity_outputs.append({"method": method, "skipped": True, "reason": "Insufficient SE info."})
            continue

        ev = evalue(float(ate), float(se))
        ev["method"] = method
        sensitivity_outputs.append(ev)

    return sensitivity_outputs
