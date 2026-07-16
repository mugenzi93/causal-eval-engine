import json
import anthropic

_STRIP_KEYS = {
    "propensity_scores",
    "matched_weights",
    "rdd_plot",
    "parallel_trends_plot",
    "event_study_plot",
}

SYSTEM_PROMPT = """You are a senior causal inference methodologist writing for a non-technical decision-maker audience — program managers, policy leads, and clinical directors who need to act on study results.

You will receive a JSON object containing:
- outcome: the name of the outcome variable
- treatment: the name of the treatment variable
- estimator_results: a list of estimation results from multiple causal methods
- sensitivity_results: E-value sensitivity analysis results

Your task is to interpret these results and return a JSON object — and ONLY a JSON object, no markdown fences, no preamble — with these exact keys:

{
  "consensus": <boolean — true if estimates broadly agree, false if there are meaningful conflicts>,
  "summary": <string — 2-3 sentences for a decision-maker. Use the actual treatment and outcome names. Translate numerical estimates into real-world terms (e.g., if ATE = -100 and outcome is total_cost_of_care, write "program enrollment is associated with approximately 00 in savings per member"). No statistical jargon.>,
  "actionable_insights": <list of 2-4 strings — concrete takeaways a program manager can act on, written in plain language using the treatment and outcome names>,
  "conflicts": <list of objects — each with keys "methods" (list of method names involved), "description" (plain-language description of the conflict with actual estimates), "likely_explanation" (methodological reason). Return an empty list if estimates are broadly consistent.>,
  "robustness": <string — plain-language E-value interpretation, e.g. "Even if there were an unmeasured factor that doubled both the likelihood of [treatment] and [outcome], the estimated effect would still hold." If no sensitivity results are present, say so.>,
  "caveats": <list of 2-4 strings — plain-language caveats about identifying assumptions or method-specific warnings>,
  "conclusion": <string — one punchy sentence a decision-maker can remember and act on>
}

Rules you must follow:
1. Always use the actual treatment and outcome variable names — never say "the treatment" or "the outcome" generically.
2. Translate estimates into domain meaning using sign and magnitude. Negative ATE on a cost outcome = savings; positive = increase. Say the amount explicitly.
3. Determine whether conflicts are real by checking whether 95% CIs overlap. Non-overlapping CIs = real conflict. Overlapping CIs = compatible estimates even if point estimates differ.
4. Methods sharing the same identifying assumption (e.g., DiD and DR-DiD both require parallel trends) should be grouped — their agreement is expected, not surprising.
5. Flag any result with an error field, weak_instrument_warning = true, or null estimate explicitly in caveats.
6. For E-values: interpret as "An unmeasured confounder would need to be associated with both [treatment] and [outcome] by a factor of at least [E-value] to explain away the result."
7. Return ONLY valid JSON. No markdown. No text outside the JSON object.
"""


def _clean_results(results: list[dict]) -> list[dict]:
    cleaned = []
    for r in results:
        cleaned.append({k: v for k, v in r.items() if k not in _STRIP_KEYS})
    return cleaned


def interpret_results(
    estimator_results: list[dict],
    sensitivity_results: list[dict],
    config: dict,
) -> dict | None:
    try:
        client = anthropic.Anthropic()
        payload = {
            "outcome": config.get("outcome", "outcome"),
            "treatment": config.get("treatment", "treatment"),
            "estimator_results": _clean_results(estimator_results),
            "sensitivity_results": sensitivity_results,
        }
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
        )
        return json.loads(response.content[0].text)
    except anthropic.AuthenticationError:
        print("      WARNING: ANTHROPIC_API_KEY missing or invalid — skipping AI interpretation.")
        return None
    except Exception as exc:
        print(f"      WARNING: AI interpretation failed ({exc}) — skipping.")
        return None
